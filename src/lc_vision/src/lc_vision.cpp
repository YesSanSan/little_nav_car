#include "lc_vision.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

#include <astra/astra.hpp>
#include <foxglove_msgs/msg/compressed_video.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/qos.hpp>

#include "rclcpp_components/register_node_macro.hpp"

namespace lc_vision {

namespace {

using namespace std::chrono_literals;

std::string avErrorToString(const int error_code) {
    char buffer[AV_ERROR_MAX_STRING_SIZE] = {0};
    av_strerror(error_code, buffer, sizeof(buffer));
    return std::string(buffer);
}

std::string expandHomeDirectory(const std::string &path) {
    if (path.empty() || path[0] != '~') {
        return path;
    }

    const char *home = std::getenv("HOME");
    if (home == nullptr) {
        return path;
    }

    if (path.size() == 1) {
        return std::string(home);
    }

    if (path[1] == '/') {
        return std::string(home) + path.substr(1);
    }

    return path;
}

builtin_interfaces::msg::Time toBuiltinTime(const rclcpp::Time &stamp) {
    builtin_interfaces::msg::Time msg;
    const int64_t                 total_nanoseconds = stamp.nanoseconds();
    msg.sec = static_cast<int32_t>(total_nanoseconds / 1000000000LL);
    msg.nanosec = static_cast<uint32_t>(total_nanoseconds % 1000000000LL);
    return msg;
}

std::vector<uint8_t> extractAnnexBExtradata(const AVCodecContext *codec_context) {
    if (codec_context == nullptr || codec_context->extradata == nullptr || codec_context->extradata_size <= 0) {
        return {};
    }

    const auto *extradata = reinterpret_cast<const uint8_t *>(codec_context->extradata);
    const int   size      = codec_context->extradata_size;

    if (size >= 4 && extradata[0] == 0x00 && extradata[1] == 0x00 &&
        ((extradata[2] == 0x01) || (extradata[2] == 0x00 && extradata[3] == 0x01))) {
        return std::vector<uint8_t>(extradata, extradata + size);
    }

    if (size < 7 || extradata[0] != 0x01) {
        return std::vector<uint8_t>(extradata, extradata + size);
    }

    std::vector<uint8_t> annexb;
    size_t               offset = 5;

    const auto append_nalus = [&](const uint8_t count, size_t &cursor, std::vector<uint8_t> &output) {
        for (uint8_t i = 0; i < count; ++i) {
            if (cursor + 2 > static_cast<size_t>(size)) {
                throw std::runtime_error("Invalid H.264 extradata while parsing NAL length");
            }

            const uint16_t nal_size = static_cast<uint16_t>(extradata[cursor] << 8U | extradata[cursor + 1]);
            cursor += 2;

            if (cursor + nal_size > static_cast<size_t>(size)) {
                throw std::runtime_error("Invalid H.264 extradata while parsing NAL payload");
            }

            output.insert(output.end(), {0x00, 0x00, 0x00, 0x01});
            output.insert(output.end(), extradata + cursor, extradata + cursor + nal_size);
            cursor += nal_size;
        }
    };

    const uint8_t sps_count = static_cast<uint8_t>(extradata[offset] & 0x1F);
    ++offset;
    append_nalus(sps_count, offset, annexb);

    if (offset >= static_cast<size_t>(size)) {
        return annexb;
    }

    const uint8_t pps_count = extradata[offset];
    ++offset;
    append_nalus(pps_count, offset, annexb);

    return annexb;
}

class VideoEncoder {
public:
    VideoEncoder() = default;

    ~VideoEncoder() {
        reset();
    }

    void initialize(
        const std::string &stream_name, const int width, const int height, const int fps,
        const int bitrate_kbps, const int gop_size, const AVPixelFormat input_format) {
        reset();

        stream_name_    = stream_name;
        width_          = width;
        height_         = height;
        fps_            = std::max(1, fps);
        bitrate_bps_    = std::max(1, bitrate_kbps) * 1000;
        gop_size_       = std::max(1, gop_size);
        input_format_   = input_format;
        frame_counter_  = 0;

        codec_ = avcodec_find_encoder_by_name("libx264");
        if (codec_ == nullptr) {
            codec_ = avcodec_find_encoder(AV_CODEC_ID_H264);
        }
        if (codec_ == nullptr) {
            throw std::runtime_error("No H.264 encoder is available in FFmpeg");
        }

        codec_context_ = avcodec_alloc_context3(codec_);
        if (codec_context_ == nullptr) {
            throw std::runtime_error("Failed to allocate FFmpeg codec context");
        }

        codec_context_->codec_type       = AVMEDIA_TYPE_VIDEO;
        codec_context_->codec_id         = codec_->id;
        codec_context_->width            = width_;
        codec_context_->height           = height_;
        codec_context_->time_base        = AVRational{1, fps_};
        codec_context_->framerate        = AVRational{fps_, 1};
        codec_context_->pix_fmt          = AV_PIX_FMT_YUV420P;
        codec_context_->bit_rate         = bitrate_bps_;
        codec_context_->gop_size         = gop_size_;
        codec_context_->max_b_frames     = 0;
        codec_context_->thread_count     = 1;
        codec_context_->thread_type      = FF_THREAD_SLICE;
        codec_context_->flags           |= AV_CODEC_FLAG_LOW_DELAY;

        av_opt_set(codec_context_->priv_data, "profile", "baseline", 0);
        av_opt_set(codec_context_->priv_data, "preset", "ultrafast", 0);
        av_opt_set(codec_context_->priv_data, "tune", "zerolatency", 0);
        av_opt_set(codec_context_->priv_data, "repeat-headers", "1", 0);
        av_opt_set(codec_context_->priv_data, "annexb", "1", 0);

        const int open_result = avcodec_open2(codec_context_, codec_, nullptr);
        if (open_result < 0) {
            throw std::runtime_error("Failed to open encoder for " + stream_name_ + ": " + avErrorToString(open_result));
        }

        frame_ = av_frame_alloc();
        if (frame_ == nullptr) {
            throw std::runtime_error("Failed to allocate FFmpeg frame for " + stream_name_);
        }

        frame_->format = codec_context_->pix_fmt;
        frame_->width  = width_;
        frame_->height = height_;

        const int buffer_result = av_frame_get_buffer(frame_, 32);
        if (buffer_result < 0) {
            throw std::runtime_error("Failed to allocate encoder frame buffer for " + stream_name_ + ": " +
                                     avErrorToString(buffer_result));
        }

        packet_ = av_packet_alloc();
        if (packet_ == nullptr) {
            throw std::runtime_error("Failed to allocate FFmpeg packet for " + stream_name_);
        }

        sws_context_ = sws_getCachedContext(
            nullptr, width_, height_, input_format_, width_, height_, codec_context_->pix_fmt,
            SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
        if (sws_context_ == nullptr) {
            throw std::runtime_error("Failed to create swscale context for " + stream_name_);
        }

        annexb_extradata_ = extractAnnexBExtradata(codec_context_);
    }

    [[nodiscard]] std::optional<std::vector<uint8_t>> encode(const uint8_t *data, const int stride_bytes) {
        if (codec_context_ == nullptr || frame_ == nullptr || packet_ == nullptr || sws_context_ == nullptr) {
            throw std::runtime_error("Encoder is not initialized for " + stream_name_);
        }

        if (data == nullptr) {
            return std::nullopt;
        }

        const uint8_t *src_data[4]   = {data, nullptr, nullptr, nullptr};
        int            src_lines[4]  = {stride_bytes, 0, 0, 0};

        const int writable_result = av_frame_make_writable(frame_);
        if (writable_result < 0) {
            throw std::runtime_error("Failed to make encoder frame writable for " + stream_name_ + ": " +
                                     avErrorToString(writable_result));
        }

        sws_scale(sws_context_, src_data, src_lines, 0, height_, frame_->data, frame_->linesize);
        frame_->pts = frame_counter_++;

        const int send_result = avcodec_send_frame(codec_context_, frame_);
        if (send_result < 0) {
            throw std::runtime_error("Failed to send frame to encoder for " + stream_name_ + ": " +
                                     avErrorToString(send_result));
        }

        std::vector<uint8_t> encoded_frame;

        while (true) {
            const int receive_result = avcodec_receive_packet(codec_context_, packet_);
            if (receive_result == AVERROR(EAGAIN) || receive_result == AVERROR_EOF) {
                break;
            }
            if (receive_result < 0) {
                throw std::runtime_error("Failed to receive packet from encoder for " + stream_name_ + ": " +
                                         avErrorToString(receive_result));
            }

            if ((packet_->flags & AV_PKT_FLAG_KEY) != 0 && !annexb_extradata_.empty()) {
                encoded_frame.insert(encoded_frame.end(), annexb_extradata_.begin(), annexb_extradata_.end());
            }
            encoded_frame.insert(encoded_frame.end(), packet_->data, packet_->data + packet_->size);
            av_packet_unref(packet_);
        }

        if (encoded_frame.empty()) {
            return std::nullopt;
        }

        return encoded_frame;
    }

    void reset() {
        if (codec_context_ != nullptr) {
            avcodec_send_frame(codec_context_, nullptr);
            while (packet_ != nullptr && avcodec_receive_packet(codec_context_, packet_) == 0) {
                av_packet_unref(packet_);
            }
        }

        if (sws_context_ != nullptr) {
            sws_freeContext(sws_context_);
            sws_context_ = nullptr;
        }

        if (packet_ != nullptr) {
            av_packet_free(&packet_);
        }

        if (frame_ != nullptr) {
            av_frame_free(&frame_);
        }

        if (codec_context_ != nullptr) {
            avcodec_free_context(&codec_context_);
        }

        codec_ = nullptr;
        annexb_extradata_.clear();
        stream_name_.clear();
        frame_counter_ = 0;
    }

private:
    const AVCodec  *codec_         = nullptr;
    AVCodecContext *codec_context_ = nullptr;
    AVFrame        *frame_         = nullptr;
    AVPacket       *packet_        = nullptr;
    SwsContext     *sws_context_   = nullptr;

    std::vector<uint8_t> annexb_extradata_;
    std::string          stream_name_;
    AVPixelFormat        input_format_  = AV_PIX_FMT_NONE;
    int                  width_         = 0;
    int                  height_        = 0;
    int                  fps_           = 0;
    int                  bitrate_bps_   = 0;
    int                  gop_size_      = 0;
    int64_t              frame_counter_ = 0;
};

} // namespace

template<typename StreamT>
astra::ImageStreamMode selectBestMode(
    StreamT &stream, const int requested_width, const int requested_height, const int requested_fps,
    const astra_pixel_format_t requested_format) {
    const auto available_modes = stream.available_modes();

    if (available_modes.empty()) {
        astra::ImageStreamMode fallback;
        fallback.set_width(static_cast<uint32_t>(requested_width));
        fallback.set_height(static_cast<uint32_t>(requested_height));
        fallback.set_fps(static_cast<uint8_t>(requested_fps));
        fallback.set_pixel_format(requested_format);
        return fallback;
    }

    auto best_mode = available_modes.front();
    int  best_cost = std::numeric_limits<int>::max();

    for (const auto &mode : available_modes) {
        int cost = 0;
        if (mode.pixel_format() != requested_format) {
            cost += 1000000;
        }

        cost += std::abs(static_cast<int>(mode.width()) - requested_width) * 1000;
        cost += std::abs(static_cast<int>(mode.height()) - requested_height) * 1000;
        cost += std::abs(static_cast<int>(mode.fps()) - requested_fps);

        if (cost < best_cost) {
            best_cost = cost;
            best_mode = mode;
        }
    }

    return best_mode;
}

struct LCVision::Impl {
    struct StreamConfig {
        bool        enable      = true;
        int         width       = 640;
        int         height      = 480;
        int         capture_fps = 15;
        int         publish_fps = 10;
        int         bitrate_kbps = 1200;
        int         gop_size    = 30;
        std::string frame_id;
    };

    struct DepthConfig : StreamConfig {
        int  visualization_min_mm = 300;
        int  visualization_max_mm = 5000;
        bool invalid_as_black     = true;
    };

    template<typename T>
    struct FrameBuffer {
        std::mutex      mutex;
        std::vector<T>  data;
        int             width       = 0;
        int             height      = 0;
        int64_t         frame_index = -1;
        rclcpp::Time    stamp;
        bool            available   = false;
    };

    class CaptureListener : public astra::FrameListener {
    public:
        explicit CaptureListener(Impl &owner)
            : owner_(owner) {
        }

        void on_frame_ready(astra::StreamReader &, astra::Frame &frame) override {
            const auto color_frame = frame.get<astra::ColorFrame>();
            if (color_frame.is_valid()) {
                owner_.storeColorFrame(color_frame);
            }

            const auto depth_frame = frame.get<astra::DepthFrame>();
            if (depth_frame.is_valid()) {
                owner_.storeDepthFrame(depth_frame);
            }
        }

    private:
        Impl &owner_;
    };

    explicit Impl(LCVision &node)
        : node(node),
          capture_listener(std::make_unique<CaptureListener>(*this)) {
    }

    void storeColorFrame(const astra::ColorFrame &frame) {
        if (!rgb.enable) {
            return;
        }

        std::lock_guard<std::mutex> lock(rgb_buffer.mutex);
        rgb_buffer.width       = frame.width();
        rgb_buffer.height      = frame.height();
        rgb_buffer.frame_index = static_cast<int64_t>(frame.frame_index());
        rgb_buffer.stamp       = node.now();
        rgb_buffer.available   = true;
        rgb_buffer.data.resize(static_cast<size_t>(frame.length()) * 3U);
        frame.copy_to(reinterpret_cast<astra::RgbPixel *>(rgb_buffer.data.data()));
    }

    void storeDepthFrame(const astra::DepthFrame &frame) {
        if (!depth.enable) {
            return;
        }

        std::lock_guard<std::mutex> lock(depth_buffer.mutex);
        depth_buffer.width       = frame.width();
        depth_buffer.height      = frame.height();
        depth_buffer.frame_index = static_cast<int64_t>(frame.frame_index());
        depth_buffer.stamp       = node.now();
        depth_buffer.available   = true;
        depth_buffer.data.resize(frame.length());
        frame.copy_to(depth_buffer.data.data());
    }

    [[nodiscard]] std::optional<foxglove_msgs::msg::CompressedVideo> makeRgbMessage() {
        std::vector<uint8_t> rgb_data;
        rclcpp::Time         stamp;
        int                  width       = 0;
        int                  height      = 0;
        int64_t              frame_index = -1;

        {
            std::lock_guard<std::mutex> lock(rgb_buffer.mutex);
            if (!rgb_buffer.available || rgb_buffer.frame_index == last_rgb_frame_index) {
                return std::nullopt;
            }

            rgb_data    = rgb_buffer.data;
            stamp       = rgb_buffer.stamp;
            width       = rgb_buffer.width;
            height      = rgb_buffer.height;
            frame_index = rgb_buffer.frame_index;
        }

        if (width <= 0 || height <= 0 || rgb_data.empty()) {
            return std::nullopt;
        }

        if (rgb_encoder_width != width || rgb_encoder_height != height) {
            rgb_encoder.initialize("rgb", width, height, rgb.publish_fps, rgb.bitrate_kbps, rgb.gop_size, AV_PIX_FMT_RGB24);
            rgb_encoder_width = width;
            rgb_encoder_height = height;
            RCLCPP_INFO(
                node.get_logger(),
                "Initialized RGB encoder: %dx%d @ %d fps, %d kbps, GOP %d",
                width, height, rgb.publish_fps, rgb.bitrate_kbps, rgb.gop_size);
        }

        const auto encoded = rgb_encoder.encode(rgb_data.data(), width * 3);
        if (!encoded.has_value()) {
            return std::nullopt;
        }

        foxglove_msgs::msg::CompressedVideo msg;
        msg.timestamp = toBuiltinTime(stamp);
        msg.frame_id  = rgb.frame_id;
        msg.format    = "h264";
        msg.data      = *encoded;

        last_rgb_frame_index = frame_index;
        return msg;
    }

    [[nodiscard]] std::optional<foxglove_msgs::msg::CompressedVideo> makeDepthMessage() {
        std::vector<int16_t> depth_data;
        rclcpp::Time         stamp;
        int                  width       = 0;
        int                  height      = 0;
        int64_t              frame_index = -1;

        {
            std::lock_guard<std::mutex> lock(depth_buffer.mutex);
            if (!depth_buffer.available || depth_buffer.frame_index == last_depth_frame_index) {
                return std::nullopt;
            }

            depth_data   = depth_buffer.data;
            stamp        = depth_buffer.stamp;
            width        = depth_buffer.width;
            height       = depth_buffer.height;
            frame_index  = depth_buffer.frame_index;
        }

        if (width <= 0 || height <= 0 || depth_data.empty()) {
            return std::nullopt;
        }

        depth_visualization.resize(static_cast<size_t>(width) * static_cast<size_t>(height));

        const float min_mm = static_cast<float>(depth.visualization_min_mm);
        const float max_mm = static_cast<float>(std::max(depth.visualization_max_mm, depth.visualization_min_mm + 1));
        const float range  = max_mm - min_mm;

        for (size_t i = 0; i < depth_data.size(); ++i) {
            const int16_t depth_mm = depth_data[i];
            if (depth_mm <= 0) {
                depth_visualization[i] = depth.invalid_as_black ? 0 : 255;
                continue;
            }

            const float clamped = std::clamp(static_cast<float>(depth_mm), min_mm, max_mm);
            const float normalized = 1.0f - ((clamped - min_mm) / range);
            depth_visualization[i] = static_cast<uint8_t>(std::clamp(std::lround(normalized * 255.0f), 0L, 255L));
        }

        if (depth_encoder_width != width || depth_encoder_height != height) {
            depth_encoder.initialize(
                "depth", width, height, depth.publish_fps, depth.bitrate_kbps, depth.gop_size, AV_PIX_FMT_GRAY8);
            depth_encoder_width = width;
            depth_encoder_height = height;
            RCLCPP_INFO(
                node.get_logger(),
                "Initialized depth encoder: %dx%d @ %d fps, %d kbps, GOP %d",
                width, height, depth.publish_fps, depth.bitrate_kbps, depth.gop_size);
        }

        const auto encoded = depth_encoder.encode(depth_visualization.data(), width);
        if (!encoded.has_value()) {
            return std::nullopt;
        }

        foxglove_msgs::msg::CompressedVideo msg;
        msg.timestamp = toBuiltinTime(stamp);
        msg.frame_id  = depth.frame_id;
        msg.format    = "h264";
        msg.data      = *encoded;

        last_depth_frame_index = frame_index;
        return msg;
    }

    LCVision &node;

    std::string sdk_root;
    StreamConfig rgb;
    DepthConfig  depth;

    astra::StreamSet    stream_set;
    astra::StreamReader reader;

    FrameBuffer<uint8_t> rgb_buffer;
    FrameBuffer<int16_t> depth_buffer;

    VideoEncoder rgb_encoder;
    VideoEncoder depth_encoder;

    std::vector<uint8_t> depth_visualization;

    std::unique_ptr<CaptureListener> capture_listener;

    rclcpp::Publisher<foxglove_msgs::msg::CompressedVideo>::SharedPtr rgb_video_pub;
    rclcpp::Publisher<foxglove_msgs::msg::CompressedVideo>::SharedPtr depth_video_pub;

    std::thread capture_thread;
    std::thread publish_thread;

    std::atomic<bool> running{false};
    bool              astra_initialized = false;

    int64_t last_rgb_frame_index   = -1;
    int64_t last_depth_frame_index = -1;
    int     rgb_encoder_width      = 0;
    int     rgb_encoder_height     = 0;
    int     depth_encoder_width    = 0;
    int     depth_encoder_height   = 0;
};

LCVision::LCVision(const rclcpp::NodeOptions &options)
    : Node("lc_vision", options),
      impl_(std::make_unique<Impl>(*this)) {
    RCLCPP_INFO(get_logger(), "Start LCVision!");

    try {
        getParams();
        prepareModel();

        nav_to_pose_ = rclcpp_action::create_client<NavigateToPose>(this, "navigate_to_pose");

        if (!impl_->rgb.enable && !impl_->depth.enable) {
            throw std::runtime_error("Both rgb.enable and depth.enable are false; nothing to publish");
        }

        astra::initialize();
        impl_->astra_initialized = true;
        impl_->stream_set        = astra::StreamSet();
        impl_->reader            = impl_->stream_set.create_reader();

        if (!impl_->reader.is_valid()) {
            throw std::runtime_error("Failed to create Astra stream reader");
        }

        impl_->reader.add_listener(*impl_->capture_listener);

        if (impl_->rgb.enable) {
            auto color_stream = impl_->reader.stream<astra::ColorStream>();
            if (!color_stream.is_available()) {
                throw std::runtime_error("Astra color stream is not available");
            }

            color_stream.enable_mirroring(false);
            const auto rgb_mode = selectBestMode(
                color_stream, impl_->rgb.width, impl_->rgb.height, impl_->rgb.capture_fps, ASTRA_PIXEL_FORMAT_RGB888);
            color_stream.set_mode(rgb_mode);
            color_stream.start();

            const auto active_mode = color_stream.mode();
            RCLCPP_INFO(
                get_logger(),
                "Configured RGB stream: %ux%u @ %u fps (mirroring disabled)",
                active_mode.width(), active_mode.height(), active_mode.fps());

            auto video_qos = rclcpp::QoS(rclcpp::KeepLast(5));
            video_qos.reliable();
            impl_->rgb_video_pub = create_publisher<foxglove_msgs::msg::CompressedVideo>("~/rgb/video", video_qos);
        }

        if (impl_->depth.enable) {
            auto depth_stream = impl_->reader.stream<astra::DepthStream>();
            if (!depth_stream.is_available()) {
                throw std::runtime_error("Astra depth stream is not available");
            }

            depth_stream.enable_mirroring(false);
            const auto depth_mode = selectBestMode(
                depth_stream, impl_->depth.width, impl_->depth.height, impl_->depth.capture_fps,
                ASTRA_PIXEL_FORMAT_DEPTH_MM);
            depth_stream.set_mode(depth_mode);
            depth_stream.start();

            const auto active_mode = depth_stream.mode();
            RCLCPP_INFO(
                get_logger(),
                "Configured depth stream: %ux%u @ %u fps (mirroring disabled)",
                active_mode.width(), active_mode.height(), active_mode.fps());

            auto video_qos = rclcpp::QoS(rclcpp::KeepLast(5));
            video_qos.reliable();
            impl_->depth_video_pub = create_publisher<foxglove_msgs::msg::CompressedVideo>("~/depth/video", video_qos);
        }

        impl_->running.store(true);

        impl_->capture_thread = std::thread([this]() {
            while (rclcpp::ok() && impl_->running.load()) {
                try {
                    astra_update();
                    std::this_thread::sleep_for(1ms);
                } catch (const std::exception &ex) {
                    RCLCPP_ERROR(get_logger(), "Astra capture loop failed: %s", ex.what());
                    impl_->running.store(false);
                }
            }
        });

        impl_->publish_thread = std::thread([this]() {
            auto next_rgb_publish_time   = std::chrono::steady_clock::now();
            auto next_depth_publish_time = std::chrono::steady_clock::now();
            const auto rgb_period =
                std::chrono::milliseconds(std::max(1, 1000 / std::max(1, impl_->rgb.publish_fps)));
            const auto depth_period =
                std::chrono::milliseconds(std::max(1, 1000 / std::max(1, impl_->depth.publish_fps)));

            while (rclcpp::ok() && impl_->running.load()) {
                const auto now = std::chrono::steady_clock::now();

                if (impl_->rgb.enable && now >= next_rgb_publish_time) {
                    next_rgb_publish_time = now + rgb_period;
                    try {
                        const auto msg = impl_->makeRgbMessage();
                        if (msg.has_value()) {
                            impl_->rgb_video_pub->publish(*msg);
                        }
                    } catch (const std::exception &ex) {
                        RCLCPP_ERROR(get_logger(), "RGB video encoding failed: %s", ex.what());
                    }
                }

                if (impl_->depth.enable && now >= next_depth_publish_time) {
                    next_depth_publish_time = now + depth_period;
                    try {
                        const auto msg = impl_->makeDepthMessage();
                        if (msg.has_value()) {
                            impl_->depth_video_pub->publish(*msg);
                        }
                    } catch (const std::exception &ex) {
                        RCLCPP_ERROR(get_logger(), "Depth video encoding failed: %s", ex.what());
                    }
                }

                std::this_thread::sleep_for(2ms);
            }
        });

        RCLCPP_INFO(get_logger(), "Astra SDK root: %s", impl_->sdk_root.c_str());
    } catch (...) {
        impl_->running.store(false);
        if (impl_->publish_thread.joinable()) {
            impl_->publish_thread.join();
        }
        if (impl_->capture_thread.joinable()) {
            impl_->capture_thread.join();
        }
        impl_->rgb_encoder.reset();
        impl_->depth_encoder.reset();
        if (impl_->astra_initialized) {
            astra::terminate();
            impl_->astra_initialized = false;
        }
        throw;
    }
}

LCVision::~LCVision() {
    impl_->running.store(false);

    if (impl_->publish_thread.joinable()) {
        impl_->publish_thread.join();
    }

    if (impl_->capture_thread.joinable()) {
        impl_->capture_thread.join();
    }

    try {
        if (impl_->reader.is_valid() && impl_->capture_listener != nullptr) {
            impl_->reader.remove_listener(*impl_->capture_listener);
        }
    } catch (const std::exception &ex) {
        RCLCPP_WARN(get_logger(), "Failed to remove Astra listener cleanly: %s", ex.what());
    }

    try {
        if (impl_->reader.is_valid() && impl_->rgb.enable) {
            auto color_stream = impl_->reader.stream<astra::ColorStream>();
            if (color_stream.is_available()) {
                color_stream.stop();
            }
        }
    } catch (const std::exception &ex) {
        RCLCPP_WARN(get_logger(), "Failed to stop Astra RGB stream cleanly: %s", ex.what());
    }

    try {
        if (impl_->reader.is_valid() && impl_->depth.enable) {
            auto depth_stream = impl_->reader.stream<astra::DepthStream>();
            if (depth_stream.is_available()) {
                depth_stream.stop();
            }
        }
    } catch (const std::exception &ex) {
        RCLCPP_WARN(get_logger(), "Failed to stop Astra depth stream cleanly: %s", ex.what());
    }

    impl_->rgb_encoder.reset();
    impl_->depth_encoder.reset();
    if (impl_->astra_initialized) {
        astra::terminate();
        impl_->astra_initialized = false;
    }
}

void LCVision::getParams() {
    impl_->sdk_root = expandHomeDirectory(this->declare_parameter<std::string>(
        "sdk_root",
        "/home/betty/AstraSDK-v2.1.3-Linux-arm/AstraSDK-v2.1.3-94bca0f52e-20210611T023312Z-Linux-aarch64"));

    impl_->rgb.enable       = this->declare_parameter<bool>("rgb.enable", true);
    impl_->rgb.width        = this->declare_parameter<int>("rgb.width", 640);
    impl_->rgb.height       = this->declare_parameter<int>("rgb.height", 480);
    impl_->rgb.capture_fps  = this->declare_parameter<int>("rgb.capture_fps", 15);
    impl_->rgb.publish_fps  = this->declare_parameter<int>("rgb.publish_fps", 10);
    impl_->rgb.bitrate_kbps = this->declare_parameter<int>("rgb.bitrate_kbps", 1200);
    impl_->rgb.gop_size     = this->declare_parameter<int>("rgb.gop_size", 30);
    impl_->rgb.frame_id     = this->declare_parameter<std::string>("rgb.frame_id", "camera_color_optical_frame");

    impl_->depth.enable               = this->declare_parameter<bool>("depth.enable", true);
    impl_->depth.width                = this->declare_parameter<int>("depth.width", 640);
    impl_->depth.height               = this->declare_parameter<int>("depth.height", 480);
    impl_->depth.capture_fps          = this->declare_parameter<int>("depth.capture_fps", 15);
    impl_->depth.publish_fps          = this->declare_parameter<int>("depth.publish_fps", 8);
    impl_->depth.bitrate_kbps         = this->declare_parameter<int>("depth.bitrate_kbps", 500);
    impl_->depth.gop_size             = this->declare_parameter<int>("depth.gop_size", 24);
    impl_->depth.frame_id             = this->declare_parameter<std::string>("depth.frame_id", "camera_depth_optical_frame");
    impl_->depth.visualization_min_mm = this->declare_parameter<int>("depth.visualization_min_mm", 300);
    impl_->depth.visualization_max_mm = this->declare_parameter<int>("depth.visualization_max_mm", 5000);
    impl_->depth.invalid_as_black     = this->declare_parameter<bool>("depth.invalid_as_black", true);

    RCLCPP_INFO(
        get_logger(),
        "RGB config: enable=%s capture=%dx%d@%d publish=%d bitrate=%dkbps gop=%d",
        impl_->rgb.enable ? "true" : "false", impl_->rgb.width, impl_->rgb.height, impl_->rgb.capture_fps,
        impl_->rgb.publish_fps, impl_->rgb.bitrate_kbps, impl_->rgb.gop_size);
    RCLCPP_INFO(
        get_logger(),
        "Depth config: enable=%s capture=%dx%d@%d publish=%d bitrate=%dkbps gop=%d vis=[%d,%d]mm",
        impl_->depth.enable ? "true" : "false", impl_->depth.width, impl_->depth.height, impl_->depth.capture_fps,
        impl_->depth.publish_fps, impl_->depth.bitrate_kbps, impl_->depth.gop_size,
        impl_->depth.visualization_min_mm, impl_->depth.visualization_max_mm);
}

void LCVision::prepareModel() {
    RCLCPP_INFO(get_logger(), "Vision recognition pipeline is currently disabled; camera/video framework only.");
}

void LCVision::sendGoal(const geometry_msgs::msg::PoseStamped &goal) {
    if (!nav_to_pose_->wait_for_action_server(500ms)) {
        RCLCPP_WARN(get_logger(), "navigate_to_pose action server is not available yet");
        return;
    }

    NavigateToPose::Goal nav_goal;
    nav_goal.pose = goal;
    nav_to_pose_->async_send_goal(nav_goal);
}

} // namespace lc_vision

RCLCPP_COMPONENTS_REGISTER_NODE(lc_vision::LCVision)
