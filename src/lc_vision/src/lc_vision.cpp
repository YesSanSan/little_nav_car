#include "lc_vision.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <regex>
#include <set>
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

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <astra/astra.hpp>
#include <foxglove_msgs/msg/compressed_video.hpp>
#if LC_VISION_HAVE_NCNN
#include <gpu.h>
#include <net.h>
#endif
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/qos.hpp>

#include "rclcpp_components/register_node_macro.hpp"

namespace lc_vision {

namespace {

using namespace std::chrono_literals;
namespace fs = std::filesystem;

struct ParsedMetadata {
    int                      input_size = 0;
    std::vector<std::string> class_names;
};

struct Detection {
    cv::Rect    box;
    float       score = 0.0f;
    int         class_id = -1;
    std::string label;
};

std::string avErrorToString(const int error_code) {
    char buffer[AV_ERROR_MAX_STRING_SIZE] = {0};
    av_strerror(error_code, buffer, sizeof(buffer));
    return std::string(buffer);
}

std::string trim(const std::string &value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string unquote(const std::string &value) {
    const auto trimmed = trim(value);
    if (trimmed.size() >= 2 &&
        ((trimmed.front() == '"' && trimmed.back() == '"') || (trimmed.front() == '\'' && trimmed.back() == '\''))) {
        return trimmed.substr(1, trimmed.size() - 2);
    }
    return trimmed;
}

int parseFirstInteger(const std::string &line) {
    static const std::regex kIntegerRegex(R"((-?\d+))");
    std::smatch              match;
    if (!std::regex_search(line, match, kIntegerRegex)) {
        return 0;
    }
    return std::stoi(match.str(1));
}

std::vector<std::string> defaultCocoClassNames() {
    return {
        "person",        "bicycle",      "car",           "motorcycle",    "airplane",      "bus",
        "train",         "truck",        "boat",          "traffic light", "fire hydrant",  "stop sign",
        "parking meter", "bench",        "bird",          "cat",           "dog",           "horse",
        "sheep",         "cow",          "elephant",      "bear",          "zebra",         "giraffe",
        "backpack",      "umbrella",     "handbag",       "tie",           "suitcase",      "frisbee",
        "skis",          "snowboard",    "sports ball",   "kite",          "baseball bat",  "baseball glove",
        "skateboard",    "surfboard",    "tennis racket", "bottle",        "wine glass",    "cup",
        "fork",          "knife",        "spoon",         "bowl",          "banana",        "apple",
        "sandwich",      "orange",       "broccoli",      "carrot",        "hot dog",       "pizza",
        "donut",         "cake",         "chair",         "couch",         "potted plant",  "bed",
        "dining table",  "toilet",       "tv",            "laptop",        "mouse",         "remote",
        "keyboard",      "cell phone",   "microwave",     "oven",          "toaster",       "sink",
        "refrigerator",  "book",         "clock",         "vase",          "scissors",      "teddy bear",
        "hair drier",    "toothbrush"};
}

std::string defaultAstraSdkRoot() {
    if (const char *env_value = std::getenv("ASTRA_SDK_ROOT"); env_value != nullptr && env_value[0] != '\0') {
        return std::string(env_value);
    }
    return "/home/cmls/sanwu/AstraSDK-v2.1.3-Ubuntu-x86_64/"
           "AstraSDK-v2.1.3-94bca0f52e-20210608T062039Z-Ubuntu18.04-x86_64";
}

int defaultDetectorThreadCount() {
    const auto concurrency = std::thread::hardware_concurrency();
    if (concurrency == 0U) {
        return 2;
    }
    return std::max(1, std::min(2, static_cast<int>(concurrency)));
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

ParsedMetadata parseMetadataFile(const fs::path &metadata_path) {
    ParsedMetadata result;
    std::ifstream  input(metadata_path);
    if (!input.is_open()) {
        return result;
    }

    bool in_names_block = false;
    for (std::string line; std::getline(input, line);) {
        const auto trimmed = trim(line);
        if (trimmed.empty() || trimmed.starts_with('#')) {
            continue;
        }

        if (trimmed.starts_with("imgsz:")) {
            const int parsed = parseFirstInteger(trimmed);
            if (parsed > 0) {
                result.input_size = parsed;
            }
            in_names_block = false;
            continue;
        }

        if (trimmed.starts_with("names:")) {
            const auto inline_names = trim(trimmed.substr(std::string("names:").size()));
            if (!inline_names.empty() && inline_names.front() == '{' && inline_names.back() == '}') {
                static const std::regex item_regex(R"((\d+)\s*:\s*['"]?([^,'"}]+)['"]?)");
                for (auto it = std::sregex_iterator(inline_names.begin(), inline_names.end(), item_regex);
                     it != std::sregex_iterator();
                     ++it) {
                    const int index = std::stoi((*it)[1].str());
                    if (index >= static_cast<int>(result.class_names.size())) {
                        result.class_names.resize(static_cast<size_t>(index) + 1U);
                    }
                    result.class_names[static_cast<size_t>(index)] = trim((*it)[2].str());
                }
                in_names_block = false;
            } else {
                in_names_block = true;
            }
            continue;
        }

        if (!in_names_block) {
            continue;
        }

        std::smatch match;
        if (std::regex_match(trimmed, match, std::regex(R"((\d+)\s*:\s*(.+))"))) {
            const int         index = std::stoi(match.str(1));
            const std::string name  = unquote(match.str(2));
            if (index >= static_cast<int>(result.class_names.size())) {
                result.class_names.resize(static_cast<size_t>(index) + 1U);
            }
            result.class_names[static_cast<size_t>(index)] = name;
            continue;
        }

        in_names_block = false;
    }

    return result;
}

float intersectionOverUnion(const cv::Rect2f &a, const cv::Rect2f &b) {
    const float x1 = std::max(a.x, b.x);
    const float y1 = std::max(a.y, b.y);
    const float x2 = std::min(a.x + a.width, b.x + b.width);
    const float y2 = std::min(a.y + a.height, b.y + b.height);

    const float intersection_w = std::max(0.0f, x2 - x1);
    const float intersection_h = std::max(0.0f, y2 - y1);
    const float intersection   = intersection_w * intersection_h;
    const float union_area     = a.area() + b.area() - intersection;

    if (union_area <= 0.0f) {
        return 0.0f;
    }
    return intersection / union_area;
}

void applyNms(std::vector<Detection> &detections, const float iou_threshold) {
    std::sort(detections.begin(), detections.end(), [](const Detection &lhs, const Detection &rhs) {
        return lhs.score > rhs.score;
    });

    std::vector<Detection> filtered;
    std::vector<bool>      suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }

        filtered.push_back(detections[i]);
        const cv::Rect2f lhs_rect(
            static_cast<float>(detections[i].box.x), static_cast<float>(detections[i].box.y),
            static_cast<float>(detections[i].box.width), static_cast<float>(detections[i].box.height));

        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j] || detections[i].class_id != detections[j].class_id) {
                continue;
            }

            const cv::Rect2f rhs_rect(
                static_cast<float>(detections[j].box.x), static_cast<float>(detections[j].box.y),
                static_cast<float>(detections[j].box.width), static_cast<float>(detections[j].box.height));

            if (intersectionOverUnion(lhs_rect, rhs_rect) > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    detections = std::move(filtered);
}

void drawDetections(cv::Mat &image, const std::vector<Detection> &detections, const cv::Size &source_size) {
    if (image.empty() || source_size.width <= 0 || source_size.height <= 0) {
        return;
    }

    const float scale_x = static_cast<float>(image.cols) / static_cast<float>(source_size.width);
    const float scale_y = static_cast<float>(image.rows) / static_cast<float>(source_size.height);
    const int   thickness = std::max(1, std::min(image.cols, image.rows) / 240);
    const int   font_thickness = std::max(1, thickness);
    const double font_scale = std::max(0.45, std::min(image.cols, image.rows) / 800.0);

    for (const auto &detection : detections) {
        cv::Rect scaled_box(
            static_cast<int>(std::lround(static_cast<float>(detection.box.x) * scale_x)),
            static_cast<int>(std::lround(static_cast<float>(detection.box.y) * scale_y)),
            static_cast<int>(std::lround(static_cast<float>(detection.box.width) * scale_x)),
            static_cast<int>(std::lround(static_cast<float>(detection.box.height) * scale_y)));
        scaled_box &= cv::Rect(0, 0, image.cols, image.rows);

        if (scaled_box.width <= 1 || scaled_box.height <= 1) {
            continue;
        }

        const std::string caption =
            detection.label + " " + cv::format("%.2f", static_cast<double>(detection.score));

        int       baseline = 0;
        const auto text_size =
            cv::getTextSize(caption, cv::FONT_HERSHEY_SIMPLEX, font_scale, font_thickness, &baseline);
        const int text_x = std::max(0, scaled_box.x);
        const int text_y = std::max(text_size.height + baseline + 4, scaled_box.y);
        const int bg_y   = text_y - text_size.height - baseline - 4;
        const int bg_w   = std::min(text_size.width + 8, image.cols - text_x);
        const int bg_h   = text_size.height + baseline + 8;

        cv::rectangle(image, scaled_box, cv::Scalar(0, 255, 0), thickness, cv::LINE_AA);
        cv::rectangle(image, cv::Rect(text_x, bg_y, bg_w, bg_h), cv::Scalar(0, 96, 0), cv::FILLED);
        cv::putText(
            image, caption, cv::Point(text_x + 4, text_y - 4), cv::FONT_HERSHEY_SIMPLEX, font_scale,
            cv::Scalar(255, 255, 255), font_thickness, cv::LINE_AA);
    }
}

class VideoEncoder {
public:
    VideoEncoder() = default;

    ~VideoEncoder() {
        reset();
    }

    void initialize(
        const std::string &stream_name, const int width, const int height, const int fps, const int bitrate_kbps,
        const int gop_size, const AVPixelFormat input_format) {
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
            nullptr, width_, height_, input_format_, width_, height_, codec_context_->pix_fmt, SWS_FAST_BILINEAR,
            nullptr, nullptr, nullptr);
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

        const uint8_t *src_data[4] = {data, nullptr, nullptr, nullptr};
        int            src_lines[4] = {stride_bytes, 0, 0, 0};

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
    AVPixelFormat        input_format_ = AV_PIX_FMT_NONE;
    int                  width_        = 0;
    int                  height_       = 0;
    int                  fps_          = 0;
    int                  bitrate_bps_  = 0;
    int                  gop_size_     = 0;
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
        bool        enable       = true;
        int         width        = 640;
        int         height       = 480;
        int         capture_fps  = 15;
        int         publish_fps  = 10;
        int         bitrate_kbps = 1200;
        int         gop_size     = 30;
        std::string frame_id;
    };

    struct DepthConfig : StreamConfig {
        int  visualization_min_mm = 300;
        int  visualization_max_mm = 5000;
        bool invalid_as_black     = true;
    };

    struct DetectorConfig {
        bool                     enable           = true;
        std::string              backend          = "auto";
        std::string              model_dir;
        int                      input_size       = 640;
        float                    score_threshold  = 0.25f;
        float                    nms_threshold    = 0.45f;
        int                      fps              = 5;
        int                      vulkan_device_index = 0;
        int                      cpu_num_threads     = defaultDetectorThreadCount();
        bool                     log_backend_info    = true;
        std::vector<std::string> target_classes   = {"person"};
        std::vector<std::string> class_names      = defaultCocoClassNames();
        std::set<int>            target_class_ids = {0};
    };

    template<typename T>
    struct FrameBuffer {
        std::mutex     mutex;
        std::vector<T> data;
        int            width       = 0;
        int            height      = 0;
        int64_t        frame_index = -1;
        rclcpp::Time   stamp;
        bool           available = false;
    };

    struct DetectionCache {
        std::mutex           mutex;
        std::vector<Detection> detections;
        int                  width       = 0;
        int                  height      = 0;
        int64_t              frame_index = -1;
        rclcpp::Time         stamp;
        bool                 available = false;
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

    fs::path resolveModelDirectory() const {
        fs::path model_path = fs::path(expandHomeDirectory(detector.model_dir));
        if (model_path.is_absolute()) {
            return model_path;
        }

        try {
            const auto package_share = fs::path(ament_index_cpp::get_package_share_directory("lc_vision"));
            const auto installed_path = package_share / model_path;
            if (fs::exists(installed_path)) {
                return installed_path;
            }
        } catch (...) {
        }

        const fs::path source_package_dir = fs::path(__FILE__).parent_path().parent_path();
        const fs::path source_path        = source_package_dir / model_path;
        if (fs::exists(source_path)) {
            return source_path;
        }

        return model_path;
    }

    void updateTargetClassIds() {
        detector.target_class_ids.clear();

        std::set<std::string> requested;
        for (const auto &name : detector.target_classes) {
            requested.insert(toLower(name));
        }

        for (size_t i = 0; i < detector.class_names.size(); ++i) {
            if (requested.contains(toLower(detector.class_names[i]))) {
                detector.target_class_ids.insert(static_cast<int>(i));
            }
        }

        if (detector.target_class_ids.empty() && requested.contains("person") && !detector.class_names.empty()) {
            detector.target_class_ids.insert(0);
        }
    }

    [[nodiscard]] bool prefersVulkan() const {
        const auto normalized = toLower(detector.backend);
        return normalized == "auto" || normalized == "vulkan";
    }

    [[nodiscard]] bool forcesCpuOnly() const {
        return toLower(detector.backend) == "cpu";
    }

    [[nodiscard]] bool backendValueIsKnown() const {
        const auto normalized = toLower(detector.backend);
        return normalized == "auto" || normalized == "vulkan" || normalized == "cpu";
    }

    void cleanupGpu() {
#if LC_VISION_HAVE_NCNN
        detector_net.clear();
#if NCNN_VULKAN
        if (gpu_instance_created) {
            ncnn::destroy_gpu_instance();
            gpu_instance_created = false;
        }
#endif
#else
        gpu_instance_created = false;
#endif
        using_vulkan_backend = false;
        active_backend       = "cpu";
        detected_gpu_count   = 0;
        active_gpu_index     = -1;
    }

    void logBackendSelection(const char *message) const {
        if (!detector.log_backend_info) {
            return;
        }

        RCLCPP_INFO(
            node.get_logger(), "%s backend=%s gpu_count=%d gpu_index=%d cpu_threads=%d", message,
            active_backend.c_str(), detected_gpu_count, active_gpu_index, detector.cpu_num_threads);
    }

    void configureDetectorBackend() {
#if !LC_VISION_HAVE_NCNN
        detector_runtime_enabled.store(false);
        detector_ready = false;
        active_backend = "disabled";
        return;
#else
        detector_net.opt.num_threads        = std::max(1, detector.cpu_num_threads);
        detector_net.opt.use_vulkan_compute = false;
        using_vulkan_backend                = false;
        active_backend                      = "cpu";
        detected_gpu_count                  = 0;
        active_gpu_index                    = -1;

        if (!backendValueIsKnown()) {
            RCLCPP_WARN(
                node.get_logger(),
                "Unknown detector.backend value '%s', defaulting to CPU fallback behavior.",
                detector.backend.c_str());
        }

        if (forcesCpuOnly()) {
            logBackendSelection("NCNN detector configured");
            return;
        }

#if NCNN_VULKAN
        if (!prefersVulkan()) {
            logBackendSelection("NCNN detector configured");
            return;
        }

        if (ncnn::create_gpu_instance() != 0) {
            RCLCPP_WARN(node.get_logger(), "Failed to create NCNN Vulkan instance, falling back to CPU.");
            logBackendSelection("NCNN detector configured");
            return;
        }

        gpu_instance_created = true;
        detected_gpu_count   = ncnn::get_gpu_count();
        if (detected_gpu_count <= 0) {
            RCLCPP_WARN(node.get_logger(), "No Vulkan-capable NCNN GPU found, falling back to CPU.");
            cleanupGpu();
            logBackendSelection("NCNN detector configured");
            return;
        }

        active_gpu_index = std::clamp(detector.vulkan_device_index, 0, detected_gpu_count - 1);
        detector_net.opt.use_vulkan_compute = true;
        detector_net.set_vulkan_device(active_gpu_index);
        using_vulkan_backend = true;
        active_backend       = "vulkan";
        logBackendSelection("NCNN detector configured");
#else
        if (prefersVulkan()) {
            RCLCPP_WARN(
                node.get_logger(),
                "This NCNN build does not include Vulkan support, falling back to CPU backend.");
        }
        logBackendSelection("NCNN detector configured");
#endif
#endif
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

    void updateDetectionCache(
        std::vector<Detection> detections, const int width, const int height, const int64_t frame_index,
        const rclcpp::Time &stamp) {
        std::lock_guard<std::mutex> lock(detection_cache.mutex);
        detection_cache.detections  = std::move(detections);
        detection_cache.width       = width;
        detection_cache.height      = height;
        detection_cache.frame_index = frame_index;
        detection_cache.stamp       = stamp;
        detection_cache.available   = true;
    }

    std::vector<Detection> snapshotDetections(cv::Size &source_size) {
        std::lock_guard<std::mutex> lock(detection_cache.mutex);
        source_size = cv::Size(detection_cache.width, detection_cache.height);
        return detection_cache.detections;
    }

    bool copyLatestRgbFrame(std::vector<uint8_t> &rgb_data, int &width, int &height, int64_t &frame_index, rclcpp::Time &stamp) {
        std::lock_guard<std::mutex> lock(rgb_buffer.mutex);
        if (!rgb_buffer.available || rgb_buffer.frame_index == last_detection_input_frame_index) {
            return false;
        }

        rgb_data     = rgb_buffer.data;
        width        = rgb_buffer.width;
        height       = rgb_buffer.height;
        frame_index  = rgb_buffer.frame_index;
        stamp        = rgb_buffer.stamp;
        return !rgb_data.empty() && width > 0 && height > 0;
    }

    std::vector<Detection> runDetection(
        const std::vector<uint8_t> &rgb_data, const int width, const int height, const int input_size) {
#if !LC_VISION_HAVE_NCNN
        (void)rgb_data;
        (void)width;
        (void)height;
        (void)input_size;
        return {};
#else
        if (!detector_ready || rgb_data.empty() || width <= 0 || height <= 0) {
            return {};
        }

        cv::Mat rgb(height, width, CV_8UC3, const_cast<uint8_t *>(rgb_data.data()));

        const float scale = std::min(
            static_cast<float>(input_size) / static_cast<float>(width),
            static_cast<float>(input_size) / static_cast<float>(height));
        const int resized_w = std::max(1, static_cast<int>(std::lround(static_cast<float>(width) * scale)));
        const int resized_h = std::max(1, static_cast<int>(std::lround(static_cast<float>(height) * scale)));
        const int pad_w     = std::max(0, input_size - resized_w);
        const int pad_h     = std::max(0, input_size - resized_h);
        const int pad_left  = pad_w / 2;
        const int pad_top   = pad_h / 2;

        cv::Mat resized;
        cv::resize(rgb, resized, cv::Size(resized_w, resized_h), 0.0, 0.0, cv::INTER_LINEAR);

        cv::Mat padded(input_size, input_size, CV_8UC3, cv::Scalar(114, 114, 114));
        resized.copyTo(padded(cv::Rect(pad_left, pad_top, resized_w, resized_h)));

        ncnn::Mat input = ncnn::Mat::from_pixels(padded.data, ncnn::Mat::PIXEL_RGB, input_size, input_size);
        const float norm_vals[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
        input.substract_mean_normalize(nullptr, norm_vals);

        ncnn::Extractor extractor = detector_net.create_extractor();
        extractor.set_light_mode(true);

        int input_result = -1;
#if NCNN_STRING
        if (!detector_input_name.empty()) {
            input_result = extractor.input(detector_input_name.c_str(), input);
        } else
#endif
        {
            input_result = extractor.input(detector_input_index, input);
        }
        if (input_result != 0) {
            throw std::runtime_error("Failed to feed detector input into NCNN extractor");
        }

        if (detector_output_names.empty() && detector_output_indexes.empty()) {
            throw std::runtime_error("NCNN model has no output blobs");
        }

        ncnn::Mat output;
        int extract_result = -1;
#if NCNN_STRING
        std::string output_name;
        if (!detector_output_names.empty()) {
            output_name     = detector_output_names.front();
            extract_result = extractor.extract(output_name.c_str(), output);
        } else
#endif
        {
            extract_result = extractor.extract(detector_output_indexes.front(), output);
        }
        if (extract_result != 0) {
            throw std::runtime_error("Failed to extract NCNN detector output");
        }

        const int class_count = std::max(1, static_cast<int>(detector.class_names.size()));
        const int attr_count  = class_count + 4;
        const float inv_scale = 1.0f / std::max(scale, 1e-6f);

        enum class Layout {
            AttrByBoxes,
            BoxesByAttr,
            ChannelsByBoxes
        };

        Layout layout;
        int    num_boxes = 0;

        if (output.dims == 2) {
            if (output.h == attr_count) {
                layout   = Layout::AttrByBoxes;
                num_boxes = output.w;
            } else if (output.w == attr_count) {
                layout   = Layout::BoxesByAttr;
                num_boxes = output.h;
            } else {
                throw std::runtime_error(
                    "Unsupported NCNN detection tensor shape " + std::to_string(output.w) + "x" +
                    std::to_string(output.h));
            }
        } else if (output.dims == 3) {
            if (output.c == 1 && output.h == attr_count) {
                layout   = Layout::AttrByBoxes;
                num_boxes = output.w;
            } else if (output.c == 1 && output.w == attr_count) {
                layout   = Layout::BoxesByAttr;
                num_boxes = output.h;
            } else if (output.c == attr_count) {
                layout   = Layout::ChannelsByBoxes;
                num_boxes = output.w * output.h;
            } else {
                throw std::runtime_error(
                    "Unsupported NCNN detection tensor dims c=" + std::to_string(output.c) + " h=" +
                    std::to_string(output.h) + " w=" + std::to_string(output.w));
            }
        } else {
            throw std::runtime_error("Unsupported NCNN output dims: " + std::to_string(output.dims));
        }

        const auto value_at = [&](const int attr_index, const int box_index) -> float {
            switch (layout) {
            case Layout::AttrByBoxes: {
                const float *data = static_cast<const float *>(output.data);
                return data[attr_index * num_boxes + box_index];
            }
            case Layout::BoxesByAttr: {
                const float *data = static_cast<const float *>(output.data);
                return data[box_index * attr_count + attr_index];
            }
            case Layout::ChannelsByBoxes: {
                const float *channel = output.channel(attr_index);
                return channel[box_index];
            }
            }
            return 0.0f;
        };

        std::vector<Detection> detections;
        detections.reserve(static_cast<size_t>(num_boxes / 8 + 1));

        for (int box_index = 0; box_index < num_boxes; ++box_index) {
            int   best_class = -1;
            float best_score = 0.0f;

            for (int class_index = 0; class_index < class_count; ++class_index) {
                if (!detector.target_class_ids.empty() && !detector.target_class_ids.contains(class_index)) {
                    continue;
                }

                const float score = value_at(class_index + 4, box_index);
                if (score > best_score) {
                    best_score = score;
                    best_class = class_index;
                }
            }

            if (best_class < 0 || best_score < detector.score_threshold) {
                continue;
            }

            const float center_x = value_at(0, box_index);
            const float center_y = value_at(1, box_index);
            const float box_w    = value_at(2, box_index);
            const float box_h    = value_at(3, box_index);

            const float left   = ((center_x - box_w * 0.5f) - static_cast<float>(pad_left)) * inv_scale;
            const float top    = ((center_y - box_h * 0.5f) - static_cast<float>(pad_top)) * inv_scale;
            const float right  = ((center_x + box_w * 0.5f) - static_cast<float>(pad_left)) * inv_scale;
            const float bottom = ((center_y + box_h * 0.5f) - static_cast<float>(pad_top)) * inv_scale;

            const int x1 = std::clamp(static_cast<int>(std::floor(left)), 0, width - 1);
            const int y1 = std::clamp(static_cast<int>(std::floor(top)), 0, height - 1);
            const int x2 = std::clamp(static_cast<int>(std::ceil(right)), x1 + 1, width);
            const int y2 = std::clamp(static_cast<int>(std::ceil(bottom)), y1 + 1, height);

            Detection detection;
            detection.box      = cv::Rect(x1, y1, x2 - x1, y2 - y1);
            detection.score    = best_score;
            detection.class_id = best_class;
            detection.label    = best_class < static_cast<int>(detector.class_names.size())
                                     ? detector.class_names[static_cast<size_t>(best_class)]
                                     : std::to_string(best_class);
            detections.push_back(std::move(detection));
        }

        applyNms(detections, detector.nms_threshold);
        return detections;
#endif
    }

    void detectionLoop() {
        auto next_detection_time = std::chrono::steady_clock::now();
        const auto detection_period =
            std::chrono::milliseconds(std::max(1, 1000 / std::max(1, detector.fps)));

        while (rclcpp::ok() && running.load()) {
            if (!detector_ready || !detector_runtime_enabled.load()) {
                std::this_thread::sleep_for(50ms);
                continue;
            }

            const auto now = std::chrono::steady_clock::now();
            if (now < next_detection_time) {
                std::this_thread::sleep_for(2ms);
                continue;
            }

            std::vector<uint8_t> rgb_data;
            rclcpp::Time         stamp;
            int                  width       = 0;
            int                  height      = 0;
            int64_t              frame_index = -1;
            if (!copyLatestRgbFrame(rgb_data, width, height, frame_index, stamp)) {
                std::this_thread::sleep_for(5ms);
                continue;
            }

            next_detection_time = now + detection_period;
            last_detection_input_frame_index = frame_index;

            try {
                auto detections = runDetection(rgb_data, width, height, detector.input_size);
                updateDetectionCache(std::move(detections), width, height, frame_index, stamp);
            } catch (const std::exception &ex) {
                detector_runtime_enabled.store(false);
                RCLCPP_ERROR(node.get_logger(), "NCNN inference failed, detector disabled: %s", ex.what());
                updateDetectionCache({}, width, height, frame_index, stamp);
            }
        }
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

            rgb_data     = rgb_buffer.data;
            stamp        = rgb_buffer.stamp;
            width        = rgb_buffer.width;
            height       = rgb_buffer.height;
            frame_index  = rgb_buffer.frame_index;
        }

        if (width <= 0 || height <= 0 || rgb_data.empty()) {
            return std::nullopt;
        }

        if (detector_runtime_enabled.load()) {
            cv::Size source_size;
            auto     detections = snapshotDetections(source_size);
            cv::Mat  rgb(height, width, CV_8UC3, rgb_data.data());
            drawDetections(rgb, detections, source_size);
        }

        if (rgb_encoder_width != width || rgb_encoder_height != height) {
            rgb_encoder.initialize("rgb", width, height, rgb.publish_fps, rgb.bitrate_kbps, rgb.gop_size, AV_PIX_FMT_RGB24);
            rgb_encoder_width  = width;
            rgb_encoder_height = height;
            RCLCPP_INFO(
                node.get_logger(), "Initialized RGB encoder: %dx%d @ %d fps, %d kbps, GOP %d", width, height,
                rgb.publish_fps, rgb.bitrate_kbps, rgb.gop_size);
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

            depth_data    = depth_buffer.data;
            stamp         = depth_buffer.stamp;
            width         = depth_buffer.width;
            height        = depth_buffer.height;
            frame_index   = depth_buffer.frame_index;
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

            const float clamped    = std::clamp(static_cast<float>(depth_mm), min_mm, max_mm);
            const float normalized = 1.0f - ((clamped - min_mm) / range);
            depth_visualization[i] = static_cast<uint8_t>(std::clamp(std::lround(normalized * 255.0f), 0L, 255L));
        }

        cv::Mat depth_gray(height, width, CV_8UC1, depth_visualization.data());
        cv::Mat depth_rgb;
        cv::cvtColor(depth_gray, depth_rgb, cv::COLOR_GRAY2RGB);

        if (detector_runtime_enabled.load()) {
            cv::Size source_size;
            auto     detections = snapshotDetections(source_size);
            drawDetections(depth_rgb, detections, source_size);
        }

        if (depth_encoder_width != width || depth_encoder_height != height) {
            depth_encoder.initialize(
                "depth", width, height, depth.publish_fps, depth.bitrate_kbps, depth.gop_size, AV_PIX_FMT_RGB24);
            depth_encoder_width  = width;
            depth_encoder_height = height;
            RCLCPP_INFO(
                node.get_logger(), "Initialized depth encoder: %dx%d @ %d fps, %d kbps, GOP %d", width, height,
                depth.publish_fps, depth.bitrate_kbps, depth.gop_size);
        }

        const auto encoded = depth_encoder.encode(depth_rgb.data, width * 3);
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
    DetectorConfig detector;

    astra::StreamSet    stream_set;
    astra::StreamReader reader;

    FrameBuffer<uint8_t> rgb_buffer;
    FrameBuffer<int16_t> depth_buffer;
    DetectionCache       detection_cache;

    VideoEncoder rgb_encoder;
    VideoEncoder depth_encoder;

    std::vector<uint8_t> depth_visualization;
    std::unique_ptr<CaptureListener> capture_listener;

    rclcpp::Publisher<foxglove_msgs::msg::CompressedVideo>::SharedPtr rgb_video_pub;
    rclcpp::Publisher<foxglove_msgs::msg::CompressedVideo>::SharedPtr depth_video_pub;

    std::thread capture_thread;
    std::thread publish_thread;
    std::thread detection_thread;

    std::atomic<bool> running{false};
    std::atomic<bool> detector_runtime_enabled{false};
    bool              detector_ready     = false;
    bool              astra_initialized  = false;
    bool              gpu_instance_created = false;
    bool              using_vulkan_backend = false;
    int               detected_gpu_count   = 0;
    int               active_gpu_index     = -1;
    std::string       active_backend       = "cpu";

#if LC_VISION_HAVE_NCNN
    ncnn::Net                detector_net;
    int                      detector_input_index = 0;
    std::string              detector_input_name;
    std::vector<int>         detector_output_indexes;
    std::vector<std::string> detector_output_names;
#endif

    int64_t last_rgb_frame_index             = -1;
    int64_t last_depth_frame_index           = -1;
    int64_t last_detection_input_frame_index = -1;
    int     rgb_encoder_width                = 0;
    int     rgb_encoder_height               = 0;
    int     depth_encoder_width              = 0;
    int     depth_encoder_height             = 0;
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
                get_logger(), "Configured RGB stream: %ux%u @ %u fps (mirroring disabled)", active_mode.width(),
                active_mode.height(), active_mode.fps());

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
                get_logger(), "Configured depth stream: %ux%u @ %u fps (mirroring disabled)", active_mode.width(),
                active_mode.height(), active_mode.fps());

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

        if (impl_->detector_runtime_enabled.load()) {
            impl_->detection_thread = std::thread([this]() {
                impl_->detectionLoop();
            });
        }

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
        if (impl_->detection_thread.joinable()) {
            impl_->detection_thread.join();
        }
        if (impl_->capture_thread.joinable()) {
            impl_->capture_thread.join();
        }
        impl_->rgb_encoder.reset();
        impl_->depth_encoder.reset();
        impl_->cleanupGpu();
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
    if (impl_->detection_thread.joinable()) {
        impl_->detection_thread.join();
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
    impl_->cleanupGpu();
    if (impl_->astra_initialized) {
        astra::terminate();
        impl_->astra_initialized = false;
    }
}

void LCVision::getParams() {
    impl_->sdk_root = expandHomeDirectory(this->declare_parameter<std::string>(
        "sdk_root", defaultAstraSdkRoot()));

    impl_->rgb.enable       = this->declare_parameter<bool>("rgb.enable", true);
    impl_->rgb.width        = this->declare_parameter<int>("rgb.width", 640);
    impl_->rgb.height       = this->declare_parameter<int>("rgb.height", 480);
    impl_->rgb.capture_fps  = this->declare_parameter<int>("rgb.capture_fps", 15);
    impl_->rgb.publish_fps  = this->declare_parameter<int>("rgb.publish_fps", 10);
    impl_->rgb.bitrate_kbps = this->declare_parameter<int>("rgb.bitrate_kbps", 1200);
    impl_->rgb.gop_size     = this->declare_parameter<int>("rgb.gop_size", 30);
    impl_->rgb.frame_id     = this->declare_parameter<std::string>("rgb.frame_id", "cam_link");

    impl_->depth.enable               = this->declare_parameter<bool>("depth.enable", true);
    impl_->depth.width                = this->declare_parameter<int>("depth.width", 640);
    impl_->depth.height               = this->declare_parameter<int>("depth.height", 480);
    impl_->depth.capture_fps          = this->declare_parameter<int>("depth.capture_fps", 15);
    impl_->depth.publish_fps          = this->declare_parameter<int>("depth.publish_fps", 8);
    impl_->depth.bitrate_kbps         = this->declare_parameter<int>("depth.bitrate_kbps", 500);
    impl_->depth.gop_size             = this->declare_parameter<int>("depth.gop_size", 24);
    impl_->depth.frame_id             = this->declare_parameter<std::string>("depth.frame_id", "cam_link");
    impl_->depth.visualization_min_mm = this->declare_parameter<int>("depth.visualization_min_mm", 300);
    impl_->depth.visualization_max_mm = this->declare_parameter<int>("depth.visualization_max_mm", 5000);
    impl_->depth.invalid_as_black     = this->declare_parameter<bool>("depth.invalid_as_black", true);

    impl_->detector.enable = this->declare_parameter<bool>("detector.enable", true);
    impl_->detector.backend = this->declare_parameter<std::string>("detector.backend", "auto");
    impl_->detector.model_dir =
        this->declare_parameter<std::string>("detector.model_dir", "models/yolo26n_ncnn_model");
    impl_->detector.input_size      = this->declare_parameter<int>("detector.input_size", 640);
    impl_->detector.score_threshold = this->declare_parameter<double>("detector.score_threshold", 0.25);
    impl_->detector.nms_threshold   = this->declare_parameter<double>("detector.nms_threshold", 0.45);
    impl_->detector.fps             = this->declare_parameter<int>("detector.fps", 5);
    impl_->detector.vulkan_device_index = this->declare_parameter<int>("detector.vulkan_device_index", 0);
    impl_->detector.cpu_num_threads     = this->declare_parameter<int>("detector.cpu_num_threads", defaultDetectorThreadCount());
    impl_->detector.log_backend_info    = this->declare_parameter<bool>("detector.log_backend_info", true);
    impl_->detector.target_classes =
        this->declare_parameter<std::vector<std::string>>("detector.target_classes", std::vector<std::string>{"person"});

    RCLCPP_INFO(
        get_logger(), "RGB config: enable=%s capture=%dx%d@%d publish=%d bitrate=%dkbps gop=%d",
        impl_->rgb.enable ? "true" : "false", impl_->rgb.width, impl_->rgb.height, impl_->rgb.capture_fps,
        impl_->rgb.publish_fps, impl_->rgb.bitrate_kbps, impl_->rgb.gop_size);
    RCLCPP_INFO(
        get_logger(), "Depth config: enable=%s capture=%dx%d@%d publish=%d bitrate=%dkbps gop=%d vis=[%d,%d]mm",
        impl_->depth.enable ? "true" : "false", impl_->depth.width, impl_->depth.height, impl_->depth.capture_fps,
        impl_->depth.publish_fps, impl_->depth.bitrate_kbps, impl_->depth.gop_size,
        impl_->depth.visualization_min_mm, impl_->depth.visualization_max_mm);
    RCLCPP_INFO(
        get_logger(),
        "Detector config: enable=%s backend=%s model_dir=%s input=%d score=%.2f nms=%.2f fps=%d vulkan_device=%d cpu_threads=%d",
        impl_->detector.enable ? "true" : "false", impl_->detector.backend.c_str(), impl_->detector.model_dir.c_str(),
        impl_->detector.input_size, impl_->detector.score_threshold, impl_->detector.nms_threshold, impl_->detector.fps,
        impl_->detector.vulkan_device_index, impl_->detector.cpu_num_threads);
}

void LCVision::prepareModel() {
    impl_->detector_ready = false;
    impl_->detector_runtime_enabled.store(false);
    impl_->cleanupGpu();

#if !LC_VISION_HAVE_NCNN
    if (impl_->detector.enable) {
        RCLCPP_WARN(get_logger(), "NCNN was not found at build time; detector overlays are disabled.");
    }
    return;
#else
    if (!impl_->detector.enable) {
        RCLCPP_INFO(get_logger(), "NCNN detector disabled by parameter.");
        return;
    }

    if (!impl_->rgb.enable) {
        RCLCPP_WARN(get_logger(), "NCNN detector requires RGB input; detector disabled because rgb.enable=false.");
        return;
    }

    try {
        const fs::path model_dir = impl_->resolveModelDirectory();
        if (!fs::exists(model_dir) || !fs::is_directory(model_dir)) {
            throw std::runtime_error("Model directory does not exist: " + model_dir.string());
        }

        fs::path model_param;
        for (const auto &entry : fs::directory_iterator(model_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".param") {
                model_param = entry.path();
                break;
            }
        }
        if (model_param.empty()) {
            throw std::runtime_error("No .param file found in model directory: " + model_dir.string());
        }

        const fs::path model_bin = model_param.parent_path() / (model_param.stem().string() + ".bin");
        if (!fs::exists(model_bin)) {
            throw std::runtime_error("NCNN .bin file not found next to " + model_param.string());
        }

        const fs::path metadata_path = model_dir / "metadata.yaml";
        const ParsedMetadata metadata = parseMetadataFile(metadata_path);
        if (!metadata.class_names.empty()) {
            impl_->detector.class_names = metadata.class_names;
        }
        if (metadata.input_size > 0) {
            impl_->detector.input_size = metadata.input_size;
        }

        impl_->updateTargetClassIds();

        impl_->detector_net.clear();
        impl_->configureDetectorBackend();

        if (impl_->detector_net.load_param(model_param.c_str()) != 0) {
            throw std::runtime_error("Failed to load NCNN param file: " + model_param.string());
        }
        if (impl_->detector_net.load_model(model_bin.c_str()) != 0) {
            throw std::runtime_error("Failed to load NCNN model weights: " + model_bin.string());
        }

        const auto input_indexes  = impl_->detector_net.input_indexes();
        const auto output_indexes = impl_->detector_net.output_indexes();
        if (input_indexes.empty()) {
            throw std::runtime_error("NCNN model exposes no input blobs");
        }
        if (output_indexes.empty()) {
            throw std::runtime_error("NCNN model exposes no output blobs");
        }

        impl_->detector_input_index = input_indexes.front();
        impl_->detector_output_indexes.assign(output_indexes.begin(), output_indexes.end());
#if NCNN_STRING
        const auto input_names  = impl_->detector_net.input_names();
        const auto output_names = impl_->detector_net.output_names();
        impl_->detector_input_name = input_names.empty() ? std::string() : std::string(input_names.front());
        impl_->detector_output_names.clear();
        impl_->detector_output_names.reserve(output_names.size());
        for (const char *name : output_names) {
            impl_->detector_output_names.emplace_back(name);
        }
        std::sort(impl_->detector_output_names.begin(), impl_->detector_output_names.end());
#else
        impl_->detector_input_name.clear();
        impl_->detector_output_names.clear();
#endif

        impl_->detector_ready = true;
        impl_->detector_runtime_enabled.store(true);
        const std::string output_desc =
            !impl_->detector_output_names.empty() ? impl_->detector_output_names.front()
                                                  : std::to_string(impl_->detector_output_indexes.front());
        const std::string input_desc =
            !impl_->detector_input_name.empty() ? impl_->detector_input_name : std::to_string(impl_->detector_input_index);
        RCLCPP_INFO(
            get_logger(), "Loaded NCNN detector from %s using input '%s' and output '%s' on backend '%s'",
            model_dir.string().c_str(), input_desc.c_str(), output_desc.c_str(), impl_->active_backend.c_str());
    } catch (const std::exception &ex) {
        RCLCPP_ERROR(get_logger(), "Failed to prepare NCNN detector, overlays disabled: %s", ex.what());
        impl_->detector_ready = false;
        impl_->detector_runtime_enabled.store(false);
        impl_->cleanupGpu();
    }
#endif
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
