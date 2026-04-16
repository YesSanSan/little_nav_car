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
#include <geometry_msgs/msg/point.hpp>
#include <lc_vision/msg/detection_depth.hpp>
#include <lc_vision/msg/detection_depth_array.hpp>
#if LC_VISION_DETECTOR_FRAMEWORK_OPENVINO
#include <openvino/openvino.hpp>
#elif LC_VISION_DETECTOR_FRAMEWORK_NCNN
#include <gpu.h>
#include <net.h>
#endif
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/qos.hpp>
#include <std_msgs/msg/header.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

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

struct DepthDebugData {
    std::vector<int> histogram_counts;
    int              histogram_min_mm    = 0;
    int              histogram_bin_size_mm = 1;
    int              peak_bin_start      = -1;
    int              peak_bin_end        = -1;
    cv::Rect         depth_roi;
};

struct DetectionDepthResult {
    Detection      detection;
    bool           depth_valid           = false;
    float          depth_mm              = 0.0f;
    float          peak_min_mm           = 0.0f;
    float          peak_max_mm           = 0.0f;
    uint32_t       roi_valid_pixel_count = 0;
    uint32_t       peak_pixel_count      = 0;
    DepthDebugData debug;
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

std::string defaultDetectorModelDir() {
#if LC_VISION_DETECTOR_FRAMEWORK_OPENVINO
    return "models/yolo26n_openvino_model";
#else
    return "models/yolo26n_ncnn_model";
#endif
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

std::string buildDetectionCaption(
    const DetectionDepthResult &result, const bool annotate_depth, const bool annotate_peak_range) {
    std::string caption =
        result.detection.label + " " + cv::format("%.2f", static_cast<double>(result.detection.score));
    if (!annotate_depth) {
        return caption;
    }

    if (!result.depth_valid) {
        caption += " depth=invalid";
        return caption;
    }

    caption += " " + cv::format("%.0fmm", static_cast<double>(result.depth_mm));
    if (annotate_peak_range) {
        caption += " [" + cv::format("%.0f", static_cast<double>(result.peak_min_mm)) + "," +
                   cv::format("%.0f", static_cast<double>(result.peak_max_mm)) + "]";
    }
    return caption;
}

void drawDetections(
    cv::Mat &image, const std::vector<DetectionDepthResult> &results, const cv::Size &source_size,
    const bool annotate_depth, const bool annotate_peak_range = false) {
    if (image.empty() || source_size.width <= 0 || source_size.height <= 0) {
        return;
    }

    const float scale_x = static_cast<float>(image.cols) / static_cast<float>(source_size.width);
    const float scale_y = static_cast<float>(image.rows) / static_cast<float>(source_size.height);
    const int   thickness = std::max(1, std::min(image.cols, image.rows) / 240);
    const int   font_thickness = std::max(1, thickness);
    const double font_scale = std::max(0.45, std::min(image.cols, image.rows) / 800.0);

    for (const auto &result : results) {
        const auto &detection = result.detection;
        cv::Rect scaled_box(
            static_cast<int>(std::lround(static_cast<float>(detection.box.x) * scale_x)),
            static_cast<int>(std::lround(static_cast<float>(detection.box.y) * scale_y)),
            static_cast<int>(std::lround(static_cast<float>(detection.box.width) * scale_x)),
            static_cast<int>(std::lround(static_cast<float>(detection.box.height) * scale_y)));
        scaled_box &= cv::Rect(0, 0, image.cols, image.rows);

        if (scaled_box.width <= 1 || scaled_box.height <= 1) {
            continue;
        }

        const std::string caption = buildDetectionCaption(result, annotate_depth, annotate_peak_range);

        int       baseline = 0;
        const auto text_size =
            cv::getTextSize(caption, cv::FONT_HERSHEY_SIMPLEX, font_scale, font_thickness, &baseline);
        const int text_x = std::max(0, scaled_box.x);
        const int text_y = std::max(text_size.height + baseline + 4, scaled_box.y);
        const int bg_y   = text_y - text_size.height - baseline - 4;
        const int bg_w   = std::min(text_size.width + 8, image.cols - text_x);
        const int bg_h   = text_size.height + baseline + 8;

        const cv::Scalar box_color = result.depth_valid ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 140, 255);
        cv::rectangle(image, scaled_box, box_color, thickness, cv::LINE_AA);
        cv::rectangle(image, cv::Rect(text_x, bg_y, bg_w, bg_h), box_color * 0.38, cv::FILLED);
        cv::putText(
            image, caption, cv::Point(text_x + 4, text_y - 4), cv::FONT_HERSHEY_SIMPLEX, font_scale,
            cv::Scalar(255, 255, 255), font_thickness, cv::LINE_AA);
    }
}

template<typename ValueAt>
std::vector<Detection> decodeDetections(
    const int width, const int height, const int pad_left, const int pad_top, const float scale,
    const std::vector<std::string> &class_names, const std::set<int> &target_class_ids, const float score_threshold,
    const float nms_threshold, const int num_boxes, const ValueAt &value_at) {
    const int   class_count = std::max(1, static_cast<int>(class_names.size()));
    const float inv_scale   = 1.0f / std::max(scale, 1e-6f);

    std::vector<Detection> detections;
    detections.reserve(static_cast<size_t>(num_boxes / 8 + 1));

    for (int box_index = 0; box_index < num_boxes; ++box_index) {
        int   best_class = -1;
        float best_score = 0.0f;

        for (int class_index = 0; class_index < class_count; ++class_index) {
            if (!target_class_ids.empty() && !target_class_ids.contains(class_index)) {
                continue;
            }

            const float score = value_at(class_index + 4, box_index);
            if (score > best_score) {
                best_score = score;
                best_class = class_index;
            }
        }

        if (best_class < 0 || best_score < score_threshold) {
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
        detection.label    = best_class < static_cast<int>(class_names.size())
                                 ? class_names[static_cast<size_t>(best_class)]
                                 : std::to_string(best_class);
        detections.push_back(std::move(detection));
    }

    applyNms(detections, nms_threshold);
    return detections;
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
        struct EstimationConfig {
            int   min_valid_pixels        = 50;
            int   histogram_bin_size_mm   = 50;
            float peak_min_ratio          = 0.25f;
            int   peak_min_count          = 5;
            int   min_peak_pixels         = 20;
            float trim_ratio              = 0.1f;
            bool  annotate_depth_on_rgb   = true;
            bool  annotate_depth_on_depth = true;
        };

        struct DebugMarkerConfig {
            bool        enable                  = false;
            std::string topic                   = "~/depth_debug_marker";
            std::string frame_id                = "base_link";
            float       z                       = 0.05f;
            float       line_width              = 0.02f;
            float       alpha                   = 0.9f;
            bool        use_detection_label_ns  = true;
        };

        int  visualization_min_mm = 300;
        int  visualization_max_mm = 5000;
        int  roi_min_valid_mm     = 300;
        int  roi_max_valid_mm     = 5000;
        bool invalid_as_black     = true;
        EstimationConfig estimation;
        DebugMarkerConfig debug_marker;
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
        std::mutex                       mutex;
        std::vector<DetectionDepthResult> detections;
        int                              width       = 0;
        int                              height      = 0;
        int64_t                          frame_index = -1;
        rclcpp::Time                     stamp;
        bool                             available = false;
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
        std::string configured_model_dir = detector.model_dir.empty() ? defaultDetectorModelDir() : detector.model_dir;
        fs::path    model_path           = fs::path(expandHomeDirectory(configured_model_dir));

        const auto matches_current_framework = [](const fs::path &dir) {
            if (!fs::exists(dir) || !fs::is_directory(dir)) {
                return false;
            }
#if LC_VISION_DETECTOR_FRAMEWORK_OPENVINO
            constexpr const char *required_extension = ".xml";
#elif LC_VISION_DETECTOR_FRAMEWORK_NCNN
            constexpr const char *required_extension = ".param";
#else
            return true;
#endif
            for (const auto &entry : fs::directory_iterator(dir)) {
                if (entry.is_regular_file() && entry.path().extension() == required_extension) {
                    return true;
                }
            }
            return false;
        };

        if (model_path.is_absolute() && matches_current_framework(model_path)) {
            return model_path;
        }

        try {
            const auto package_share = fs::path(ament_index_cpp::get_package_share_directory("lc_vision"));
            const auto installed_path = package_share / model_path;
            if (matches_current_framework(installed_path)) {
                return installed_path;
            }
        } catch (...) {
        }

        const fs::path source_package_dir = fs::path(__FILE__).parent_path().parent_path();
        const fs::path source_path        = source_package_dir / model_path;
        if (matches_current_framework(source_path)) {
            return source_path;
        }

        if (configured_model_dir != defaultDetectorModelDir()) {
            const fs::path fallback_path = fs::path(defaultDetectorModelDir());

            try {
                const auto package_share = fs::path(ament_index_cpp::get_package_share_directory("lc_vision"));
                const auto installed_fallback = package_share / fallback_path;
                if (fs::exists(installed_fallback)) {
                    return installed_fallback;
                }
            } catch (...) {
            }

            const fs::path source_fallback = source_package_dir / fallback_path;
            if (fs::exists(source_fallback)) {
                return source_fallback;
            }
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
#if LC_VISION_DETECTOR_FRAMEWORK_NCNN
        const auto normalized = toLower(detector.backend);
        return normalized == "auto" || normalized == "vulkan";
#else
        return false;
#endif
    }

    [[nodiscard]] bool forcesCpuOnly() const {
        return toLower(detector.backend) == "cpu";
    }

    [[nodiscard]] bool backendValueIsKnown() const {
        const auto normalized = toLower(detector.backend);
#if LC_VISION_DETECTOR_FRAMEWORK_OPENVINO
        return normalized == "auto" || normalized == "cpu" || normalized == "gpu";
#else
        return normalized == "auto" || normalized == "vulkan" || normalized == "cpu";
#endif
    }

    [[nodiscard]] std::string requestedDetectorDevice() const {
#if LC_VISION_DETECTOR_FRAMEWORK_OPENVINO
        const auto normalized = toLower(detector.backend);
        if (normalized == "cpu") {
            return "CPU";
        }
        if (normalized == "gpu") {
            return "GPU";
        }
        return "AUTO";
#else
        return "CPU";
#endif
    }

    void cleanupGpu() {
#if LC_VISION_DETECTOR_FRAMEWORK_NCNN
        detector_net.clear();
#if NCNN_VULKAN
        if (gpu_instance_created) {
            ncnn::destroy_gpu_instance();
            gpu_instance_created = false;
        }
#endif
#elif LC_VISION_DETECTOR_FRAMEWORK_OPENVINO
        detector_infer_request = {};
        detector_compiled_model = {};
        detector_model.reset();
#else
        gpu_instance_created = false;
#endif
        using_vulkan_backend = false;
        active_backend       = "disabled";
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
#if LC_VISION_DETECTOR_FRAMEWORK_OPENVINO
        if (!backendValueIsKnown()) {
            RCLCPP_WARN(
                node.get_logger(), "Unknown detector.backend value '%s', defaulting to AUTO for OpenVINO.",
                detector.backend.c_str());
        }

        detector_device_name = requestedDetectorDevice();
        active_backend       = "openvino:" + toLower(detector_device_name);
        logBackendSelection("OpenVINO detector configured");
#elif LC_VISION_DETECTOR_FRAMEWORK_NCNN
        detector_net.opt.num_threads        = std::max(1, detector.cpu_num_threads);
        detector_net.opt.use_vulkan_compute = false;
        using_vulkan_backend                = false;
        active_backend                      = "ncnn:cpu";
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
            active_backend = "ncnn:cpu";
            logBackendSelection("NCNN detector configured");
            return;
        }

        active_gpu_index = std::clamp(detector.vulkan_device_index, 0, detected_gpu_count - 1);
        detector_net.opt.use_vulkan_compute = true;
        detector_net.set_vulkan_device(active_gpu_index);
        using_vulkan_backend = true;
        active_backend       = "ncnn:vulkan";
        logBackendSelection("NCNN detector configured");
#else
        if (prefersVulkan()) {
            RCLCPP_WARN(
                node.get_logger(),
                "This NCNN build does not include Vulkan support, falling back to CPU backend.");
        }
        logBackendSelection("NCNN detector configured");
#endif
#else
        detector_runtime_enabled.store(false);
        detector_ready = false;
        active_backend = "disabled";
        return;
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
        std::vector<DetectionDepthResult> detections, const int width, const int height, const int64_t frame_index,
        const rclcpp::Time &stamp) {
        std::lock_guard<std::mutex> lock(detection_cache.mutex);
        detection_cache.detections  = std::move(detections);
        detection_cache.width       = width;
        detection_cache.height      = height;
        detection_cache.frame_index = frame_index;
        detection_cache.stamp       = stamp;
        detection_cache.available   = true;
    }

    std::vector<DetectionDepthResult> snapshotDetections(cv::Size &source_size) {
        std::lock_guard<std::mutex> lock(detection_cache.mutex);
        source_size = cv::Size(detection_cache.width, detection_cache.height);
        return detection_cache.detections;
    }

    bool copyLatestDepthFrame(
        std::vector<int16_t> &depth_data, int &width, int &height, int64_t &frame_index, rclcpp::Time &stamp) {
        std::lock_guard<std::mutex> lock(depth_buffer.mutex);
        if (!depth_buffer.available) {
            return false;
        }

        depth_data    = depth_buffer.data;
        width         = depth_buffer.width;
        height        = depth_buffer.height;
        frame_index   = depth_buffer.frame_index;
        stamp         = depth_buffer.stamp;
        return !depth_data.empty() && width > 0 && height > 0;
    }

    cv::Rect scaleRectToSize(const cv::Rect &source_box, const cv::Size &source_size, const cv::Size &target_size) const {
        if (source_size.width <= 0 || source_size.height <= 0 || target_size.width <= 0 || target_size.height <= 0) {
            return {};
        }

        const float scale_x = static_cast<float>(target_size.width) / static_cast<float>(source_size.width);
        const float scale_y = static_cast<float>(target_size.height) / static_cast<float>(source_size.height);
        cv::Rect scaled_box(
            static_cast<int>(std::lround(static_cast<float>(source_box.x) * scale_x)),
            static_cast<int>(std::lround(static_cast<float>(source_box.y) * scale_y)),
            static_cast<int>(std::lround(static_cast<float>(source_box.width) * scale_x)),
            static_cast<int>(std::lround(static_cast<float>(source_box.height) * scale_y)));
        scaled_box &= cv::Rect(0, 0, target_size.width, target_size.height);
        return scaled_box;
    }

    DetectionDepthResult estimateDepthForDetection(
        const Detection &detection, const cv::Size &rgb_size, const std::vector<int16_t> &depth_data,
        const cv::Size &depth_size) const {
        DetectionDepthResult result;
        result.detection = detection;
        result.debug.histogram_min_mm = depth.roi_min_valid_mm;
        result.debug.histogram_bin_size_mm = std::max(1, depth.estimation.histogram_bin_size_mm);
        result.debug.depth_roi = scaleRectToSize(detection.box, rgb_size, depth_size);

        const cv::Rect roi = result.debug.depth_roi & cv::Rect(0, 0, depth_size.width, depth_size.height);
        if (roi.width <= 1 || roi.height <= 1) {
            return result;
        }

        std::vector<int> valid_depths;
        valid_depths.reserve(static_cast<size_t>(roi.area()));
        for (int y = roi.y; y < roi.y + roi.height; ++y) {
            const size_t row_offset = static_cast<size_t>(y) * static_cast<size_t>(depth_size.width);
            for (int x = roi.x; x < roi.x + roi.width; ++x) {
                const int depth_mm = depth_data[row_offset + static_cast<size_t>(x)];
                if (depth_mm <= 0) {
                    continue;
                }
                if (depth_mm < depth.roi_min_valid_mm || depth_mm > depth.roi_max_valid_mm) {
                    continue;
                }
                valid_depths.push_back(depth_mm);
            }
        }

        result.roi_valid_pixel_count = static_cast<uint32_t>(valid_depths.size());
        if (valid_depths.size() < static_cast<size_t>(std::max(1, depth.estimation.min_valid_pixels))) {
            return result;
        }

        const int bin_size_mm = std::max(1, depth.estimation.histogram_bin_size_mm);
        const int histogram_span_mm = std::max(depth.roi_max_valid_mm - depth.roi_min_valid_mm + 1, bin_size_mm);
        const int histogram_bin_count = std::max(1, (histogram_span_mm + bin_size_mm - 1) / bin_size_mm);
        result.debug.histogram_counts.assign(static_cast<size_t>(histogram_bin_count), 0);

        for (const int depth_mm : valid_depths) {
            const int bin_index =
                std::clamp((depth_mm - depth.roi_min_valid_mm) / bin_size_mm, 0, histogram_bin_count - 1);
            ++result.debug.histogram_counts[static_cast<size_t>(bin_index)];
        }

        auto peak_it = std::max_element(result.debug.histogram_counts.begin(), result.debug.histogram_counts.end());
        if (peak_it == result.debug.histogram_counts.end() || *peak_it <= 0) {
            return result;
        }

        const int peak_bin = static_cast<int>(std::distance(result.debug.histogram_counts.begin(), peak_it));
        const int peak_count = *peak_it;
        const float min_ratio = std::clamp(depth.estimation.peak_min_ratio, 0.0f, 1.0f);
        const int min_count = std::max(1, depth.estimation.peak_min_count);

        int peak_start = peak_bin;
        while (peak_start > 0) {
            const int neighbor_count = result.debug.histogram_counts[static_cast<size_t>(peak_start - 1)];
            if (neighbor_count < min_count || static_cast<float>(neighbor_count) < static_cast<float>(peak_count) * min_ratio) {
                break;
            }
            --peak_start;
        }

        int peak_end = peak_bin;
        while (peak_end + 1 < histogram_bin_count) {
            const int neighbor_count = result.debug.histogram_counts[static_cast<size_t>(peak_end + 1)];
            if (neighbor_count < min_count || static_cast<float>(neighbor_count) < static_cast<float>(peak_count) * min_ratio) {
                break;
            }
            ++peak_end;
        }

        result.debug.peak_bin_start = peak_start;
        result.debug.peak_bin_end   = peak_end;
        result.peak_min_mm = static_cast<float>(depth.roi_min_valid_mm + peak_start * bin_size_mm);
        result.peak_max_mm = static_cast<float>(std::min(depth.roi_max_valid_mm, depth.roi_min_valid_mm + (peak_end + 1) * bin_size_mm - 1));

        std::vector<int> peak_depths;
        peak_depths.reserve(valid_depths.size());
        for (const int depth_mm : valid_depths) {
            if (static_cast<float>(depth_mm) >= result.peak_min_mm && static_cast<float>(depth_mm) <= result.peak_max_mm) {
                peak_depths.push_back(depth_mm);
            }
        }

        result.peak_pixel_count = static_cast<uint32_t>(peak_depths.size());
        if (peak_depths.size() < static_cast<size_t>(std::max(1, depth.estimation.min_peak_pixels))) {
            return result;
        }

        std::sort(peak_depths.begin(), peak_depths.end());
        const float trim_ratio = std::clamp(depth.estimation.trim_ratio, 0.0f, 0.49f);
        const size_t trim_count = static_cast<size_t>(std::floor(static_cast<double>(peak_depths.size()) * trim_ratio));
        size_t keep_begin = trim_count;
        size_t keep_end   = peak_depths.size() - trim_count;
        if (keep_begin >= keep_end) {
            keep_begin = 0;
            keep_end   = peak_depths.size();
        }

        const auto begin_it = peak_depths.begin() + static_cast<std::ptrdiff_t>(keep_begin);
        const auto end_it   = peak_depths.begin() + static_cast<std::ptrdiff_t>(keep_end);
        const double sum = std::accumulate(begin_it, end_it, 0.0);
        const size_t keep_count = static_cast<size_t>(std::distance(begin_it, end_it));
        if (keep_count == 0U) {
            return result;
        }

        result.depth_mm = static_cast<float>(sum / static_cast<double>(keep_count));
        result.depth_valid = true;
        return result;
    }

    std::vector<DetectionDepthResult> estimateDepthResults(
        const std::vector<Detection> &detections, const cv::Size &rgb_size, const std::vector<int16_t> &depth_data,
        const cv::Size &depth_size) const {
        std::vector<DetectionDepthResult> results;
        results.reserve(detections.size());
        for (const auto &detection : detections) {
            results.push_back(estimateDepthForDetection(detection, rgb_size, depth_data, depth_size));
        }
        return results;
    }

    void publishDetectionDepths(
        const std::vector<DetectionDepthResult> &results, const int width, const int height, const rclcpp::Time &stamp) {
        if (detections_depth_pub == nullptr) {
            return;
        }

        lc_vision::msg::DetectionDepthArray msg;
        msg.header.stamp = toBuiltinTime(stamp);
        msg.header.frame_id = rgb.frame_id;
        msg.source_width = static_cast<uint32_t>(std::max(0, width));
        msg.source_height = static_cast<uint32_t>(std::max(0, height));
        msg.detections.reserve(results.size());

        for (const auto &result : results) {
            lc_vision::msg::DetectionDepth detection_msg;
            detection_msg.header = msg.header;
            detection_msg.label = result.detection.label;
            detection_msg.class_id = result.detection.class_id;
            detection_msg.score = result.detection.score;
            detection_msg.x = result.detection.box.x;
            detection_msg.y = result.detection.box.y;
            detection_msg.width = result.detection.box.width;
            detection_msg.height = result.detection.box.height;
            detection_msg.depth_valid = result.depth_valid;
            detection_msg.depth_mm = result.depth_mm;
            detection_msg.peak_min_mm = result.peak_min_mm;
            detection_msg.peak_max_mm = result.peak_max_mm;
            detection_msg.roi_valid_pixel_count = result.roi_valid_pixel_count;
            detection_msg.peak_pixel_count = result.peak_pixel_count;
            msg.detections.push_back(std::move(detection_msg));
        }

        detections_depth_pub->publish(msg);
    }

    void publishDepthDebugMarkers(const std::vector<DetectionDepthResult> &results, const rclcpp::Time &stamp) {
        if (depth_debug_marker_pub == nullptr || !depth.debug_marker.enable) {
            return;
        }

        visualization_msgs::msg::MarkerArray marker_array;
        visualization_msgs::msg::Marker clear_marker;
        clear_marker.header.stamp = toBuiltinTime(stamp);
        clear_marker.header.frame_id = depth.debug_marker.frame_id;
        clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        marker_array.markers.push_back(clear_marker);
        int marker_id = 0;
        constexpr int circle_segments = 72;
        constexpr double kPi = 3.14159265358979323846;

        for (const auto &result : results) {
            if (!result.depth_valid) {
                continue;
            }

            visualization_msgs::msg::Marker marker;
            marker.header.stamp = toBuiltinTime(stamp);
            marker.header.frame_id = depth.debug_marker.frame_id;
            marker.ns = depth.debug_marker.use_detection_label_ns ? result.detection.label : "depth_debug";
            marker.id = marker_id++;
            marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.pose.orientation.w = 1.0;
            marker.scale.x = std::max(1.0e-4f, depth.debug_marker.line_width);
            marker.color.r = 0.2f;
            marker.color.g = 0.95f;
            marker.color.b = 0.2f;
            marker.color.a = std::clamp(depth.debug_marker.alpha, 0.0f, 1.0f);

            const double radius_m = std::max(0.0, static_cast<double>(result.depth_mm) / 1000.0);
            marker.points.reserve(circle_segments + 1);
            for (int i = 0; i <= circle_segments; ++i) {
                const double angle = (2.0 * kPi * static_cast<double>(i)) / static_cast<double>(circle_segments);
                geometry_msgs::msg::Point point;
                point.x = radius_m * std::cos(angle);
                point.y = radius_m * std::sin(angle);
                point.z = depth.debug_marker.z;
                marker.points.push_back(std::move(point));
            }

            marker_array.markers.push_back(std::move(marker));
        }
        depth_debug_marker_pub->publish(marker_array);
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
#if LC_VISION_DETECTOR_FRAMEWORK_OPENVINO
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

        ov::Tensor input_tensor(
            ov::element::f32, ov::Shape{1, 3, static_cast<size_t>(input_size), static_cast<size_t>(input_size)});
        float       *input_data     = input_tensor.data<float>();
        const size_t channel_stride = static_cast<size_t>(input_size) * static_cast<size_t>(input_size);

        for (int y = 0; y < input_size; ++y) {
            const auto *row = padded.ptr<uint8_t>(y);
            for (int x = 0; x < input_size; ++x) {
                const size_t index = static_cast<size_t>(y) * static_cast<size_t>(input_size) + static_cast<size_t>(x);
                const auto  *pixel = row + (static_cast<size_t>(x) * 3U);
                input_data[index]                  = static_cast<float>(pixel[0]) / 255.0f;
                input_data[channel_stride + index] = static_cast<float>(pixel[1]) / 255.0f;
                input_data[channel_stride * 2U + index] = static_cast<float>(pixel[2]) / 255.0f;
            }
        }

        detector_infer_request.set_input_tensor(input_tensor);
        detector_infer_request.infer();

        const ov::Tensor output = detector_infer_request.get_output_tensor(0);
        auto             shape  = output.get_shape();
        while (shape.size() > 2 && !shape.empty() && shape.front() == 1U) {
            shape.erase(shape.begin());
        }

        const auto type       = output.get_element_type();
        const float *data_f32 = type == ov::element::f32 ? output.data<const float>() : nullptr;
        const ov::float16 *data_f16 = type == ov::element::f16 ? output.data<const ov::float16>() : nullptr;

        if (data_f32 == nullptr && data_f16 == nullptr) {
            throw std::runtime_error("Unsupported OpenVINO output element type: " + type.to_string());
        }

        if (shape.size() != 2 || static_cast<int>(shape[1]) != 6) {
            throw std::runtime_error(
                "Unsupported OpenVINO detection tensor shape for Ultralytics export: rank=" +
                std::to_string(shape.size()) + (shape.empty() ? std::string() : " first_dim=" + std::to_string(shape[0])));
        }

        const auto read_at = [&](const int flat_index) -> float {
            if (data_f32 != nullptr) {
                return data_f32[flat_index];
            }
            return static_cast<float>(data_f16[flat_index]);
        };

        const int num_boxes = static_cast<int>(shape[0]);
        const float inv_scale = 1.0f / std::max(scale, 1e-6f);
        std::vector<Detection> detections;
        detections.reserve(static_cast<size_t>(num_boxes));

        for (int row = 0; row < num_boxes; ++row) {
            const int base = row * 6;
            const float score = read_at(base + 4);
            if (score < detector.score_threshold) {
                continue;
            }

            const int class_id = static_cast<int>(std::lround(read_at(base + 5)));
            if (!detector.target_class_ids.empty() && !detector.target_class_ids.contains(class_id)) {
                continue;
            }

            const float raw_x1 = read_at(base + 0);
            const float raw_y1 = read_at(base + 1);
            const float raw_x2 = read_at(base + 2);
            const float raw_y2 = read_at(base + 3);

            const int x1 = std::clamp(static_cast<int>(std::floor((raw_x1 - static_cast<float>(pad_left)) * inv_scale)), 0, width - 1);
            const int y1 = std::clamp(static_cast<int>(std::floor((raw_y1 - static_cast<float>(pad_top)) * inv_scale)), 0, height - 1);
            const int x2 = std::clamp(static_cast<int>(std::ceil((raw_x2 - static_cast<float>(pad_left)) * inv_scale)), x1 + 1, width);
            const int y2 = std::clamp(static_cast<int>(std::ceil((raw_y2 - static_cast<float>(pad_top)) * inv_scale)), y1 + 1, height);

            Detection detection;
            detection.box      = cv::Rect(x1, y1, x2 - x1, y2 - y1);
            detection.score    = score;
            detection.class_id = class_id;
            detection.label    = class_id >= 0 && class_id < static_cast<int>(detector.class_names.size())
                                     ? detector.class_names[static_cast<size_t>(class_id)]
                                     : std::to_string(class_id);
            detections.push_back(std::move(detection));
        }

        applyNms(detections, detector.nms_threshold);
        return detections;
#elif LC_VISION_DETECTOR_FRAMEWORK_NCNN
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
            output_name    = detector_output_names.front();
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

        return decodeDetections(
            width, height, pad_left, pad_top, scale, detector.class_names, detector.target_class_ids,
            detector.score_threshold, detector.nms_threshold, num_boxes, value_at);
#else
        (void)rgb_data;
        (void)width;
        (void)height;
        (void)input_size;
        return {};
#endif
    }

    void buildDepthVisualization(
        const std::vector<int16_t> &depth_data, const int width, const int height, std::vector<uint8_t> &output) const {
        output.resize(static_cast<size_t>(width) * static_cast<size_t>(height));

        const float min_mm = static_cast<float>(depth.visualization_min_mm);
        const float max_mm = static_cast<float>(std::max(depth.visualization_max_mm, depth.visualization_min_mm + 1));
        const float range  = max_mm - min_mm;

        for (size_t i = 0; i < depth_data.size(); ++i) {
            const int16_t depth_mm = depth_data[i];
            if (depth_mm <= 0) {
                output[i] = depth.invalid_as_black ? 0 : 255;
                continue;
            }

            const float clamped    = std::clamp(static_cast<float>(depth_mm), min_mm, max_mm);
            const float normalized = 1.0f - ((clamped - min_mm) / range);
            output[i] = static_cast<uint8_t>(std::clamp(std::lround(normalized * 255.0f), 0L, 255L));
        }
    }

#if LC_VISION_ENABLE_DEPTH_HISTOGRAM_DEBUG || LC_VISION_ENABLE_DEPTH_PEAK_MASK_DEBUG
    int findBestDebugDetectionIndex(const std::vector<DetectionDepthResult> &results) const {
        int best_index = -1;
        float best_score = -1.0f;
        for (size_t i = 0; i < results.size(); ++i) {
            if (!results[i].depth_valid) {
                continue;
            }
            if (results[i].detection.score > best_score) {
                best_score = results[i].detection.score;
                best_index = static_cast<int>(i);
            }
        }
        return best_index;
    }
#endif

#if LC_VISION_ENABLE_DEPTH_HISTOGRAM_DEBUG
    cv::Mat renderDepthHistogramImage(const DetectionDepthResult &result) const {
        constexpr int canvas_width = 640;
        constexpr int canvas_height = 360;
        constexpr int margin_left = 52;
        constexpr int margin_right = 18;
        constexpr int margin_top = 36;
        constexpr int margin_bottom = 56;

        cv::Mat canvas(canvas_height, canvas_width, CV_8UC3, cv::Scalar(22, 22, 24));
        const auto &counts = result.debug.histogram_counts;
        if (counts.empty()) {
            return canvas;
        }

        const int plot_width = canvas_width - margin_left - margin_right;
        const int plot_height = canvas_height - margin_top - margin_bottom;
        const int max_count = std::max(1, *std::max_element(counts.begin(), counts.end()));
        const int bin_count = static_cast<int>(counts.size());

        cv::rectangle(
            canvas, cv::Rect(margin_left, margin_top, plot_width, plot_height), cv::Scalar(70, 70, 74), 1, cv::LINE_AA);

        for (int i = 0; i < bin_count; ++i) {
            const float x0f = static_cast<float>(margin_left) + (static_cast<float>(i) / static_cast<float>(bin_count)) * plot_width;
            const float x1f = static_cast<float>(margin_left) + (static_cast<float>(i + 1) / static_cast<float>(bin_count)) * plot_width;
            const int x0 = static_cast<int>(std::floor(x0f));
            const int x1 = std::max(x0 + 1, static_cast<int>(std::ceil(x1f)));
            const int bar_height = static_cast<int>(std::lround((static_cast<double>(counts[static_cast<size_t>(i)]) / max_count) * plot_height));
            const cv::Scalar color =
                (i >= result.debug.peak_bin_start && i <= result.debug.peak_bin_end) ? cv::Scalar(60, 210, 120)
                                                                                      : cv::Scalar(160, 140, 90);
            cv::rectangle(
                canvas, cv::Rect(x0, margin_top + plot_height - bar_height, std::max(1, x1 - x0), std::max(1, bar_height)),
                color, cv::FILLED);
        }

        const std::string title =
            result.detection.label + " " + cv::format("%.0fmm", static_cast<double>(result.depth_mm));
        cv::putText(
            canvas, title, cv::Point(18, 24), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(240, 240, 240), 2, cv::LINE_AA);
        cv::putText(
            canvas,
            "peak=[" + cv::format("%.0f", static_cast<double>(result.peak_min_mm)) + "," +
                cv::format("%.0f", static_cast<double>(result.peak_max_mm)) + "] mm",
            cv::Point(18, canvas_height - 24), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(210, 210, 210), 1,
            cv::LINE_AA);
        return canvas;
    }

    void updateHistogramDebugFrame(const std::vector<DetectionDepthResult> &results, const int64_t frame_index, const rclcpp::Time &stamp) {
        const int best_index = findBestDebugDetectionIndex(results);
        if (best_index < 0) {
            return;
        }

        cv::Mat histogram_rgb = renderDepthHistogramImage(results[static_cast<size_t>(best_index)]);
        std::lock_guard<std::mutex> lock(depth_histogram_buffer.mutex);
        depth_histogram_buffer.width = histogram_rgb.cols;
        depth_histogram_buffer.height = histogram_rgb.rows;
        depth_histogram_buffer.frame_index = frame_index;
        depth_histogram_buffer.stamp = stamp;
        depth_histogram_buffer.available = true;
        depth_histogram_buffer.data.assign(
            histogram_rgb.data, histogram_rgb.data + static_cast<ptrdiff_t>(histogram_rgb.total() * histogram_rgb.elemSize()));
    }
#endif

#if LC_VISION_ENABLE_DEPTH_PEAK_MASK_DEBUG
    cv::Mat renderPeakMaskDebugImage(
        const std::vector<int16_t> &depth_data, const cv::Size &depth_size, const DetectionDepthResult &result) const {
        std::vector<uint8_t> depth_gray;
        buildDepthVisualization(depth_data, depth_size.width, depth_size.height, depth_gray);
        cv::Mat depth_gray_mat(depth_size.height, depth_size.width, CV_8UC1, depth_gray.data());
        cv::Mat debug_rgb;
        cv::cvtColor(depth_gray_mat, debug_rgb, cv::COLOR_GRAY2RGB);

        debug_rgb.convertTo(debug_rgb, -1, 0.35, 0.0);
        const cv::Rect roi = result.debug.depth_roi & cv::Rect(0, 0, depth_size.width, depth_size.height);
        for (int y = roi.y; y < roi.y + roi.height; ++y) {
            const size_t row_offset = static_cast<size_t>(y) * static_cast<size_t>(depth_size.width);
            for (int x = roi.x; x < roi.x + roi.width; ++x) {
                const int depth_mm = depth_data[row_offset + static_cast<size_t>(x)];
                cv::Vec3b &pixel = debug_rgb.at<cv::Vec3b>(y, x);
                if (depth_mm > 0 && static_cast<float>(depth_mm) >= result.peak_min_mm &&
                    static_cast<float>(depth_mm) <= result.peak_max_mm) {
                    pixel = cv::Vec3b(60, 220, 100);
                } else {
                    pixel = cv::Vec3b(
                        static_cast<uint8_t>(std::min(255, pixel[0] / 2)),
                        static_cast<uint8_t>(std::min(255, pixel[1] / 2)),
                        static_cast<uint8_t>(std::min(255, pixel[2] / 2)));
                }
            }
        }

        drawDetections(debug_rgb, std::vector<DetectionDepthResult>{result}, depth_size, true, true);
        return debug_rgb;
    }

    void updatePeakMaskDebugFrame(
        const std::vector<int16_t> &depth_data, const cv::Size &depth_size, const std::vector<DetectionDepthResult> &results,
        const int64_t frame_index, const rclcpp::Time &stamp) {
        const int best_index = findBestDebugDetectionIndex(results);
        if (best_index < 0) {
            return;
        }

        cv::Mat debug_rgb = renderPeakMaskDebugImage(depth_data, depth_size, results[static_cast<size_t>(best_index)]);
        std::lock_guard<std::mutex> lock(depth_peak_mask_buffer.mutex);
        depth_peak_mask_buffer.width = debug_rgb.cols;
        depth_peak_mask_buffer.height = debug_rgb.rows;
        depth_peak_mask_buffer.frame_index = frame_index;
        depth_peak_mask_buffer.stamp = stamp;
        depth_peak_mask_buffer.available = true;
        depth_peak_mask_buffer.data.assign(
            debug_rgb.data, debug_rgb.data + static_cast<ptrdiff_t>(debug_rgb.total() * debug_rgb.elemSize()));
    }
#endif

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
                std::vector<int16_t> depth_data;
                rclcpp::Time         depth_stamp;
                int                  depth_width = 0;
                int                  depth_height = 0;
                int64_t              depth_frame_index = -1;

                std::vector<DetectionDepthResult> results;
                if (depth.enable && copyLatestDepthFrame(depth_data, depth_width, depth_height, depth_frame_index, depth_stamp)) {
                    results = estimateDepthResults(detections, cv::Size(width, height), depth_data, cv::Size(depth_width, depth_height));
                } else {
                    results.reserve(detections.size());
                    for (const auto &detection : detections) {
                        DetectionDepthResult result;
                        result.detection = detection;
                        results.push_back(std::move(result));
                    }
                }

                updateDetectionCache(results, width, height, frame_index, stamp);
                publishDetectionDepths(results, width, height, stamp);
                publishDepthDebugMarkers(results, stamp);

#if LC_VISION_ENABLE_DEPTH_HISTOGRAM_DEBUG
                updateHistogramDebugFrame(results, frame_index, stamp);
#endif
#if LC_VISION_ENABLE_DEPTH_PEAK_MASK_DEBUG
                if (!depth_data.empty() && depth_width > 0 && depth_height > 0) {
                    updatePeakMaskDebugFrame(depth_data, cv::Size(depth_width, depth_height), results, frame_index, stamp);
                }
#endif
            } catch (const std::exception &ex) {
                detector_runtime_enabled.store(false);
                RCLCPP_ERROR(node.get_logger(), "Detector inference failed, detector disabled: %s", ex.what());
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
            drawDetections(rgb, detections, source_size, depth.estimation.annotate_depth_on_rgb);
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

        buildDepthVisualization(depth_data, width, height, depth_visualization);

        cv::Mat depth_gray(height, width, CV_8UC1, depth_visualization.data());
        cv::Mat depth_rgb;
        cv::cvtColor(depth_gray, depth_rgb, cv::COLOR_GRAY2RGB);

        if (detector_runtime_enabled.load()) {
            cv::Size source_size;
            auto     detections = snapshotDetections(source_size);
            drawDetections(depth_rgb, detections, source_size, depth.estimation.annotate_depth_on_depth);
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

#if LC_VISION_ENABLE_DEPTH_HISTOGRAM_DEBUG
    [[nodiscard]] std::optional<foxglove_msgs::msg::CompressedVideo> makeDepthHistogramMessage() {
        std::vector<uint8_t> rgb_data;
        rclcpp::Time         stamp;
        int                  width       = 0;
        int                  height      = 0;
        int64_t              frame_index = -1;

        {
            std::lock_guard<std::mutex> lock(depth_histogram_buffer.mutex);
            if (!depth_histogram_buffer.available || depth_histogram_buffer.frame_index == last_depth_histogram_frame_index) {
                return std::nullopt;
            }

            rgb_data = depth_histogram_buffer.data;
            stamp = depth_histogram_buffer.stamp;
            width = depth_histogram_buffer.width;
            height = depth_histogram_buffer.height;
            frame_index = depth_histogram_buffer.frame_index;
        }

        if (width <= 0 || height <= 0 || rgb_data.empty()) {
            return std::nullopt;
        }

        if (depth_histogram_encoder_width != width || depth_histogram_encoder_height != height) {
            depth_histogram_encoder.initialize(
                "depth_histogram", width, height, depth.publish_fps, depth.bitrate_kbps, depth.gop_size, AV_PIX_FMT_RGB24);
            depth_histogram_encoder_width = width;
            depth_histogram_encoder_height = height;
        }

        const auto encoded = depth_histogram_encoder.encode(rgb_data.data(), width * 3);
        if (!encoded.has_value()) {
            return std::nullopt;
        }

        foxglove_msgs::msg::CompressedVideo msg;
        msg.timestamp = toBuiltinTime(stamp);
        msg.frame_id = depth.frame_id;
        msg.format = "h264";
        msg.data = *encoded;
        last_depth_histogram_frame_index = frame_index;
        return msg;
    }
#endif

#if LC_VISION_ENABLE_DEPTH_PEAK_MASK_DEBUG
    [[nodiscard]] std::optional<foxglove_msgs::msg::CompressedVideo> makeDepthPeakMaskMessage() {
        std::vector<uint8_t> rgb_data;
        rclcpp::Time         stamp;
        int                  width       = 0;
        int                  height      = 0;
        int64_t              frame_index = -1;

        {
            std::lock_guard<std::mutex> lock(depth_peak_mask_buffer.mutex);
            if (!depth_peak_mask_buffer.available || depth_peak_mask_buffer.frame_index == last_depth_peak_mask_frame_index) {
                return std::nullopt;
            }

            rgb_data = depth_peak_mask_buffer.data;
            stamp = depth_peak_mask_buffer.stamp;
            width = depth_peak_mask_buffer.width;
            height = depth_peak_mask_buffer.height;
            frame_index = depth_peak_mask_buffer.frame_index;
        }

        if (width <= 0 || height <= 0 || rgb_data.empty()) {
            return std::nullopt;
        }

        if (depth_peak_mask_encoder_width != width || depth_peak_mask_encoder_height != height) {
            depth_peak_mask_encoder.initialize(
                "depth_peak_mask", width, height, depth.publish_fps, depth.bitrate_kbps, depth.gop_size, AV_PIX_FMT_RGB24);
            depth_peak_mask_encoder_width = width;
            depth_peak_mask_encoder_height = height;
        }

        const auto encoded = depth_peak_mask_encoder.encode(rgb_data.data(), width * 3);
        if (!encoded.has_value()) {
            return std::nullopt;
        }

        foxglove_msgs::msg::CompressedVideo msg;
        msg.timestamp = toBuiltinTime(stamp);
        msg.frame_id = depth.frame_id;
        msg.format = "h264";
        msg.data = *encoded;
        last_depth_peak_mask_frame_index = frame_index;
        return msg;
    }
#endif

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
#if LC_VISION_ENABLE_DEPTH_HISTOGRAM_DEBUG
    FrameBuffer<uint8_t> depth_histogram_buffer;
#endif
#if LC_VISION_ENABLE_DEPTH_PEAK_MASK_DEBUG
    FrameBuffer<uint8_t> depth_peak_mask_buffer;
#endif

    VideoEncoder rgb_encoder;
    VideoEncoder depth_encoder;
#if LC_VISION_ENABLE_DEPTH_HISTOGRAM_DEBUG
    VideoEncoder depth_histogram_encoder;
#endif
#if LC_VISION_ENABLE_DEPTH_PEAK_MASK_DEBUG
    VideoEncoder depth_peak_mask_encoder;
#endif

    std::vector<uint8_t> depth_visualization;
    std::unique_ptr<CaptureListener> capture_listener;

    rclcpp::Publisher<foxglove_msgs::msg::CompressedVideo>::SharedPtr rgb_video_pub;
    rclcpp::Publisher<foxglove_msgs::msg::CompressedVideo>::SharedPtr depth_video_pub;
    rclcpp::Publisher<lc_vision::msg::DetectionDepthArray>::SharedPtr detections_depth_pub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr depth_debug_marker_pub;
#if LC_VISION_ENABLE_DEPTH_HISTOGRAM_DEBUG
    rclcpp::Publisher<foxglove_msgs::msg::CompressedVideo>::SharedPtr depth_histogram_video_pub;
#endif
#if LC_VISION_ENABLE_DEPTH_PEAK_MASK_DEBUG
    rclcpp::Publisher<foxglove_msgs::msg::CompressedVideo>::SharedPtr depth_peak_mask_video_pub;
#endif

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
    std::string       active_backend       = "disabled";

#if LC_VISION_DETECTOR_FRAMEWORK_NCNN
    ncnn::Net                detector_net;
    int                      detector_input_index = 0;
    std::string              detector_input_name;
    std::vector<int>         detector_output_indexes;
    std::vector<std::string> detector_output_names;
#elif LC_VISION_DETECTOR_FRAMEWORK_OPENVINO
    ov::Core                  detector_core;
    std::shared_ptr<ov::Model> detector_model;
    ov::CompiledModel         detector_compiled_model;
    ov::InferRequest          detector_infer_request;
    std::string               detector_device_name = "AUTO";
#endif

    int64_t last_rgb_frame_index             = -1;
    int64_t last_depth_frame_index           = -1;
    int64_t last_detection_input_frame_index = -1;
    int64_t last_depth_histogram_frame_index = -1;
    int64_t last_depth_peak_mask_frame_index = -1;
    int     rgb_encoder_width                = 0;
    int     rgb_encoder_height               = 0;
    int     depth_encoder_width              = 0;
    int     depth_encoder_height             = 0;
    int     depth_histogram_encoder_width    = 0;
    int     depth_histogram_encoder_height   = 0;
    int     depth_peak_mask_encoder_width    = 0;
    int     depth_peak_mask_encoder_height   = 0;
    int     last_depth_debug_marker_count    = 0;
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

        auto detection_qos = rclcpp::QoS(rclcpp::KeepLast(10));
        detection_qos.reliable();
        impl_->detections_depth_pub =
            create_publisher<lc_vision::msg::DetectionDepthArray>("~/detections_depth", detection_qos);

        if (impl_->depth.debug_marker.enable) {
            impl_->depth_debug_marker_pub = create_publisher<visualization_msgs::msg::MarkerArray>(
                impl_->depth.debug_marker.topic, rclcpp::QoS(rclcpp::KeepLast(10)).reliable());
        }

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
#if LC_VISION_ENABLE_DEPTH_HISTOGRAM_DEBUG
            impl_->depth_histogram_video_pub =
                create_publisher<foxglove_msgs::msg::CompressedVideo>("~/depth_histogram/video", video_qos);
#endif
#if LC_VISION_ENABLE_DEPTH_PEAK_MASK_DEBUG
            impl_->depth_peak_mask_video_pub =
                create_publisher<foxglove_msgs::msg::CompressedVideo>("~/depth_peak_mask/video", video_qos);
#endif
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
#if LC_VISION_ENABLE_DEPTH_HISTOGRAM_DEBUG
                        if (impl_->depth_histogram_video_pub != nullptr) {
                            const auto histogram_msg = impl_->makeDepthHistogramMessage();
                            if (histogram_msg.has_value()) {
                                impl_->depth_histogram_video_pub->publish(*histogram_msg);
                            }
                        }
#endif
#if LC_VISION_ENABLE_DEPTH_PEAK_MASK_DEBUG
                        if (impl_->depth_peak_mask_video_pub != nullptr) {
                            const auto peak_mask_msg = impl_->makeDepthPeakMaskMessage();
                            if (peak_mask_msg.has_value()) {
                                impl_->depth_peak_mask_video_pub->publish(*peak_mask_msg);
                            }
                        }
#endif
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
#if LC_VISION_ENABLE_DEPTH_HISTOGRAM_DEBUG
        impl_->depth_histogram_encoder.reset();
#endif
#if LC_VISION_ENABLE_DEPTH_PEAK_MASK_DEBUG
        impl_->depth_peak_mask_encoder.reset();
#endif
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
#if LC_VISION_ENABLE_DEPTH_HISTOGRAM_DEBUG
    impl_->depth_histogram_encoder.reset();
#endif
#if LC_VISION_ENABLE_DEPTH_PEAK_MASK_DEBUG
    impl_->depth_peak_mask_encoder.reset();
#endif
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
    impl_->depth.roi_min_valid_mm     = this->declare_parameter<int>("depth.roi_min_valid_mm", 300);
    impl_->depth.roi_max_valid_mm     = this->declare_parameter<int>("depth.roi_max_valid_mm", 5000);
    impl_->depth.invalid_as_black     = this->declare_parameter<bool>("depth.invalid_as_black", true);
    impl_->depth.estimation.min_valid_pixels =
        this->declare_parameter<int>("depth.estimation.min_valid_pixels", 50);
    impl_->depth.estimation.histogram_bin_size_mm =
        this->declare_parameter<int>("depth.estimation.histogram_bin_size_mm", 50);
    impl_->depth.estimation.peak_min_ratio =
        static_cast<float>(this->declare_parameter<double>("depth.estimation.peak_min_ratio", 0.25));
    impl_->depth.estimation.peak_min_count =
        this->declare_parameter<int>("depth.estimation.peak_min_count", 5);
    impl_->depth.estimation.min_peak_pixels =
        this->declare_parameter<int>("depth.estimation.min_peak_pixels", 20);
    impl_->depth.estimation.trim_ratio =
        static_cast<float>(this->declare_parameter<double>("depth.estimation.trim_ratio", 0.1));
    impl_->depth.estimation.annotate_depth_on_rgb =
        this->declare_parameter<bool>("depth.estimation.annotate_depth_on_rgb", true);
    impl_->depth.estimation.annotate_depth_on_depth =
        this->declare_parameter<bool>("depth.estimation.annotate_depth_on_depth", true);
    impl_->depth.debug_marker.enable =
        this->declare_parameter<bool>("depth.debug_marker.enable", false);
    impl_->depth.debug_marker.topic =
        this->declare_parameter<std::string>("depth.debug_marker.topic", "~/depth_debug_marker");
    impl_->depth.debug_marker.frame_id =
        this->declare_parameter<std::string>("depth.debug_marker.frame_id", "base_link");
    impl_->depth.debug_marker.z =
        static_cast<float>(this->declare_parameter<double>("depth.debug_marker.z", 0.05));
    impl_->depth.debug_marker.line_width =
        static_cast<float>(this->declare_parameter<double>("depth.debug_marker.line_width", 0.02));
    impl_->depth.debug_marker.alpha =
        static_cast<float>(this->declare_parameter<double>("depth.debug_marker.alpha", 0.9));
    impl_->depth.debug_marker.use_detection_label_ns =
        this->declare_parameter<bool>("depth.debug_marker.use_detection_label_ns", true);

    impl_->detector.enable = this->declare_parameter<bool>("detector.enable", true);
    impl_->detector.backend = this->declare_parameter<std::string>("detector.backend", "auto");
    impl_->detector.model_dir =
        this->declare_parameter<std::string>("detector.model_dir", defaultDetectorModelDir());
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
        get_logger(),
        "Depth config: enable=%s capture=%dx%d@%d publish=%d bitrate=%dkbps gop=%d vis=[%d,%d]mm roi_valid=[%d,%d]mm",
        impl_->depth.enable ? "true" : "false", impl_->depth.width, impl_->depth.height, impl_->depth.capture_fps,
        impl_->depth.publish_fps, impl_->depth.bitrate_kbps, impl_->depth.gop_size,
        impl_->depth.visualization_min_mm, impl_->depth.visualization_max_mm, impl_->depth.roi_min_valid_mm,
        impl_->depth.roi_max_valid_mm);
    RCLCPP_INFO(
        get_logger(),
        "Detector config: enable=%s backend=%s model_dir=%s input=%d score=%.2f nms=%.2f fps=%d vulkan_device=%d cpu_threads=%d",
        impl_->detector.enable ? "true" : "false", impl_->detector.backend.c_str(), impl_->detector.model_dir.c_str(),
        impl_->detector.input_size, impl_->detector.score_threshold, impl_->detector.nms_threshold, impl_->detector.fps,
        impl_->detector.vulkan_device_index, impl_->detector.cpu_num_threads);
    RCLCPP_INFO(
        get_logger(),
        "Depth estimation config: min_valid=%d bin=%dmm peak_ratio=%.2f peak_min=%d min_peak=%d trim=%.2f annotate_rgb=%s annotate_depth=%s marker=%s",
        impl_->depth.estimation.min_valid_pixels, impl_->depth.estimation.histogram_bin_size_mm,
        impl_->depth.estimation.peak_min_ratio, impl_->depth.estimation.peak_min_count,
        impl_->depth.estimation.min_peak_pixels, impl_->depth.estimation.trim_ratio,
        impl_->depth.estimation.annotate_depth_on_rgb ? "true" : "false",
        impl_->depth.estimation.annotate_depth_on_depth ? "true" : "false",
        impl_->depth.debug_marker.enable ? "true" : "false");
}

void LCVision::prepareModel() {
    impl_->detector_ready = false;
    impl_->detector_runtime_enabled.store(false);
    impl_->cleanupGpu();

#if LC_VISION_DETECTOR_FRAMEWORK_OPENVINO
    if (!impl_->detector.enable) {
        RCLCPP_INFO(get_logger(), "OpenVINO detector disabled by parameter.");
        return;
    }

    if (!impl_->rgb.enable) {
        RCLCPP_WARN(get_logger(), "OpenVINO detector requires RGB input; detector disabled because rgb.enable=false.");
        return;
    }

    try {
        const fs::path model_dir = impl_->resolveModelDirectory();
        if (!fs::exists(model_dir) || !fs::is_directory(model_dir)) {
            throw std::runtime_error("Model directory does not exist: " + model_dir.string());
        }

        fs::path model_xml;
        for (const auto &entry : fs::directory_iterator(model_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".xml") {
                model_xml = entry.path();
                break;
            }
        }
        if (model_xml.empty()) {
            throw std::runtime_error("No .xml file found in model directory: " + model_dir.string());
        }

        const fs::path model_bin = model_xml.parent_path() / (model_xml.stem().string() + ".bin");
        if (!fs::exists(model_bin)) {
            throw std::runtime_error("OpenVINO .bin file not found next to " + model_xml.string());
        }

        const fs::path metadata_path = model_dir / "metadata.yaml";
        const ParsedMetadata metadata = parseMetadataFile(metadata_path);
        if (!metadata.class_names.empty()) {
            impl_->detector.class_names = metadata.class_names;
        }

        impl_->configureDetectorBackend();
        impl_->detector_model          = impl_->detector_core.read_model(model_xml.string());
        impl_->detector_compiled_model = impl_->detector_core.compile_model(impl_->detector_model, impl_->detector_device_name);
        impl_->detector_infer_request  = impl_->detector_compiled_model.create_infer_request();

        const auto inputs = impl_->detector_compiled_model.inputs();
        const auto outputs = impl_->detector_compiled_model.outputs();
        if (inputs.empty()) {
            throw std::runtime_error("OpenVINO model exposes no input tensors");
        }
        if (outputs.empty()) {
            throw std::runtime_error("OpenVINO model exposes no output tensors");
        }

        const auto input_shape = inputs.front().get_shape();
        if (input_shape.size() == 4 && input_shape[2] > 0 && input_shape[3] > 0) {
            impl_->detector.input_size = static_cast<int>(input_shape[2]);
        } else if (metadata.input_size > 0) {
            impl_->detector.input_size = metadata.input_size;
        }

        impl_->updateTargetClassIds();
        impl_->detector_ready = true;
        impl_->detector_runtime_enabled.store(true);

        RCLCPP_INFO(
            get_logger(), "Loaded OpenVINO detector from %s using device '%s' and input size %d",
            model_dir.string().c_str(), impl_->detector_device_name.c_str(), impl_->detector.input_size);
    } catch (const std::exception &ex) {
        RCLCPP_ERROR(get_logger(), "Failed to prepare OpenVINO detector, overlays disabled: %s", ex.what());
        impl_->detector_ready = false;
        impl_->detector_runtime_enabled.store(false);
        impl_->cleanupGpu();
    }
#elif LC_VISION_DETECTOR_FRAMEWORK_NCNN
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
#else
    if (impl_->detector.enable) {
        RCLCPP_WARN(get_logger(), "Neither OpenVINO nor NCNN was found at build time; detector overlays are disabled.");
    }
    return;
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
