#include "lc_vision_internal.hpp"

namespace lc_vision {

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

int normalizeRotationDegrees(int rotation_degrees) {
    int normalized = rotation_degrees % 360;
    if (normalized < 0) {
        normalized += 360;
    }
    return normalized;
}

int rotationDegreesToQuarterTurns(const int rotation_degrees) {
    const int normalized = normalizeRotationDegrees(rotation_degrees);
    if (normalized % 90 != 0) {
        throw std::runtime_error("image.rotation_degrees must be one of 0, 90, 180, 270");
    }
    return normalized / 90;
}

cv::Point2f rotatePoint(const cv::Point2f &point, const cv::Size &source_size, const int quarter_turns) {
    const int normalized_turns = ((quarter_turns % 4) + 4) % 4;
    switch (normalized_turns) {
    case 0:
        return point;
    case 1:
        return cv::Point2f(
            static_cast<float>(source_size.height) - 1.0f - point.y,
            point.x);
    case 2:
        return cv::Point2f(
            static_cast<float>(source_size.width) - 1.0f - point.x,
            static_cast<float>(source_size.height) - 1.0f - point.y);
    case 3:
        return cv::Point2f(
            point.y,
            static_cast<float>(source_size.width) - 1.0f - point.x);
    default:
        return point;
    }
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

geometry_msgs::msg::Quaternion quaternionFromYaw(const double yaw) {
    tf2::Quaternion tf_quaternion;
    tf_quaternion.setRPY(0.0, 0.0, yaw);
    return tf2::toMsg(tf_quaternion);
}

double rectIou(const cv::Rect &lhs, const cv::Rect &rhs) {
    const cv::Rect intersection = lhs & rhs;
    if (intersection.area() <= 0) {
        return 0.0;
    }

    const double union_area = static_cast<double>(lhs.area() + rhs.area() - intersection.area());
    if (union_area <= 0.0) {
        return 0.0;
    }

    return static_cast<double>(intersection.area()) / union_area;
}

std::vector<uint8_t> extractAnnexBExtradata(const AVCodecContext *codec_context) {
    if (codec_context == nullptr || codec_context->extradata == nullptr || codec_context->extradata_size <= 0) {
        return {};
    }

    const auto *extradata = reinterpret_cast<const uint8_t *>(codec_context->extradata);
    const int   size = codec_context->extradata_size;

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
            const std::string name = unquote(match.str(2));
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
    const float intersection = intersection_w * intersection_h;
    const float union_area = a.area() + b.area() - intersection;

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
    const bool annotate_depth, const bool annotate_peak_range) {
    if (image.empty() || source_size.width <= 0 || source_size.height <= 0) {
        return;
    }

    const float  scale_x = static_cast<float>(image.cols) / static_cast<float>(source_size.width);
    const float  scale_y = static_cast<float>(image.rows) / static_cast<float>(source_size.height);
    const int    thickness = std::max(1, std::min(image.cols, image.rows) / 240);
    const int    font_thickness = std::max(1, thickness);
    const double font_scale = std::max(0.45, std::min(image.cols, image.rows) / 800.0);

    for (const auto &result : results) {
        const auto &detection = result.detection;
        cv::Rect    scaled_box(
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
        const int bg_y = text_y - text_size.height - baseline - 4;
        const int bg_w = std::min(text_size.width + 8, image.cols - text_x);
        const int bg_h = text_size.height + baseline + 8;

        const cv::Scalar box_color = result.depth_valid ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 140, 255);
        cv::rectangle(image, scaled_box, box_color, thickness, cv::LINE_AA);
        cv::rectangle(image, cv::Rect(text_x, bg_y, bg_w, bg_h), box_color * 0.38, cv::FILLED);
        cv::putText(
            image, caption, cv::Point(text_x + 4, text_y - 4), cv::FONT_HERSHEY_SIMPLEX, font_scale,
            cv::Scalar(255, 255, 255), font_thickness, cv::LINE_AA);
    }
}

VideoEncoder::~VideoEncoder() {
    reset();
}

void VideoEncoder::initialize(
    const std::string &stream_name, const int width, const int height, const int fps, const int bitrate_kbps,
    const int gop_size, const AVPixelFormat input_format) {
    reset();

    stream_name_ = stream_name;
    width_ = width;
    height_ = height;
    fps_ = std::max(1, fps);
    bitrate_bps_ = std::max(1, bitrate_kbps) * 1000;
    gop_size_ = std::max(1, gop_size);
    input_format_ = input_format;
    frame_counter_ = 0;

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

    codec_context_->codec_type = AVMEDIA_TYPE_VIDEO;
    codec_context_->codec_id = codec_->id;
    codec_context_->width = width_;
    codec_context_->height = height_;
    codec_context_->time_base = AVRational{1, fps_};
    codec_context_->framerate = AVRational{fps_, 1};
    codec_context_->pix_fmt = AV_PIX_FMT_YUV420P;
    codec_context_->bit_rate = bitrate_bps_;
    codec_context_->gop_size = gop_size_;
    codec_context_->max_b_frames = 0;
    codec_context_->thread_count = 1;
    codec_context_->thread_type = FF_THREAD_SLICE;
    codec_context_->flags |= AV_CODEC_FLAG_LOW_DELAY;

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
    frame_->width = width_;
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
        nullptr, width_, height_, input_format_, width_, height_, codec_context_->pix_fmt, SWS_FAST_BILINEAR, nullptr,
        nullptr, nullptr);
    if (sws_context_ == nullptr) {
        throw std::runtime_error("Failed to create swscale context for " + stream_name_);
    }

    annexb_extradata_ = extractAnnexBExtradata(codec_context_);
}

std::optional<std::vector<uint8_t>> VideoEncoder::encode(const uint8_t *data, const int stride_bytes) {
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

void VideoEncoder::reset() {
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
    input_format_ = AV_PIX_FMT_NONE;
    width_ = 0;
    height_ = 0;
    fps_ = 0;
    bitrate_bps_ = 0;
    gop_size_ = 0;
    frame_counter_ = 0;
}

LCVision::Impl::Impl(LCVision &owner)
    : node(owner),
      capture_listener(std::make_unique<CaptureListener>(*this)),
      tf_buffer(std::make_unique<tf2_ros::Buffer>(owner.get_clock())),
      tf_listener(std::make_shared<tf2_ros::TransformListener>(
          *tf_buffer, owner.get_node_base_interface(), owner.get_node_logging_interface(),
          owner.get_node_parameters_interface(), owner.get_node_topics_interface(), false)) {
}

fs::path LCVision::Impl::resolveModelDirectory(const DetectorBackend &backend) const {
    const std::string configured_model_dir = detector.model_dir.empty() ? backend.defaultModelDir() : detector.model_dir;
    fs::path          model_path = fs::path(expandHomeDirectory(configured_model_dir));

    const auto has_required_model = [&backend](const fs::path &dir) {
        if (!fs::exists(dir) || !fs::is_directory(dir)) {
            return false;
        }

        const auto extension = backend.modelFileExtension();
        for (const auto &entry : fs::directory_iterator(dir)) {
            if (entry.is_regular_file() && entry.path().extension() == extension) {
                return true;
            }
        }
        return false;
    };

    if (model_path.is_absolute() && has_required_model(model_path)) {
        return model_path;
    }

    try {
        const auto package_share = fs::path(ament_index_cpp::get_package_share_directory("lc_vision"));
        const auto installed_path = package_share / model_path;
        if (has_required_model(installed_path)) {
            return installed_path;
        }
    } catch (...) {
    }

    const fs::path source_package_dir = fs::path(__FILE__).parent_path().parent_path();
    const fs::path source_path = source_package_dir / model_path;
    if (has_required_model(source_path)) {
        return source_path;
    }

    if (configured_model_dir != backend.defaultModelDir()) {
        const fs::path fallback_path = fs::path(backend.defaultModelDir());

        try {
            const auto package_share = fs::path(ament_index_cpp::get_package_share_directory("lc_vision"));
            const auto installed_fallback = package_share / fallback_path;
            if (has_required_model(installed_fallback)) {
                return installed_fallback;
            }
        } catch (...) {
        }

        const fs::path source_fallback = source_package_dir / fallback_path;
        if (has_required_model(source_fallback)) {
            return source_fallback;
        }
    }

    return model_path;
}

void LCVision::Impl::updateTargetClassIds() {
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

void LCVision::Impl::resetDetectorState() {
    if (detector_backend != nullptr) {
        detector_backend->reset();
        detector_backend.reset();
    }
    detector_runtime_enabled.store(false);
    detector_ready = false;
    resolved_detector_framework = DetectorFramework::None;
    active_backend = "disabled";
    detected_gpu_count = 0;
    active_gpu_index = -1;
    last_detection_input_frame_index = -1;
}

void LCVision::Impl::storeColorFrame(const astra::ColorFrame &frame) {
    if (!rgb.enable) {
        return;
    }

    std::lock_guard<std::mutex> lock(rgb_buffer.mutex);
    rgb_buffer.width = frame.width();
    rgb_buffer.height = frame.height();
    rgb_buffer.frame_index = static_cast<int64_t>(frame.frame_index());
    rgb_buffer.stamp = node.now();
    rgb_buffer.available = true;
    rgb_buffer.data.resize(static_cast<size_t>(frame.length()) * 3U);
    frame.copy_to(reinterpret_cast<astra::RgbPixel *>(rgb_buffer.data.data()));
}

void LCVision::Impl::storeDepthFrame(const astra::DepthFrame &frame) {
    if (!depth.enable) {
        return;
    }

    std::lock_guard<std::mutex> lock(depth_buffer.mutex);
    depth_buffer.width = frame.width();
    depth_buffer.height = frame.height();
    depth_buffer.frame_index = static_cast<int64_t>(frame.frame_index());
    depth_buffer.stamp = node.now();
    depth_buffer.available = true;
    depth_buffer.data.resize(frame.length());
    frame.copy_to(depth_buffer.data.data());
}

void LCVision::Impl::storePointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (msg == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(pointcloud_cache.mutex);
    pointcloud_cache.pointcloud = *msg;
    pointcloud_cache.available = true;
}

void LCVision::Impl::updateDetectionCache(
    std::vector<DetectionDepthResult> detections, const int width, const int height, const int64_t frame_index,
    const rclcpp::Time &stamp) {
    std::lock_guard<std::mutex> lock(detection_cache.mutex);
    detection_cache.detections = std::move(detections);
    detection_cache.width = width;
    detection_cache.height = height;
    detection_cache.frame_index = frame_index;
    detection_cache.stamp = stamp;
    detection_cache.available = true;
}

std::vector<DetectionDepthResult> LCVision::Impl::snapshotDetections(cv::Size &source_size) {
    std::lock_guard<std::mutex> lock(detection_cache.mutex);
    source_size = cv::Size(detection_cache.width, detection_cache.height);
    return detection_cache.detections;
}

bool LCVision::Impl::copyLatestDepthFrame(
    std::vector<int16_t> &depth_data, int &width, int &height, int64_t &frame_index, rclcpp::Time &stamp) {
    std::lock_guard<std::mutex> lock(depth_buffer.mutex);
    if (!depth_buffer.available) {
        return false;
    }

    depth_data = depth_buffer.data;
    width = depth_buffer.width;
    height = depth_buffer.height;
    frame_index = depth_buffer.frame_index;
    stamp = depth_buffer.stamp;
    return !depth_data.empty() && width > 0 && height > 0;
}

bool LCVision::Impl::copyLatestRgbFrame(
    std::vector<uint8_t> &rgb_data, int &width, int &height, int64_t &frame_index, rclcpp::Time &stamp) {
    std::lock_guard<std::mutex> lock(rgb_buffer.mutex);
    if (!rgb_buffer.available || rgb_buffer.frame_index == last_detection_input_frame_index) {
        return false;
    }

    rgb_data = rgb_buffer.data;
    width = rgb_buffer.width;
    height = rgb_buffer.height;
    frame_index = rgb_buffer.frame_index;
    stamp = rgb_buffer.stamp;
    return !rgb_data.empty() && width > 0 && height > 0;
}

bool LCVision::Impl::copyLatestPointCloud(sensor_msgs::msg::PointCloud2 &pointcloud) const {
    std::lock_guard<std::mutex> lock(pointcloud_cache.mutex);
    if (!pointcloud_cache.available) {
        return false;
    }

    pointcloud = pointcloud_cache.pointcloud;
    return pointcloud.width > 0U && pointcloud.height > 0U && !pointcloud.data.empty();
}

bool LCVision::Impl::isHistogramDebugVideoEnabled() const {
    return depth.debug_video.histogram.enable && depth_histogram_video_pub != nullptr;
}

bool LCVision::Impl::isPeakMaskDebugVideoEnabled() const {
    return depth.debug_video.peak_mask.enable && depth_peak_mask_video_pub != nullptr;
}

void LCVision::Impl::rotateRgbFrame(std::vector<uint8_t> &rgb_data, int &width, int &height) const {
    const int quarter_turns = rotationDegreesToQuarterTurns(image.rotation_degrees);
    if (quarter_turns == 0 || width <= 0 || height <= 0 || rgb_data.empty()) {
        return;
    }

    cv::Mat source(height, width, CV_8UC3, rgb_data.data());
    cv::Mat rotated;
    switch (quarter_turns) {
    case 1:
        cv::rotate(source, rotated, cv::ROTATE_90_CLOCKWISE);
        break;
    case 2:
        cv::rotate(source, rotated, cv::ROTATE_180);
        break;
    case 3:
        cv::rotate(source, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);
        break;
    default:
        return;
    }

    width = rotated.cols;
    height = rotated.rows;
    rgb_data.assign(
        rotated.data,
        rotated.data + static_cast<std::ptrdiff_t>(rotated.total() * rotated.elemSize()));
}

void LCVision::Impl::rotateDepthFrame(std::vector<int16_t> &depth_data, int &width, int &height) const {
    const int quarter_turns = rotationDegreesToQuarterTurns(image.rotation_degrees);
    if (quarter_turns == 0 || width <= 0 || height <= 0 || depth_data.empty()) {
        return;
    }

    cv::Mat source(height, width, CV_16SC1, depth_data.data());
    cv::Mat rotated;
    switch (quarter_turns) {
    case 1:
        cv::rotate(source, rotated, cv::ROTATE_90_CLOCKWISE);
        break;
    case 2:
        cv::rotate(source, rotated, cv::ROTATE_180);
        break;
    case 3:
        cv::rotate(source, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);
        break;
    default:
        return;
    }

    width = rotated.cols;
    height = rotated.rows;
    depth_data.assign(
        reinterpret_cast<const int16_t *>(rotated.data),
        reinterpret_cast<const int16_t *>(rotated.data) + static_cast<std::ptrdiff_t>(rotated.total()));
}

void LCVision::Impl::resetEncoders() {
    rgb_encoder.reset();
    depth_encoder.reset();
    depth_histogram_encoder.reset();
    depth_peak_mask_encoder.reset();

    rgb_encoder_width = 0;
    rgb_encoder_height = 0;
    depth_encoder_width = 0;
    depth_encoder_height = 0;
    depth_histogram_encoder_width = 0;
    depth_histogram_encoder_height = 0;
    depth_peak_mask_encoder_width = 0;
    depth_peak_mask_encoder_height = 0;
}

void LCVision::Impl::stopThreads() {
    if (publish_thread.joinable()) {
        publish_thread.join();
    }
    if (detection_thread.joinable()) {
        detection_thread.join();
    }
    if (capture_thread.joinable()) {
        capture_thread.join();
    }
}

void LCVision::Impl::stopStreams() {
    try {
        if (reader.is_valid() && capture_listener != nullptr) {
            reader.remove_listener(*capture_listener);
        }
    } catch (const std::exception &ex) {
        RCLCPP_WARN(node.get_logger(), "Failed to remove Astra listener cleanly: %s", ex.what());
    }

    try {
        if (reader.is_valid() && rgb.enable) {
            auto color_stream = reader.stream<astra::ColorStream>();
            if (color_stream.is_available()) {
                color_stream.stop();
            }
        }
    } catch (const std::exception &ex) {
        RCLCPP_WARN(node.get_logger(), "Failed to stop Astra RGB stream cleanly: %s", ex.what());
    }

    try {
        if (reader.is_valid() && depth.enable) {
            auto depth_stream_reader = reader.stream<astra::DepthStream>();
            if (depth_stream_reader.is_available()) {
                depth_stream_reader.stop();
            }
        }
    } catch (const std::exception &ex) {
        RCLCPP_WARN(node.get_logger(), "Failed to stop Astra depth stream cleanly: %s", ex.what());
    }
}

void LCVision::Impl::shutdown() {
    running.store(false);
    stopThreads();
    stopStreams();
    resetEncoders();
    resetDetectorState();
    if (astra_initialized) {
        astra::terminate();
        astra_initialized = false;
    }
}

} // namespace lc_vision
