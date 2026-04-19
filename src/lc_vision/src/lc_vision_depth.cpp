#include "lc_vision_internal.hpp"

namespace lc_vision {

cv::Rect LCVision::Impl::scaleRectToSize(
    const cv::Rect &source_box, const cv::Size &source_size, const cv::Size &target_size) const {
    if (source_size.width <= 0 || source_size.height <= 0 || target_size.width <= 0 || target_size.height <= 0) {
        return {};
    }

    const float scale_x = static_cast<float>(target_size.width) / static_cast<float>(source_size.width);
    const float scale_y = static_cast<float>(target_size.height) / static_cast<float>(source_size.height);
    cv::Rect    scaled_box(
        static_cast<int>(std::lround(static_cast<float>(source_box.x) * scale_x)),
        static_cast<int>(std::lround(static_cast<float>(source_box.y) * scale_y)),
        static_cast<int>(std::lround(static_cast<float>(source_box.width) * scale_x)),
        static_cast<int>(std::lround(static_cast<float>(source_box.height) * scale_y)));
    scaled_box &= cv::Rect(0, 0, target_size.width, target_size.height);
    return scaled_box;
}

void LCVision::Impl::populateCameraPoints(
    std::vector<DetectionDepthResult> &results, const cv::Size &rotated_depth_size, const cv::Size &raw_depth_size) const {
    if (!depth_stream.has_value()) {
        return;
    }

    const int quarter_turns = rotationDegreesToQuarterTurns(image.rotation_degrees);
    const auto mapper = depth_stream->coordinateMapper();
    for (auto &result : results) {
        if (!result.depth_valid || result.peak_pixel_count == 0U) {
            continue;
        }

        const cv::Point2f raw_centroid = rotatePoint(
            cv::Point2f(result.peak_centroid_x, result.peak_centroid_y), rotated_depth_size, -quarter_turns);
        float world_x = 0.0f;
        float world_y = 0.0f;
        float world_z = 0.0f;
        mapper.convert_depth_to_world(
            std::clamp(raw_centroid.x, 0.0f, static_cast<float>(raw_depth_size.width - 1)),
            std::clamp(raw_centroid.y, 0.0f, static_cast<float>(raw_depth_size.height - 1)), result.depth_mm, world_x,
            world_y, world_z);

        if (depth.camera_point.flip_x) {
            world_x = -world_x;
        }

        // Astra/OpenNI world coordinates use +Y upward, while ROS optical frames use +Y downward.
        result.camera_point.x = static_cast<double>(world_x) / 1000.0;
        result.camera_point.y = static_cast<double>(-world_y) / 1000.0;
        result.camera_point.z = static_cast<double>(world_z) / 1000.0;
        result.camera_point_valid = std::isfinite(result.camera_point.x) && std::isfinite(result.camera_point.y) &&
                                    std::isfinite(result.camera_point.z);
    }
}

DetectionDepthResult LCVision::Impl::estimateDepthForDetection(
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

    const int   peak_bin = static_cast<int>(std::distance(result.debug.histogram_counts.begin(), peak_it));
    const int   peak_count = *peak_it;
    const float min_ratio = std::clamp(depth.estimation.peak_min_ratio, 0.0f, 1.0f);
    const int   min_count = std::max(1, depth.estimation.peak_min_count);

    int peak_start = peak_bin;
    while (peak_start > 0) {
        const int neighbor_count = result.debug.histogram_counts[static_cast<size_t>(peak_start - 1)];
        if (neighbor_count < min_count ||
            static_cast<float>(neighbor_count) < static_cast<float>(peak_count) * min_ratio) {
            break;
        }
        --peak_start;
    }

    int peak_end = peak_bin;
    while (peak_end + 1 < histogram_bin_count) {
        const int neighbor_count = result.debug.histogram_counts[static_cast<size_t>(peak_end + 1)];
        if (neighbor_count < min_count ||
            static_cast<float>(neighbor_count) < static_cast<float>(peak_count) * min_ratio) {
            break;
        }
        ++peak_end;
    }

    result.debug.peak_bin_start = peak_start;
    result.debug.peak_bin_end = peak_end;
    result.peak_min_mm = static_cast<float>(depth.roi_min_valid_mm + peak_start * bin_size_mm);
    result.peak_max_mm = static_cast<float>(
        std::min(depth.roi_max_valid_mm, depth.roi_min_valid_mm + (peak_end + 1) * bin_size_mm - 1));

    std::vector<int> peak_depths;
    peak_depths.reserve(valid_depths.size());
    double peak_sum_x = 0.0;
    double peak_sum_y = 0.0;
    for (int y = roi.y; y < roi.y + roi.height; ++y) {
        const size_t row_offset = static_cast<size_t>(y) * static_cast<size_t>(depth_size.width);
        for (int x = roi.x; x < roi.x + roi.width; ++x) {
            const int depth_mm = depth_data[row_offset + static_cast<size_t>(x)];
            if (depth_mm <= 0) {
                continue;
            }
            if (static_cast<float>(depth_mm) < result.peak_min_mm || static_cast<float>(depth_mm) > result.peak_max_mm) {
                continue;
            }

            peak_depths.push_back(depth_mm);
            peak_sum_x += static_cast<double>(x);
            peak_sum_y += static_cast<double>(y);
        }
    }

    result.peak_pixel_count = static_cast<uint32_t>(peak_depths.size());
    if (peak_depths.size() < static_cast<size_t>(std::max(1, depth.estimation.min_peak_pixels))) {
        return result;
    }

    result.peak_centroid_x = static_cast<float>(peak_sum_x / static_cast<double>(peak_depths.size()));
    result.peak_centroid_y = static_cast<float>(peak_sum_y / static_cast<double>(peak_depths.size()));

    std::sort(peak_depths.begin(), peak_depths.end());
    const float  trim_ratio = std::clamp(depth.estimation.trim_ratio, 0.0f, 0.49f);
    const size_t trim_count =
        static_cast<size_t>(std::floor(static_cast<double>(peak_depths.size()) * trim_ratio));
    size_t keep_begin = trim_count;
    size_t keep_end = peak_depths.size() - trim_count;
    if (keep_begin >= keep_end) {
        keep_begin = 0;
        keep_end = peak_depths.size();
    }

    const auto   begin_it = peak_depths.begin() + static_cast<std::ptrdiff_t>(keep_begin);
    const auto   end_it = peak_depths.begin() + static_cast<std::ptrdiff_t>(keep_end);
    const double sum = std::accumulate(begin_it, end_it, 0.0);
    const size_t keep_count = static_cast<size_t>(std::distance(begin_it, end_it));
    if (keep_count == 0U) {
        return result;
    }

    result.depth_mm = static_cast<float>(sum / static_cast<double>(keep_count));
    result.depth_valid = true;
    return result;
}

std::vector<DetectionDepthResult> LCVision::Impl::estimateDepthResults(
    const std::vector<Detection> &detections, const cv::Size &rgb_size, const std::vector<int16_t> &depth_data,
    const cv::Size &depth_size) const {
    std::vector<DetectionDepthResult> results;
    results.reserve(detections.size());
    for (const auto &detection : detections) {
        results.push_back(estimateDepthForDetection(detection, rgb_size, depth_data, depth_size));
    }
    return results;
}

void LCVision::Impl::publishDetectionDepths(
    const std::vector<DetectionDepthResult> &results, const int width, const int height, const rclcpp::Time &stamp) {
    if (detections_depth_pub == nullptr) {
        return;
    }

    lc_vision::msg::DetectionDepthArray msg;
    msg.header.stamp = toBuiltinTime(stamp);
    msg.header.frame_id = depth.frame_id;
    msg.source_width = static_cast<uint32_t>(std::max(0, width));
    msg.source_height = static_cast<uint32_t>(std::max(0, height));
    msg.detections.reserve(results.size());

    for (const auto &result : results) {
        lc_vision::msg::DetectionDepth detection_msg;
        detection_msg.header = msg.header;
        detection_msg.label = result.detection.label;
        detection_msg.class_id = result.detection.class_id;
        detection_msg.track_id = result.track_id;
        detection_msg.score = result.detection.score;
        detection_msg.x = result.detection.box.x;
        detection_msg.y = result.detection.box.y;
        detection_msg.width = result.detection.box.width;
        detection_msg.height = result.detection.box.height;
        detection_msg.selected = result.selected;
        detection_msg.depth_valid = result.depth_valid;
        detection_msg.depth_mm = result.depth_mm;
        detection_msg.peak_min_mm = result.peak_min_mm;
        detection_msg.peak_max_mm = result.peak_max_mm;
        detection_msg.roi_valid_pixel_count = result.roi_valid_pixel_count;
        detection_msg.peak_pixel_count = result.peak_pixel_count;
        detection_msg.camera_point_valid = result.camera_point_valid;
        detection_msg.camera_point = result.camera_point;
        msg.detections.push_back(std::move(detection_msg));
    }

    detections_depth_pub->publish(msg);
}

void LCVision::Impl::publishDepthDebugMarkers(
    const std::vector<DetectionDepthResult> &results, const rclcpp::Time &stamp) {
    if (depth_debug_marker_pub == nullptr || !depth.debug_marker.enable) {
        return;
    }

    visualization_msgs::msg::MarkerArray marker_array;
    visualization_msgs::msg::Marker      clear_marker;
    clear_marker.header.stamp = toBuiltinTime(stamp);
    clear_marker.header.frame_id = depth.debug_marker.frame_id;
    clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.push_back(clear_marker);
    int marker_id = 0;
    constexpr int    circle_segments = 72;
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

void LCVision::Impl::buildDepthVisualization(
    const std::vector<int16_t> &depth_data, const int width, const int height, std::vector<uint8_t> &output) const {
    output.resize(static_cast<size_t>(width) * static_cast<size_t>(height));

    const float min_mm = static_cast<float>(depth.visualization_min_mm);
    const float max_mm = static_cast<float>(std::max(depth.visualization_max_mm, depth.visualization_min_mm + 1));
    const float range = max_mm - min_mm;

    for (size_t i = 0; i < depth_data.size(); ++i) {
        const int16_t depth_mm = depth_data[i];
        if (depth_mm <= 0) {
            output[i] = depth.invalid_as_black ? 0 : 255;
            continue;
        }

        const float clamped = std::clamp(static_cast<float>(depth_mm), min_mm, max_mm);
        const float normalized = 1.0f - ((clamped - min_mm) / range);
        output[i] = static_cast<uint8_t>(std::clamp(std::lround(normalized * 255.0f), 0L, 255L));
    }
}

int LCVision::Impl::findBestDebugDetectionIndex(const std::vector<DetectionDepthResult> &results) const {
    int   best_index = -1;
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

cv::Mat LCVision::Impl::renderDepthHistogramImage(const DetectionDepthResult &result) const {
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
        const float x0f =
            static_cast<float>(margin_left) + (static_cast<float>(i) / static_cast<float>(bin_count)) * plot_width;
        const float x1f = static_cast<float>(margin_left) +
                          (static_cast<float>(i + 1) / static_cast<float>(bin_count)) * plot_width;
        const int bar_height = static_cast<int>(
            std::lround((static_cast<double>(counts[static_cast<size_t>(i)]) / max_count) * plot_height));
        const int       x0 = static_cast<int>(std::floor(x0f));
        const int       x1 = std::max(x0 + 1, static_cast<int>(std::ceil(x1f)));
        const cv::Scalar color =
            (i >= result.debug.peak_bin_start && i <= result.debug.peak_bin_end) ? cv::Scalar(60, 210, 120)
                                                                                  : cv::Scalar(160, 140, 90);
        cv::rectangle(
            canvas, cv::Rect(x0, margin_top + plot_height - bar_height, std::max(1, x1 - x0), std::max(1, bar_height)),
            color, cv::FILLED);
    }

    const std::string title = result.detection.label + " " + cv::format("%.0fmm", static_cast<double>(result.depth_mm));
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

void LCVision::Impl::updateHistogramDebugFrame(
    const std::vector<DetectionDepthResult> &results, const int64_t frame_index, const rclcpp::Time &stamp) {
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

cv::Mat LCVision::Impl::renderPeakMaskDebugImage(
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
            const int  depth_mm = depth_data[row_offset + static_cast<size_t>(x)];
            cv::Vec3b &pixel = debug_rgb.at<cv::Vec3b>(y, x);
            if (depth_mm > 0 && static_cast<float>(depth_mm) >= result.peak_min_mm &&
                static_cast<float>(depth_mm) <= result.peak_max_mm) {
                pixel = cv::Vec3b(60, 220, 100);
            } else {
                pixel = cv::Vec3b(
                    static_cast<uint8_t>(std::min(255, pixel[0] / 2)), static_cast<uint8_t>(std::min(255, pixel[1] / 2)),
                    static_cast<uint8_t>(std::min(255, pixel[2] / 2)));
            }
        }
    }

    drawDetections(debug_rgb, std::vector<DetectionDepthResult>{result}, depth_size, true, true);
    return debug_rgb;
}

void LCVision::Impl::updatePeakMaskDebugFrame(
    const std::vector<int16_t> &depth_data, const cv::Size &depth_size,
    const std::vector<DetectionDepthResult> &results, const int64_t frame_index, const rclcpp::Time &stamp) {
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

std::optional<foxglove_msgs::msg::CompressedVideo> LCVision::Impl::makeRgbMessage() {
    std::vector<uint8_t> rgb_data;
    rclcpp::Time         stamp;
    int                  width = 0;
    int                  height = 0;
    int64_t              frame_index = -1;

    {
        std::lock_guard<std::mutex> lock(rgb_buffer.mutex);
        if (!rgb_buffer.available || rgb_buffer.frame_index == last_rgb_frame_index) {
            return std::nullopt;
        }

        rgb_data = rgb_buffer.data;
        stamp = rgb_buffer.stamp;
        width = rgb_buffer.width;
        height = rgb_buffer.height;
        frame_index = rgb_buffer.frame_index;
    }

    if (width <= 0 || height <= 0 || rgb_data.empty()) {
        return std::nullopt;
    }

    rotateRgbFrame(rgb_data, width, height);

    if (detector_runtime_enabled.load()) {
        cv::Size source_size;
        auto     detections = snapshotDetections(source_size);
        cv::Mat  rgb_mat(height, width, CV_8UC3, rgb_data.data());
        drawDetections(rgb_mat, detections, source_size, depth.estimation.annotate_depth_on_rgb);
    }

    if (rgb_encoder_width != width || rgb_encoder_height != height) {
        rgb_encoder.initialize("rgb", width, height, rgb.publish_fps, rgb.bitrate_kbps, rgb.gop_size, AV_PIX_FMT_RGB24);
        rgb_encoder_width = width;
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
    msg.frame_id = rgb.frame_id;
    msg.format = "h264";
    msg.data = *encoded;

    last_rgb_frame_index = frame_index;
    return msg;
}

std::optional<foxglove_msgs::msg::CompressedVideo> LCVision::Impl::makeDepthMessage() {
    std::vector<int16_t> depth_data;
    rclcpp::Time         stamp;
    int                  width = 0;
    int                  height = 0;
    int64_t              frame_index = -1;

    {
        std::lock_guard<std::mutex> lock(depth_buffer.mutex);
        if (!depth_buffer.available || depth_buffer.frame_index == last_depth_frame_index) {
            return std::nullopt;
        }

        depth_data = depth_buffer.data;
        stamp = depth_buffer.stamp;
        width = depth_buffer.width;
        height = depth_buffer.height;
        frame_index = depth_buffer.frame_index;
    }

    if (width <= 0 || height <= 0 || depth_data.empty()) {
        return std::nullopt;
    }

    rotateDepthFrame(depth_data, width, height);
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
        depth_encoder.initialize("depth", width, height, depth.publish_fps, depth.bitrate_kbps, depth.gop_size, AV_PIX_FMT_RGB24);
        depth_encoder_width = width;
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
    msg.frame_id = depth.frame_id;
    msg.format = "h264";
    msg.data = *encoded;

    last_depth_frame_index = frame_index;
    return msg;
}

std::optional<foxglove_msgs::msg::CompressedVideo> LCVision::Impl::makeDepthHistogramMessage() {
    std::vector<uint8_t> rgb_data;
    rclcpp::Time         stamp;
    int                  width = 0;
    int                  height = 0;
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

std::optional<foxglove_msgs::msg::CompressedVideo> LCVision::Impl::makeDepthPeakMaskMessage() {
    std::vector<uint8_t> rgb_data;
    rclcpp::Time         stamp;
    int                  width = 0;
    int                  height = 0;
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

} // namespace lc_vision
