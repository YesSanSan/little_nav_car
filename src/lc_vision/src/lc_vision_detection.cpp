#include "lc_vision_internal.hpp"

namespace lc_vision {

std::vector<Detection> LCVision::Impl::runDetection(
    const std::vector<uint8_t> &rgb_data, const int width, const int height, const int input_size) {
    if (!detector_ready || detector_backend == nullptr || rgb_data.empty() || width <= 0 || height <= 0) {
        return {};
    }

    return detector_backend->infer(rgb_data, width, height, input_size, detector);
}

void LCVision::Impl::detectionLoop() {
    auto       next_detection_time = std::chrono::steady_clock::now();
    const auto detection_period = std::chrono::milliseconds(std::max(1, 1000 / std::max(1, detector.fps)));

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
        int                  width = 0;
        int                  height = 0;
        int64_t              frame_index = -1;
        if (!copyLatestRgbFrame(rgb_data, width, height, frame_index, stamp)) {
            std::this_thread::sleep_for(5ms);
            continue;
        }

        next_detection_time = now + detection_period;
        last_detection_input_frame_index = frame_index;

        try {
            auto                 detections = runDetection(rgb_data, width, height, detector.input_size);
            std::vector<int16_t> depth_data;
            rclcpp::Time         depth_stamp;
            int                  depth_width = 0;
            int                  depth_height = 0;
            int64_t              depth_frame_index = -1;

            std::vector<DetectionDepthResult> results;
            if (depth.enable && copyLatestDepthFrame(depth_data, depth_width, depth_height, depth_frame_index, depth_stamp)) {
                results =
                    estimateDepthResults(detections, cv::Size(width, height), depth_data, cv::Size(depth_width, depth_height));
            } else {
                results.reserve(detections.size());
                for (const auto &detection : detections) {
                    DetectionDepthResult result;
                    result.detection = detection;
                    results.push_back(std::move(result));
                }
            }

            populateCameraPoints(results);
            maybeDispatchNavigationGoal(results, width, height, stamp);
            updateDetectionCache(results, width, height, frame_index, stamp);
            publishDetectionDepths(results, width, height, stamp);
            publishDepthDebugMarkers(results, stamp);

            if (isHistogramDebugVideoEnabled()) {
                updateHistogramDebugFrame(results, frame_index, stamp);
            }
            if (isPeakMaskDebugVideoEnabled() && !depth_data.empty() && depth_width > 0 && depth_height > 0) {
                updatePeakMaskDebugFrame(depth_data, cv::Size(depth_width, depth_height), results, frame_index, stamp);
            }
        } catch (const std::exception &ex) {
            detector_runtime_enabled.store(false);
            RCLCPP_ERROR(node.get_logger(), "Detector inference failed, detector disabled: %s", ex.what());
            updateDetectionCache({}, width, height, frame_index, stamp);
        }
    }
}

} // namespace lc_vision
