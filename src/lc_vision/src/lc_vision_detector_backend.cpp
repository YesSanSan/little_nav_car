#include "lc_vision_internal.hpp"

namespace lc_vision {

#if LC_VISION_HAVE_OPENVINO
std::unique_ptr<DetectorBackend> createOpenVINOBackend();
#endif

#if LC_VISION_HAVE_NCNN
std::unique_ptr<DetectorBackend> createNCNNBackend();
#endif

std::string detectorFrameworkToString(const DetectorFramework framework) {
    switch (framework) {
    case DetectorFramework::OpenVINO:
        return "openvino";
    case DetectorFramework::NCNN:
        return "ncnn";
    case DetectorFramework::None:
    default:
        return "none";
    }
}

DetectorFramework detectorFrameworkFromString(const std::string &value) {
    const auto normalized = toLower(value);
    if (normalized == "openvino") {
        return DetectorFramework::OpenVINO;
    }
    if (normalized == "ncnn") {
        return DetectorFramework::NCNN;
    }
    return DetectorFramework::None;
}

std::string availableDetectorFrameworksDescription() {
    std::vector<std::string> available;
#if LC_VISION_HAVE_OPENVINO
    available.emplace_back("openvino");
#endif
#if LC_VISION_HAVE_NCNN
    available.emplace_back("ncnn");
#endif

    if (available.empty()) {
        return "none";
    }

    std::string result;
    for (size_t i = 0; i < available.size(); ++i) {
        if (i > 0) {
            result += ",";
        }
        result += available[i];
    }
    return result;
}

std::unique_ptr<DetectorBackend> createDetectorBackend(
    const std::string &requested_framework, DetectorFramework &resolved_framework, const rclcpp::Logger &logger) {
    const auto normalized = toLower(requested_framework);

    const auto create_for_framework = [&](const DetectorFramework framework) -> std::unique_ptr<DetectorBackend> {
        switch (framework) {
        case DetectorFramework::OpenVINO:
#if LC_VISION_HAVE_OPENVINO
            return createOpenVINOBackend();
#else
            return nullptr;
#endif
        case DetectorFramework::NCNN:
#if LC_VISION_HAVE_NCNN
            return createNCNNBackend();
#else
            return nullptr;
#endif
        case DetectorFramework::None:
        default:
            return nullptr;
        }
    };

    if (normalized.empty() || normalized == "auto") {
#if LC_VISION_HAVE_OPENVINO
        resolved_framework = DetectorFramework::OpenVINO;
        return createOpenVINOBackend();
#elif LC_VISION_HAVE_NCNN
        resolved_framework = DetectorFramework::NCNN;
        return createNCNNBackend();
#else
        resolved_framework = DetectorFramework::None;
        return nullptr;
#endif
    }

    const auto requested = detectorFrameworkFromString(normalized);
    if (requested == DetectorFramework::None) {
        RCLCPP_WARN(
            logger, "Unknown detector.framework value '%s', falling back to auto. Available frameworks: %s",
            requested_framework.c_str(), availableDetectorFrameworksDescription().c_str());
        return createDetectorBackend("auto", resolved_framework, logger);
    }

    auto backend = create_for_framework(requested);
    if (backend == nullptr) {
        RCLCPP_WARN(
            logger, "Requested detector.framework '%s' is unavailable in this build. Available frameworks: %s",
            requested_framework.c_str(), availableDetectorFrameworksDescription().c_str());
        resolved_framework = DetectorFramework::None;
        return nullptr;
    }

    resolved_framework = requested;
    return backend;
}

} // namespace lc_vision
