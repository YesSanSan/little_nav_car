#pragma once

#include "lc_vision.hpp"

#include <action_msgs/msg/goal_status_array.hpp>
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
#include <sstream>
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
#include <builtin_interfaces/msg/time.hpp>
#include <foxglove_msgs/msg/compressed_video.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <lc_vision/msg/detection_depth.hpp>
#include <lc_vision/msg/detection_depth_array.hpp>
#include <lc_vision/msg/selected_target_status.hpp>
#include <nav2_msgs/msg/costmap.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/node_interfaces/node_parameters_interface.hpp>
#include <rclcpp/parameter.hpp>
#include <rclcpp/qos.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_msgs/msg/header.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/time.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace lc_vision {

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
    int              histogram_min_mm = 0;
    int              histogram_bin_size_mm = 1;
    int              peak_bin_start = -1;
    int              peak_bin_end = -1;
    cv::Rect         depth_roi;
};

enum class DepthQualityState {
    Invalid,
    Reliable,
    NearFieldOccluded,
    BackgroundSuspect,
};

struct DetectionDepthResult {
    Detection                detection;
    bool                     depth_valid = false;
    float                    depth_mm = 0.0f;
    float                    peak_min_mm = 0.0f;
    float                    peak_max_mm = 0.0f;
    uint32_t                 roi_total_pixel_count = 0;
    uint32_t                 roi_valid_pixel_count = 0;
    uint32_t                 roi_invalid_pixel_count = 0;
    float                    roi_invalid_ratio = 0.0f;
    uint32_t                 peak_pixel_count = 0;
    uint32_t                 histogram_peak_count = 0;
    float                    histogram_peak_ratio = 0.0f;
    uint32_t                 histogram_second_peak_count = 0;
    float                    histogram_second_peak_depth_mm = 0.0f;
    float                    nearest_valid_depth_mm = 0.0f;
    bool                     camera_point_valid = false;
    geometry_msgs::msg::Point camera_point;
    float                    peak_centroid_x = 0.0f;
    float                    peak_centroid_y = 0.0f;
    int32_t                  track_id = -1;
    bool                     selected = false;
    DepthQualityState        depth_quality_state = DepthQualityState::Invalid;
    bool                     control_distance_valid = false;
    float                    control_distance_mm = 0.0f;
    DepthDebugData           debug;
};

struct StreamConfig {
    bool        enable = true;
    int         width = 640;
    int         height = 480;
    int         capture_fps = 15;
    int         publish_fps = 10;
    int         bitrate_kbps = 1200;
    int         gop_size = 30;
    std::string frame_id;
};

struct DepthDebugVideoStreamConfig {
    bool        enable = false;
    std::string topic;
};

struct DepthConfig : StreamConfig {
    struct CameraPointConfig {
        bool flip_x = false;
    };

    struct EstimationConfig {
        int   min_valid_pixels = 50;
        int   histogram_bin_size_mm = 50;
        float peak_min_ratio = 0.25f;
        int   peak_min_count = 5;
        int   min_peak_pixels = 20;
        float trim_ratio = 0.1f;
        bool  near_field_enable = true;
        int   near_field_min_history_depth_mm = 450;
        int   near_field_max_expected_depth_mm = 900;
        float near_field_invalid_ratio_threshold = 0.45f;
        int   near_field_depth_jump_mm = 350;
        int   near_field_nearest_valid_margin_mm = 200;
        float near_field_history_timeout_sec = 1.0f;
        int   near_field_min_reliable_frames = 2;
        bool  annotate_depth_on_rgb = true;
        bool  annotate_depth_on_depth = true;
    };

    struct DebugMarkerConfig {
        bool        enable = false;
        std::string topic = "~/depth_debug_marker";
        std::string frame_id = "base_link";
        float       z = 0.05f;
        float       line_width = 0.02f;
        float       alpha = 0.9f;
        bool        use_detection_label_ns = true;
    };

    struct DebugVideoConfig {
        DepthDebugVideoStreamConfig histogram{false, "~/depth_histogram/video"};
        DepthDebugVideoStreamConfig peak_mask{false, "~/depth_peak_mask/video"};
    };

    int               visualization_min_mm = 300;
    int               visualization_max_mm = 5000;
    int               roi_min_valid_mm = 300;
    int               roi_max_valid_mm = 5000;
    bool              invalid_as_black = true;
    bool              enable_registration = true;
    CameraPointConfig camera_point;
    EstimationConfig  estimation;
    DebugMarkerConfig debug_marker;
    DebugVideoConfig  debug_video;
};

struct DetectorConfig {
    bool                     enable = true;
    std::string              framework = "auto";
    std::string              backend = "auto";
    std::string              model_dir;
    int                      input_size = 640;
    float                    score_threshold = 0.25f;
    float                    nms_threshold = 0.45f;
    int                      fps = 5;
    int                      vulkan_device_index = 0;
    int                      cpu_num_threads = 2;
    bool                     log_backend_info = true;
    std::vector<std::string> target_classes = {"person"};
    std::vector<std::string> class_names;
    std::set<int>            target_class_ids = {0};
};

struct TrackingConfig {
    struct LidarConfig {
        bool        enable = true;
        std::string scan_topic = "/scan_filtered";
        float       min_forward_protection_m = 0.80f;
    };

    struct NavFallbackConfig {
        bool        enable = true;
        std::string cmd_vel_topic = "/cmd_vel_nav";
        float       no_cmd_vel_timeout_sec = 1.5f;
    };

    struct MapMatchConfig {
        float distance_tolerance_m = 0.6f;
        float angle_tolerance_rad = 0.8f;
    };

    struct RecoveryConfig {
        bool        enable = true;
        std::string cmd_vel_topic = "/cmd_vel_nav";
        float       turn_speed_rad_s = 0.8f;
        float       reacquire_turn_speed_rad_s = 0.25f;
        float       lost_delay_sec = 0.2f;
        float       memory_timeout_sec = 5.0f;
        float       search_timeout_sec = 8.0f;
    };

    struct VisualServoConfig {
        bool  enable = true;
        float engage_distance_m = 1.1f;
        float lost_target_distance_m = 1.8f;
        float align_deadband_px = 36.0f;
        float forward_deadband_px = 20.0f;
        float align_band_ratio = 0.2f;
        float turn_gain = 0.9f;
        float min_turn_speed_rad_s = 0.15f;
        float max_turn_speed_rad_s = 0.45f;
        float linear_gain = 0.45f;
        float min_linear_speed_m_s = 0.08f;
        float max_linear_speed_m_s = 0.3f;
    };

    struct DebugConfig {
        bool  enable_verbose_logs = false;
        int   max_track_memories = 32;
        float track_memory_timeout_sec = 3.0f;
    };

    bool  enable = true;
    float initial_center_gate_ratio = 0.3f;
    int   max_lost_frames = 10;
    float min_iou_for_match = 0.05f;
    float max_center_distance_px = 120.0f;
    float max_depth_delta_mm = 800.0f;
    LidarConfig lidar;
    NavFallbackConfig nav_fallback;
    MapMatchConfig map_match;
    RecoveryConfig recovery;
    VisualServoConfig visual_servo;
    DebugConfig debug;
};

struct GoalConfig {
    struct DebugMarkerConfig {
        bool        enable = false;
        std::string topic = "~/target_map_debug_marker";
        std::string frame_id = "map";
        float       point_scale = 0.18f;
        float       text_scale = 0.16f;
        float       alpha = 0.9f;
        float       z = 0.08f;
        float       text_z_offset = 0.18f;
    };

    bool        enable = true;
    std::string global_frame_id = "map";
    std::string robot_frame_id = "base_link";
    std::string costmap_topic = "/global_costmap/costmap_raw";
    float       standoff_distance_m = 0.5f;
    float       max_target_distance_m = 1.0f;
    int         min_stable_frames = 3;
    float       clearance_radius_m = 0.25f;
    int         max_cell_cost = 80;
    float       longitudinal_search_step_m = 0.05f;
    float       max_backoff_distance_m = 1.5f;
    float       lateral_search_step_m = 0.1f;
    float       max_lateral_offset_m = 0.4f;
    bool        publish_debug_topics = true;
    DebugMarkerConfig debug_marker;
};

struct ImageConfig {
    int rotation_degrees = 0;
};

enum class DetectorFramework {
    None,
    OpenVINO,
    NCNN,
};

struct DetectorPreparationResult {
    std::string       active_backend = "disabled";
    int               detected_gpu_count = 0;
    int               active_gpu_index = -1;
    std::optional<int> input_size;
};

class DetectorBackend {
public:
    virtual ~DetectorBackend() = default;

    virtual DetectorFramework framework() const = 0;
    virtual std::string defaultModelDir() const = 0;
    virtual std::string modelFileExtension() const = 0;
    virtual DetectorPreparationResult prepare(
        const DetectorConfig &config, const fs::path &model_dir, const ParsedMetadata &metadata,
        const rclcpp::Logger &logger) = 0;
    virtual std::vector<Detection> infer(
        const std::vector<uint8_t> &rgb_data, int width, int height, int input_size,
        const DetectorConfig &config) = 0;
    virtual void reset() = 0;
};

std::string avErrorToString(int error_code);
std::string trim(const std::string &value);
std::string toLower(std::string value);
std::string unquote(const std::string &value);
int parseFirstInteger(const std::string &line);
int normalizeRotationDegrees(int rotation_degrees);
int rotationDegreesToQuarterTurns(int rotation_degrees);
cv::Point2f rotatePoint(const cv::Point2f &point, const cv::Size &source_size, int quarter_turns);
std::vector<std::string> defaultCocoClassNames();
std::string defaultAstraSdkRoot();
int defaultDetectorThreadCount();
std::string expandHomeDirectory(const std::string &path);
builtin_interfaces::msg::Time toBuiltinTime(const rclcpp::Time &stamp);
geometry_msgs::msg::Quaternion quaternionFromYaw(double yaw);
double rectIou(const cv::Rect &lhs, const cv::Rect &rhs);
std::vector<uint8_t> extractAnnexBExtradata(const AVCodecContext *codec_context);
ParsedMetadata parseMetadataFile(const fs::path &metadata_path);
float intersectionOverUnion(const cv::Rect2f &a, const cv::Rect2f &b);
void applyNms(std::vector<Detection> &detections, float iou_threshold);
std::string buildDetectionCaption(
    const DetectionDepthResult &result, bool annotate_depth, bool annotate_peak_range);
void drawDetections(
    cv::Mat &image, const std::vector<DetectionDepthResult> &results, const cv::Size &source_size,
    bool annotate_depth, bool annotate_peak_range = false);
std::string detectorFrameworkToString(DetectorFramework framework);
DetectorFramework detectorFrameworkFromString(const std::string &value);
std::string availableDetectorFrameworksDescription();
std::unique_ptr<DetectorBackend> createDetectorBackend(
    const std::string &requested_framework, DetectorFramework &resolved_framework, const rclcpp::Logger &logger);

template<typename ValueAt>
std::vector<Detection> decodeDetections(
    int width, int height, int pad_left, int pad_top, float scale, const std::vector<std::string> &class_names,
    const std::set<int> &target_class_ids, float score_threshold, float nms_threshold, int num_boxes,
    const ValueAt &value_at) {
    const int   class_count = std::max(1, static_cast<int>(class_names.size()));
    const float inv_scale = 1.0f / std::max(scale, 1.0e-6f);

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
        const float box_w = value_at(2, box_index);
        const float box_h = value_at(3, box_index);

        const float left = ((center_x - box_w * 0.5f) - static_cast<float>(pad_left)) * inv_scale;
        const float top = ((center_y - box_h * 0.5f) - static_cast<float>(pad_top)) * inv_scale;
        const float right = ((center_x + box_w * 0.5f) - static_cast<float>(pad_left)) * inv_scale;
        const float bottom = ((center_y + box_h * 0.5f) - static_cast<float>(pad_top)) * inv_scale;

        const int x1 = std::clamp(static_cast<int>(std::floor(left)), 0, width - 1);
        const int y1 = std::clamp(static_cast<int>(std::floor(top)), 0, height - 1);
        const int x2 = std::clamp(static_cast<int>(std::ceil(right)), x1 + 1, width);
        const int y2 = std::clamp(static_cast<int>(std::ceil(bottom)), y1 + 1, height);

        Detection detection;
        detection.box = cv::Rect(x1, y1, x2 - x1, y2 - y1);
        detection.score = best_score;
        detection.class_id = best_class;
        detection.label =
            best_class < static_cast<int>(class_names.size()) ? class_names[static_cast<size_t>(best_class)]
                                                              : std::to_string(best_class);
        detections.push_back(std::move(detection));
    }

    applyNms(detections, nms_threshold);
    return detections;
}

class VideoEncoder {
public:
    VideoEncoder() = default;
    ~VideoEncoder();

    void initialize(
        const std::string &stream_name, int width, int height, int fps, int bitrate_kbps, int gop_size,
        AVPixelFormat input_format);
    [[nodiscard]] std::optional<std::vector<uint8_t>> encode(const uint8_t *data, int stride_bytes);
    void reset();

private:
    const AVCodec  *codec_ = nullptr;
    AVCodecContext *codec_context_ = nullptr;
    AVFrame        *frame_ = nullptr;
    AVPacket       *packet_ = nullptr;
    SwsContext     *sws_context_ = nullptr;

    std::vector<uint8_t> annexb_extradata_;
    std::string          stream_name_;
    AVPixelFormat        input_format_ = AV_PIX_FMT_NONE;
    int                  width_ = 0;
    int                  height_ = 0;
    int                  fps_ = 0;
    int                  bitrate_bps_ = 0;
    int                  gop_size_ = 0;
    int64_t              frame_counter_ = 0;
};

template<typename StreamT>
astra::ImageStreamMode selectBestMode(
    StreamT &stream, int requested_width, int requested_height, int requested_fps,
    astra_pixel_format_t requested_format) {
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

template<typename T>
struct FrameBuffer {
    std::mutex     mutex;
    std::vector<T> data;
    int            width = 0;
    int            height = 0;
    int64_t        frame_index = -1;
    rclcpp::Time   stamp;
    bool           available = false;
};

struct DetectionCache {
    std::mutex                        mutex;
    std::vector<DetectionDepthResult> detections;
    int                               width = 0;
    int                               height = 0;
    int64_t                           frame_index = -1;
    rclcpp::Time                      stamp;
    bool                              available = false;
};

struct CostmapCache {
    mutable std::mutex      mutex;
    nav2_msgs::msg::Costmap costmap;
    bool                    available = false;
};

struct LaserScanCache {
    mutable std::mutex          mutex;
    sensor_msgs::msg::LaserScan scan;
    bool                        available = false;
};

struct TrackedTargetState {
    bool                 active = false;
    int32_t              current_track_id = -1;
    int32_t              next_track_id = 1;
    int                  stable_frames = 0;
    int                  lost_frames = 0;
    bool                 goal_dispatched = false;
    DetectionDepthResult last_result;
    geometry_msgs::msg::PointStamped last_map_point;
    bool                 last_map_point_valid = false;
    rclcpp::Time         last_stamp;
};

enum class RecoveryPhase {
    Inactive,
    Searching,
    ReacquiredAligning,
};

struct RememberedTargetState {
    bool                           available = false;
    rclcpp::Time                   stamp;
    geometry_msgs::msg::PointStamped map_point;
    bool                           map_point_valid = false;
    float                          image_offset_px = 0.0f;
    float                          camera_lateral_m = 0.0f;
    float                          depth_m = std::numeric_limits<float>::infinity();
    bool                           reliable_depth_available = false;
    rclcpp::Time                   reliable_depth_stamp;
    float                          reliable_depth_m = std::numeric_limits<float>::infinity();
    int                            reliable_depth_streak = 0;
    DepthQualityState              last_depth_quality_state = DepthQualityState::Invalid;
    bool                           control_distance_valid = false;
    float                          control_distance_m = std::numeric_limits<float>::infinity();
};

struct DepthControlDecision {
    DepthQualityState quality_state = DepthQualityState::Invalid;
    bool              control_distance_valid = false;
    float             control_distance_m = std::numeric_limits<float>::infinity();
    bool              allow_forward_motion = false;
};

struct ForwardSafetyDecision {
    bool  target_distance_valid = false;
    float target_forward_limit_m = 0.0f;
    bool  scan_available = false;
    bool  blocked = false;
    float nearest_obstacle_x_m = std::numeric_limits<float>::infinity();
};

struct RecoveryRotationState {
    RecoveryPhase phase = RecoveryPhase::Inactive;
    int          direction = 1;
    rclcpp::Time start_stamp;
    rclcpp::Time reacquired_stamp;

    [[nodiscard]] bool isActive() const {
        return phase != RecoveryPhase::Inactive;
    }

    [[nodiscard]] bool isSearching() const {
        return phase == RecoveryPhase::Searching;
    }

    [[nodiscard]] bool isReacquiredAligning() const {
        return phase == RecoveryPhase::ReacquiredAligning;
    }
};

struct VisualServoState {
    bool         active = false;
    bool         command_active = false;
    rclcpp::Time activation_stamp;
};

struct NavFallbackState {
    bool         active = false;
    std::string  reason;
    rclcpp::Time activated_stamp;
};

struct TrackMemoryState {
    int32_t                        track_id = -1;
    geometry_msgs::msg::PointStamped map_point;
    float                          robot_distance_m = std::numeric_limits<float>::infinity();
    float                          robot_bearing_rad = 0.0f;
    rclcpp::Time                   stamp;
};

struct LCVision::Impl {
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

    explicit Impl(LCVision &node);

    fs::path resolveModelDirectory(const DetectorBackend &backend) const;
    void updateTargetClassIds();
    void resetDetectorState();
    void storeColorFrame(const astra::ColorFrame &frame);
    void storeDepthFrame(const astra::DepthFrame &frame);
    void updateDetectionCache(
        std::vector<DetectionDepthResult> detections, int width, int height, int64_t frame_index,
        const rclcpp::Time &stamp);
    std::vector<DetectionDepthResult> snapshotDetections(cv::Size &source_size);
    bool copyLatestDepthFrame(
        std::vector<int16_t> &depth_data, int &width, int &height, int64_t &frame_index, rclcpp::Time &stamp);
    bool copyLatestRgbFrame(
        std::vector<uint8_t> &rgb_data, int &width, int &height, int64_t &frame_index, rclcpp::Time &stamp);
    [[nodiscard]] bool isHistogramDebugVideoEnabled() const;
    [[nodiscard]] bool isPeakMaskDebugVideoEnabled() const;
    void rotateRgbFrame(std::vector<uint8_t> &rgb_data, int &width, int &height) const;
    void rotateDepthFrame(std::vector<int16_t> &depth_data, int &width, int &height) const;
    cv::Rect scaleRectToSize(const cv::Rect &source_box, const cv::Size &source_size, const cv::Size &target_size)
        const;
    void populateCameraPoints(
        std::vector<DetectionDepthResult> &results, const cv::Size &rotated_depth_size, const cv::Size &raw_depth_size)
        const;
    void clearTrackedTarget();
    static cv::Point2f detectionCenter(const DetectionDepthResult &result);
    bool buildDetectionMapPoint(
        const DetectionDepthResult &result, const rclcpp::Time &stamp, geometry_msgs::msg::PointStamped &map_point) const;
    bool computeRobotRelativeObservation(
        const geometry_msgs::msg::PointStamped &map_point, float &distance_m, float &bearing_rad) const;
    static double angleDifferenceRad(double lhs, double rhs);
    int32_t allocateTrackId();
    void upsertTrackMemory(
        int32_t track_id, const geometry_msgs::msg::PointStamped &map_point, float robot_distance_m,
        float robot_bearing_rad, const rclcpp::Time &stamp);
    void pruneTrackMemories(const rclcpp::Time &stamp);
    void refreshTrackMemoryFromDetection(
        const DetectionDepthResult &result, int32_t track_id, const rclcpp::Time &stamp);
    void assignTrackIds(std::vector<DetectionDepthResult> &results, const rclcpp::Time &stamp);
    void handleNavigateToPoseStatus(const action_msgs::msg::GoalStatusArray::SharedPtr msg);
    DepthQualityState evaluateSelectedTargetDepthQuality(
        const DetectionDepthResult &result, const rclcpp::Time &stamp) const;
    DepthControlDecision buildDepthControlDecision(
        const DetectionDepthResult &result, DepthQualityState quality_state, const rclcpp::Time &stamp) const;
    void rememberTargetObservation(
        const DetectionDepthResult &result, int width, const rclcpp::Time &stamp,
        const std::optional<geometry_msgs::msg::PointStamped> &map_point = std::nullopt);
    void publishSelectedTargetStatus(const std::optional<DetectionDepthResult> &selected, const rclcpp::Time &stamp);
    void publishTrackingCommand(double linear_velocity, double angular_velocity);
    void handleNavigationCmdVel(const geometry_msgs::msg::Twist::SharedPtr msg);
    void activateVisualNavigationFallback(const std::string &reason, const rclcpp::Time &stamp);
    void clearVisualNavigationFallback(const std::string &reason);
    bool consumeVisualNavigationFallback();
    bool maybeHandleNavigationStall(const rclcpp::Time &stamp);
    void stopTrackingMotion(const std::string &reason, bool clear_visual_servo_mode = false);
    void stopRecoveryRotation(const std::string &reason);
    void startRecoveryRotation(int direction, const std::vector<DetectionDepthResult> &results, const rclcpp::Time &stamp);
    void markRecoveryTargetReacquired(const DetectionDepthResult &selected, int width, const rclcpp::Time &stamp);
    bool shouldKeepVisualServo(const rclcpp::Time &stamp) const;
    bool isTargetAlignedForForwardMotion(const DetectionDepthResult &selected, int width) const;
    bool maybeRunVisualServo(const DetectionDepthResult &selected, int width, const rclcpp::Time &stamp);
    bool tryHandleSelectedTargetWithVisualServo(
        const DetectionDepthResult &selected, int width, const rclcpp::Time &stamp);
    bool tryPopulateSelectedTargetMapPoint(
        const DetectionDepthResult &selected, const geometry_msgs::msg::PointStamped &camera_point,
        int width, const rclcpp::Time &stamp, geometry_msgs::msg::PointStamped &map_point);
    void maybeRecoverLostTarget(
        const std::vector<DetectionDepthResult> &results, const rclcpp::Time &stamp);
    int selectTrackedDetection(
        std::vector<DetectionDepthResult> &results, int width, int height, const rclcpp::Time &stamp);
    void storeCostmap(const nav2_msgs::msg::Costmap::SharedPtr msg);
    bool copyLatestCostmap(nav2_msgs::msg::Costmap &costmap) const;
    void storeLaserScan(const sensor_msgs::msg::LaserScan::SharedPtr msg);
    bool copyLatestLaserScan(sensor_msgs::msg::LaserScan &scan) const;
    bool transformPointToFrame(
        const geometry_msgs::msg::PointStamped &input, const std::string &target_frame,
        geometry_msgs::msg::PointStamped &output) const;
    ForwardSafetyDecision evaluateForwardSafety(
        const DetectionDepthResult &selected, const rclcpp::Time &stamp) const;
    bool worldToCostmapCell(
        const nav2_msgs::msg::Costmap &costmap, double world_x, double world_y, int &cell_x, int &cell_y) const;
    bool isFreeCost(uint8_t cost) const;
    bool isCellFree(const nav2_msgs::msg::Costmap &costmap, int cell_x, int cell_y) const;
    bool hasLocalClearance(
        const nav2_msgs::msg::Costmap &costmap, double world_x, double world_y, double clearance_radius_m) const;
    bool isRayFree(
        const nav2_msgs::msg::Costmap &costmap, const geometry_msgs::msg::Point &start,
        const geometry_msgs::msg::Point &end) const;
    std::optional<geometry_msgs::msg::PoseStamped> buildSafeGoalPose(
        const geometry_msgs::msg::PointStamped &target_map_point, const rclcpp::Time &stamp) const;
    void publishTrackedTargetDebug(
        const geometry_msgs::msg::PointStamped &camera_point, const geometry_msgs::msg::PointStamped &map_point,
        const std::optional<geometry_msgs::msg::PoseStamped> &goal_pose);
    void publishGoalDebugMarkers(const std::vector<DetectionDepthResult> &results, const rclcpp::Time &stamp);
    void maybeDispatchNavigationGoal(
        std::vector<DetectionDepthResult> &results, int width, int height, const rclcpp::Time &stamp);
    DetectionDepthResult estimateDepthForDetection(
        const Detection &detection, const cv::Size &rgb_size, const std::vector<int16_t> &depth_data,
        const cv::Size &depth_size) const;
    std::vector<DetectionDepthResult> estimateDepthResults(
        const std::vector<Detection> &detections, const cv::Size &rgb_size, const std::vector<int16_t> &depth_data,
        const cv::Size &depth_size) const;
    void publishDetectionDepths(
        const std::vector<DetectionDepthResult> &results, int width, int height, const rclcpp::Time &stamp);
    void publishDepthDebugMarkers(const std::vector<DetectionDepthResult> &results, const rclcpp::Time &stamp);
    std::vector<Detection> runDetection(
        const std::vector<uint8_t> &rgb_data, int width, int height, int input_size);
    void buildDepthVisualization(
        const std::vector<int16_t> &depth_data, int width, int height, std::vector<uint8_t> &output) const;
    int findBestDebugDetectionIndex(const std::vector<DetectionDepthResult> &results) const;
    cv::Mat renderDepthHistogramImage(const DetectionDepthResult &result) const;
    void updateHistogramDebugFrame(
        const std::vector<DetectionDepthResult> &results, int64_t frame_index, const rclcpp::Time &stamp);
    cv::Mat renderPeakMaskDebugImage(
        const std::vector<int16_t> &depth_data, const cv::Size &depth_size, const DetectionDepthResult &result) const;
    void updatePeakMaskDebugFrame(
        const std::vector<int16_t> &depth_data, const cv::Size &depth_size,
        const std::vector<DetectionDepthResult> &results, int64_t frame_index, const rclcpp::Time &stamp);
    void detectionLoop();
    [[nodiscard]] std::optional<foxglove_msgs::msg::CompressedVideo> makeRgbMessage();
    [[nodiscard]] std::optional<foxglove_msgs::msg::CompressedVideo> makeDepthMessage();
    [[nodiscard]] std::optional<foxglove_msgs::msg::CompressedVideo> makeDepthHistogramMessage();
    [[nodiscard]] std::optional<foxglove_msgs::msg::CompressedVideo> makeDepthPeakMaskMessage();
    void resetEncoders();
    void stopThreads();
    void stopStreams();
    void shutdown();

    LCVision &node;

    std::string sdk_root;
    StreamConfig rgb;
    DepthConfig  depth;
    ImageConfig  image;
    DetectorConfig detector;
    TrackingConfig tracking;
    GoalConfig     goal;

    astra::StreamSet                  stream_set;
    astra::StreamReader               reader;
    std::optional<astra::DepthStream> depth_stream;

    FrameBuffer<uint8_t> rgb_buffer;
    FrameBuffer<int16_t> depth_buffer;
    FrameBuffer<uint8_t> depth_histogram_buffer;
    FrameBuffer<uint8_t> depth_peak_mask_buffer;
    DetectionCache       detection_cache;
    CostmapCache         costmap_cache;
    LaserScanCache       laser_scan_cache;

    VideoEncoder rgb_encoder;
    VideoEncoder depth_encoder;
    VideoEncoder depth_histogram_encoder;
    VideoEncoder depth_peak_mask_encoder;

    std::vector<uint8_t> depth_visualization;
    std::unique_ptr<CaptureListener> capture_listener;

    rclcpp::Publisher<foxglove_msgs::msg::CompressedVideo>::SharedPtr rgb_video_pub;
    rclcpp::Publisher<foxglove_msgs::msg::CompressedVideo>::SharedPtr depth_video_pub;
    rclcpp::Publisher<foxglove_msgs::msg::CompressedVideo>::SharedPtr depth_histogram_video_pub;
    rclcpp::Publisher<foxglove_msgs::msg::CompressedVideo>::SharedPtr depth_peak_mask_video_pub;
    rclcpp::Publisher<lc_vision::msg::DetectionDepthArray>::SharedPtr detections_depth_pub;
    rclcpp::Publisher<lc_vision::msg::SelectedTargetStatus>::SharedPtr selected_target_status_pub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr depth_debug_marker_pub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr goal_debug_marker_pub;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr           recovery_cmd_vel_pub;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr selected_target_camera_point_pub;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr selected_target_map_point_pub;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr selected_goal_pose_pub;
    rclcpp::Subscription<nav2_msgs::msg::Costmap>::SharedPtr         costmap_sub;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr     laser_scan_sub;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr       navigation_cmd_vel_sub;
    rclcpp::Subscription<action_msgs::msg::GoalStatusArray>::SharedPtr navigate_to_pose_status_sub;

    std::thread capture_thread;
    std::thread publish_thread;
    std::thread detection_thread;

    std::atomic<bool> running{false};
    std::atomic<bool> detector_runtime_enabled{false};
    bool              detector_ready = false;
    bool              astra_initialized = false;
    DetectorFramework resolved_detector_framework = DetectorFramework::None;
    std::string       active_backend = "disabled";
    int               detected_gpu_count = 0;
    int               active_gpu_index = -1;
    std::unique_ptr<DetectorBackend> detector_backend;

    int64_t last_rgb_frame_index = -1;
    int64_t last_depth_frame_index = -1;
    int64_t last_detection_input_frame_index = -1;
    int64_t last_depth_histogram_frame_index = -1;
    int64_t last_depth_peak_mask_frame_index = -1;
    int     rgb_encoder_width = 0;
    int     rgb_encoder_height = 0;
    int     depth_encoder_width = 0;
    int     depth_encoder_height = 0;
    int     depth_histogram_encoder_width = 0;
    int     depth_histogram_encoder_height = 0;
    int     depth_peak_mask_encoder_width = 0;
    int     depth_peak_mask_encoder_height = 0;
    TrackedTargetState tracked_target;
    RememberedTargetState remembered_target;
    RecoveryRotationState recovery_rotation;
    VisualServoState visual_servo;
    NavFallbackState nav_fallback_state;
    std::vector<TrackMemoryState> track_memories;
    std::atomic<bool> navigation_goal_active{false};
    std::atomic<bool> external_navigation_active{false};
    std::atomic<bool> tracking_runtime_enabled{true};
    std::atomic<bool> goal_runtime_enabled{true};
    std::mutex        navigation_cmd_vel_mutex;
    rclcpp::Time      nav_goal_start_stamp;
    rclcpp::Time      last_navigation_cmd_vel_stamp;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_handle;
    std::mutex current_nav_goal_mutex;
    rclcpp_action::ClientGoalHandle<LCVision::NavigateToPose>::SharedPtr current_nav_goal_handle;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener;
};

} // namespace lc_vision
