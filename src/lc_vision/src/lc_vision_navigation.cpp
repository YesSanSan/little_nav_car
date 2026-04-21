#include "lc_vision_internal.hpp"

namespace lc_vision {

namespace {

std::string summarizeDetectionResult(const DetectionDepthResult &result) {
    std::ostringstream stream;
    stream.setf(std::ios::fixed);
    stream.precision(3);
    stream << "label=" << result.detection.label << " score=" << result.detection.score << " box=["
           << result.detection.box.x << "," << result.detection.box.y << "," << result.detection.box.width << ","
           << result.detection.box.height << "] depth_valid=" << (result.depth_valid ? "true" : "false")
           << " depth_mm=" << result.depth_mm << " roi_valid_px=" << result.roi_valid_pixel_count
           << " roi_invalid_px=" << result.roi_invalid_pixel_count << " roi_invalid_ratio=" << result.roi_invalid_ratio
           << " nearest_valid_depth_mm=" << result.nearest_valid_depth_mm << " peak_px=" << result.peak_pixel_count
           << " depth_quality=" << static_cast<int>(result.depth_quality_state)
           << " control_distance_valid=" << (result.control_distance_valid ? "true" : "false")
           << " control_distance_mm=" << result.control_distance_mm << " camera_valid="
           << (result.camera_point_valid ? "true" : "false") << " camera_point=["
           << result.camera_point.x << "," << result.camera_point.y << "," << result.camera_point.z << "] track_id="
           << result.track_id << " selected=" << (result.selected ? "true" : "false");
    return stream.str();
}

std::string summarizeDetections(const std::vector<DetectionDepthResult> &results) {
    std::ostringstream stream;
    stream << "count=" << results.size();
    for (size_t i = 0; i < results.size(); ++i) {
        stream << " {" << i << ": " << summarizeDetectionResult(results[i]) << "}";
    }
    return stream.str();
}

const char *toString(const RecoveryPhase phase) {
    switch (phase) {
    case RecoveryPhase::Inactive:
        return "inactive";
    case RecoveryPhase::Searching:
        return "searching";
    case RecoveryPhase::ReacquiredAligning:
        return "reacquired_aligning";
    }
    return "unknown";
}

const char *toString(const DepthQualityState state) {
    switch (state) {
    case DepthQualityState::Invalid:
        return "invalid";
    case DepthQualityState::Reliable:
        return "reliable";
    case DepthQualityState::NearFieldOccluded:
        return "near_field_occluded";
    case DepthQualityState::BackgroundSuspect:
        return "background_suspect";
    }
    return "unknown";
}

struct ContinuityMatchMetrics {
    double iou = 0.0;
    double center_distance_px = std::numeric_limits<double>::infinity();
    double depth_delta_mm = std::numeric_limits<double>::infinity();
    bool   depth_delta_valid = false;
    bool   iou_pass = false;
    bool   center_pass = false;
    bool   depth_pass = true;
    bool   accepted = false;
};

ContinuityMatchMetrics computeContinuityMatchMetrics(
    const DetectionDepthResult &reference, const DetectionDepthResult &candidate, const TrackingConfig &tracking) {
    ContinuityMatchMetrics metrics;
    const auto centerOf = [](const DetectionDepthResult &result) {
        return cv::Point2f(
            static_cast<float>(result.detection.box.x) + static_cast<float>(result.detection.box.width) * 0.5f,
            static_cast<float>(result.detection.box.y) + static_cast<float>(result.detection.box.height) * 0.5f);
    };
    metrics.iou = rectIou(reference.detection.box, candidate.detection.box);
    metrics.center_distance_px = cv::norm(centerOf(reference) - centerOf(candidate));
    metrics.iou_pass = metrics.iou >= static_cast<double>(tracking.min_iou_for_match);
    metrics.center_pass = metrics.center_distance_px <= static_cast<double>(tracking.max_center_distance_px);
    if (reference.depth_valid && candidate.depth_valid) {
        metrics.depth_delta_valid = true;
        metrics.depth_delta_mm = std::abs(static_cast<double>(reference.depth_mm) - static_cast<double>(candidate.depth_mm));
        metrics.depth_pass = metrics.depth_delta_mm <= static_cast<double>(tracking.max_depth_delta_mm);
    }
    metrics.accepted = metrics.depth_pass && (metrics.iou_pass || metrics.center_pass);
    return metrics;
}

std::string summarizeContinuityMatchMetrics(const ContinuityMatchMetrics &metrics) {
    std::ostringstream stream;
    stream.setf(std::ios::fixed);
    stream.precision(3);
    stream << "iou=" << metrics.iou << " center_distance_px=" << metrics.center_distance_px;
    if (metrics.depth_delta_valid) {
        stream << " depth_delta_mm=" << metrics.depth_delta_mm;
    } else {
        stream << " depth_delta_mm=n/a";
    }
    stream << " iou_pass=" << (metrics.iou_pass ? "true" : "false")
           << " center_pass=" << (metrics.center_pass ? "true" : "false")
           << " depth_pass=" << (metrics.depth_pass ? "true" : "false")
           << " accepted=" << (metrics.accepted ? "true" : "false");
    return stream.str();
}

} // namespace

void LCVision::Impl::clearTrackedTarget() {
    const int32_t next_track_id = tracked_target.next_track_id;
    tracked_target.active = false;
    tracked_target.current_track_id = -1;
    tracked_target.stable_frames = 0;
    tracked_target.lost_frames = 0;
    tracked_target.goal_dispatched = false;
    tracked_target.last_result = DetectionDepthResult();
    tracked_target.last_map_point = geometry_msgs::msg::PointStamped();
    tracked_target.last_map_point_valid = false;
    tracked_target.last_stamp = rclcpp::Time{};
    tracked_target.next_track_id = next_track_id;
}

cv::Point2f LCVision::Impl::detectionCenter(const DetectionDepthResult &result) {
    return cv::Point2f(
        static_cast<float>(result.detection.box.x) + static_cast<float>(result.detection.box.width) * 0.5f,
        static_cast<float>(result.detection.box.y) + static_cast<float>(result.detection.box.height) * 0.5f);
}

bool LCVision::Impl::buildDetectionMapPoint(
    const DetectionDepthResult &result, const rclcpp::Time &stamp, geometry_msgs::msg::PointStamped &map_point) const {
    if (!result.camera_point_valid) {
        return false;
    }

    geometry_msgs::msg::PointStamped camera_point;
    camera_point.header.stamp = toBuiltinTime(stamp);
    camera_point.header.frame_id = depth.frame_id;
    camera_point.point = result.camera_point;
    return transformPointToFrame(camera_point, goal.global_frame_id, map_point);
}

bool LCVision::Impl::computeRobotRelativeObservation(
    const geometry_msgs::msg::PointStamped &map_point, float &distance_m, float &bearing_rad) const {
    geometry_msgs::msg::PointStamped robot_point;
    if (!transformPointToFrame(map_point, goal.robot_frame_id, robot_point)) {
        return false;
    }

    distance_m = static_cast<float>(std::hypot(robot_point.point.x, robot_point.point.y));
    bearing_rad = static_cast<float>(std::atan2(robot_point.point.y, robot_point.point.x));
    return true;
}

double LCVision::Impl::angleDifferenceRad(const double lhs, const double rhs) {
    double delta = std::fmod(lhs - rhs, 2.0 * M_PI);
    if (delta > M_PI) {
        delta -= 2.0 * M_PI;
    } else if (delta < -M_PI) {
        delta += 2.0 * M_PI;
    }
    return delta;
}

int32_t LCVision::Impl::allocateTrackId() {
    return tracked_target.next_track_id++;
}

void LCVision::Impl::upsertTrackMemory(
    const int32_t track_id, const geometry_msgs::msg::PointStamped &map_point, const float robot_distance_m,
    const float robot_bearing_rad, const rclcpp::Time &stamp) {
    for (auto &memory : track_memories) {
        if (memory.track_id != track_id) {
            continue;
        }

        memory.map_point = map_point;
        memory.robot_distance_m = robot_distance_m;
        memory.robot_bearing_rad = robot_bearing_rad;
        memory.stamp = stamp;
        return;
    }

    TrackMemoryState memory;
    memory.track_id = track_id;
    memory.map_point = map_point;
    memory.robot_distance_m = robot_distance_m;
    memory.robot_bearing_rad = robot_bearing_rad;
    memory.stamp = stamp;
    track_memories.push_back(std::move(memory));
}

void LCVision::Impl::pruneTrackMemories(const rclcpp::Time &stamp) {
    const double timeout_sec = std::max(0.0f, tracking.debug.track_memory_timeout_sec);
    if (timeout_sec > 0.0) {
        track_memories.erase(
            std::remove_if(
                track_memories.begin(), track_memories.end(),
                [&](const TrackMemoryState &memory) { return (stamp - memory.stamp).seconds() > timeout_sec; }),
            track_memories.end());
    }

    const int max_memories = std::max(1, tracking.debug.max_track_memories);
    if (static_cast<int>(track_memories.size()) <= max_memories) {
        return;
    }

    std::sort(track_memories.begin(), track_memories.end(), [](const TrackMemoryState &lhs, const TrackMemoryState &rhs) {
        return lhs.stamp > rhs.stamp;
    });
    track_memories.resize(static_cast<size_t>(max_memories));
}

void LCVision::Impl::refreshTrackMemoryFromDetection(
    const DetectionDepthResult &result, const int32_t track_id, const rclcpp::Time &stamp) {
    geometry_msgs::msg::PointStamped map_point;
    if (!buildDetectionMapPoint(result, stamp, map_point)) {
        return;
    }

    float robot_distance_m = std::numeric_limits<float>::infinity();
    float robot_bearing_rad = 0.0f;
    if (!computeRobotRelativeObservation(map_point, robot_distance_m, robot_bearing_rad)) {
        return;
    }

    upsertTrackMemory(track_id, map_point, robot_distance_m, robot_bearing_rad, stamp);
    pruneTrackMemories(stamp);
}

void LCVision::Impl::assignTrackIds(std::vector<DetectionDepthResult> &results, const rclcpp::Time &stamp) {
    struct CandidateObservation {
        int index = -1;
        geometry_msgs::msg::PointStamped map_point;
        float robot_distance_m = std::numeric_limits<float>::infinity();
        float robot_bearing_rad = 0.0f;
        bool valid = false;
    };

    std::vector<CandidateObservation> observations;
    observations.reserve(results.size());
    for (size_t i = 0; i < results.size(); ++i) {
        results[i].track_id = -1;
        results[i].selected = false;

        CandidateObservation observation;
        observation.index = static_cast<int>(i);
        observation.valid =
            buildDetectionMapPoint(results[i], stamp, observation.map_point) &&
            computeRobotRelativeObservation(
                observation.map_point, observation.robot_distance_m, observation.robot_bearing_rad);
        observations.push_back(std::move(observation));
    }

    std::vector<size_t> history_order(track_memories.size());
    std::iota(history_order.begin(), history_order.end(), 0);
    std::sort(history_order.begin(), history_order.end(), [this](const size_t lhs, const size_t rhs) {
        return track_memories[lhs].stamp > track_memories[rhs].stamp;
    });

    std::vector<bool> used_history(track_memories.size(), false);
    for (auto &observation : observations) {
        if (!observation.valid) {
            continue;
        }

        for (const size_t history_index : history_order) {
            if (used_history[history_index]) {
                continue;
            }

            const auto &memory = track_memories[history_index];
            const double distance_delta =
                std::abs(static_cast<double>(observation.robot_distance_m) - static_cast<double>(memory.robot_distance_m));
            if (distance_delta > static_cast<double>(tracking.map_match.distance_tolerance_m)) {
                continue;
            }

            const double angle_delta = std::abs(angleDifferenceRad(
                static_cast<double>(observation.robot_bearing_rad), static_cast<double>(memory.robot_bearing_rad)));
            if (angle_delta > static_cast<double>(tracking.map_match.angle_tolerance_rad)) {
                continue;
            }

            results[static_cast<size_t>(observation.index)].track_id = memory.track_id;
            used_history[history_index] = true;
            break;
        }
    }

    for (const auto &observation : observations) {
        if (!observation.valid) {
            continue;
        }

        const auto &result = results[static_cast<size_t>(observation.index)];
        if (result.track_id < 0) {
            continue;
        }

        upsertTrackMemory(
            result.track_id, observation.map_point, observation.robot_distance_m, observation.robot_bearing_rad, stamp);
    }
}

void LCVision::Impl::handleNavigateToPoseStatus(const action_msgs::msg::GoalStatusArray::SharedPtr msg) {
    rclcpp_action::GoalUUID current_goal_id{};
    bool                    have_current_goal = false;
    {
        std::lock_guard<std::mutex> lock(current_nav_goal_mutex);
        if (current_nav_goal_handle != nullptr) {
            current_goal_id = current_nav_goal_handle->get_goal_id();
            have_current_goal = true;
        }
    }

    bool saw_external_active_goal = false;
    for (const auto &status : msg->status_list) {
        const bool active =
            status.status == action_msgs::msg::GoalStatus::STATUS_ACCEPTED ||
            status.status == action_msgs::msg::GoalStatus::STATUS_EXECUTING ||
            status.status == action_msgs::msg::GoalStatus::STATUS_CANCELING;
        if (!active) {
            continue;
        }

        if (!have_current_goal || status.goal_info.goal_id.uuid != current_goal_id) {
            saw_external_active_goal = true;
            break;
        }
    }

    external_navigation_active.store(saw_external_active_goal, std::memory_order_relaxed);
}

DepthQualityState LCVision::Impl::evaluateSelectedTargetDepthQuality(
    const DetectionDepthResult &result, const rclcpp::Time &stamp) const {
    if (!result.depth_valid) {
        return DepthQualityState::Invalid;
    }

    if (!depth.estimation.near_field_enable || !remembered_target.reliable_depth_available) {
        return DepthQualityState::Reliable;
    }

    const double history_age_sec = (stamp - remembered_target.reliable_depth_stamp).seconds();
    if (history_age_sec > static_cast<double>(std::max(0.0f, depth.estimation.near_field_history_timeout_sec))) {
        return DepthQualityState::Reliable;
    }

    const float history_depth_mm = remembered_target.reliable_depth_m * 1000.0f;
    if (!std::isfinite(history_depth_mm)) {
        return DepthQualityState::Reliable;
    }

    if (history_depth_mm < static_cast<float>(depth.estimation.near_field_min_history_depth_mm) ||
        history_depth_mm > static_cast<float>(depth.estimation.near_field_max_expected_depth_mm) ||
        remembered_target.reliable_depth_streak < std::max(1, depth.estimation.near_field_min_reliable_frames)) {
        return DepthQualityState::Reliable;
    }

    const bool large_depth_jump =
        result.depth_mm >= history_depth_mm + static_cast<float>(depth.estimation.near_field_depth_jump_mm);
    if (!large_depth_jump) {
        return DepthQualityState::Reliable;
    }

    const bool high_invalid_ratio =
        result.roi_invalid_ratio >= std::clamp(depth.estimation.near_field_invalid_ratio_threshold, 0.0f, 1.0f);
    const bool nearest_valid_still_far =
        result.nearest_valid_depth_mm <= 0.0f ||
        result.nearest_valid_depth_mm >=
            history_depth_mm + static_cast<float>(depth.estimation.near_field_nearest_valid_margin_mm);
    if (!nearest_valid_still_far) {
        return DepthQualityState::Reliable;
    }

    if (high_invalid_ratio) {
        return DepthQualityState::NearFieldOccluded;
    }

    return DepthQualityState::BackgroundSuspect;
}

DepthControlDecision LCVision::Impl::buildDepthControlDecision(
    const DetectionDepthResult &result, const DepthQualityState quality_state, const rclcpp::Time &stamp) const {
    DepthControlDecision decision;
    decision.quality_state = quality_state;

    switch (quality_state) {
    case DepthQualityState::Reliable:
        if (result.depth_valid) {
            decision.control_distance_valid = true;
            decision.control_distance_m = result.depth_mm / 1000.0f;
            decision.allow_forward_motion = true;
        }
        return decision;
    case DepthQualityState::NearFieldOccluded:
    case DepthQualityState::BackgroundSuspect: {
        const bool have_recent_reliable_depth =
            remembered_target.reliable_depth_available &&
            (stamp - remembered_target.reliable_depth_stamp).seconds() <=
                static_cast<double>(std::max(0.0f, depth.estimation.near_field_history_timeout_sec));
        if (!have_recent_reliable_depth) {
            return decision;
        }
        decision.control_distance_valid = true;
        decision.control_distance_m =
            std::min(remembered_target.reliable_depth_m, std::max(0.0f, goal.standoff_distance_m));
        decision.allow_forward_motion = false;
        return decision;
    }
    case DepthQualityState::Invalid:
        return decision;
    }
    return decision;
}

void LCVision::Impl::rememberTargetObservation(
    const DetectionDepthResult &result, const int width, const rclcpp::Time &stamp,
    const std::optional<geometry_msgs::msg::PointStamped> &map_point) {
    remembered_target.available = true;
    remembered_target.stamp = stamp;
    remembered_target.image_offset_px = detectionCenter(result).x - static_cast<float>(width) * 0.5f;
    remembered_target.camera_lateral_m =
        result.camera_point_valid ? static_cast<float>(result.camera_point.x) : 0.0f;
    remembered_target.last_depth_quality_state = result.depth_quality_state;
    remembered_target.control_distance_valid = result.control_distance_valid;
    remembered_target.control_distance_m =
        result.control_distance_valid ? result.control_distance_mm / 1000.0f : std::numeric_limits<float>::infinity();
    if (result.depth_quality_state == DepthQualityState::Reliable && result.depth_valid) {
        remembered_target.reliable_depth_available = true;
        remembered_target.reliable_depth_stamp = stamp;
        remembered_target.reliable_depth_m = result.depth_mm / 1000.0f;
        ++remembered_target.reliable_depth_streak;
    } else {
        remembered_target.reliable_depth_streak = 0;
    }
    remembered_target.depth_m = result.control_distance_valid
                                    ? result.control_distance_mm / 1000.0f
                                    : (remembered_target.reliable_depth_available
                                           ? remembered_target.reliable_depth_m
                                           : std::numeric_limits<float>::infinity());
    remembered_target.map_point_valid = map_point.has_value();
    if (map_point.has_value()) {
        remembered_target.map_point = *map_point;
    }
}

void LCVision::Impl::publishSelectedTargetStatus(
    const std::optional<DetectionDepthResult> &selected, const rclcpp::Time &stamp) {
    if (selected_target_status_pub == nullptr) {
        return;
    }

    lc_vision::msg::SelectedTargetStatus msg;
    msg.header.stamp = toBuiltinTime(stamp);
    msg.header.frame_id = depth.frame_id;
    msg.track_id = -1;
    msg.depth_quality_state = "no_target";
    if (selected.has_value()) {
        const auto &result = *selected;
        msg.label = result.detection.label;
        msg.track_id = result.track_id;
        msg.selected = result.selected;
        msg.depth_quality_state = toString(result.depth_quality_state);
        msg.camera_depth_valid = result.depth_valid;
        msg.camera_depth_mm = result.depth_mm;
        msg.control_distance_valid = result.control_distance_valid;
        msg.control_distance_mm = result.control_distance_mm;
        msg.roi_invalid_ratio = result.roi_invalid_ratio;
        msg.roi_valid_pixel_count = result.roi_valid_pixel_count;
        msg.roi_invalid_pixel_count = result.roi_invalid_pixel_count;
        msg.nearest_valid_depth_mm = result.nearest_valid_depth_mm;
    }
    selected_target_status_pub->publish(msg);
}

void LCVision::Impl::publishTrackingCommand(const double linear_velocity, const double angular_velocity) {
    if (recovery_cmd_vel_pub == nullptr) {
        return;
    }

    geometry_msgs::msg::Twist cmd;
    cmd.linear.x = linear_velocity;
    cmd.angular.z = angular_velocity;
    recovery_cmd_vel_pub->publish(cmd);
}

void LCVision::Impl::stopTrackingMotion(const std::string &reason, const bool clear_visual_servo_mode) {
    const bool had_motion = recovery_rotation.isActive() || visual_servo.command_active;
    recovery_rotation.phase = RecoveryPhase::Inactive;
    recovery_rotation.reacquired_stamp = rclcpp::Time{};
    visual_servo.command_active = false;
    if (clear_visual_servo_mode) {
        visual_servo.active = false;
    }

    if (!had_motion) {
        return;
    }

    publishTrackingCommand(0.0, 0.0);
    RCLCPP_INFO(node.get_logger(), "Stopped visual tracking motion: %s", reason.c_str());
}

void LCVision::Impl::stopRecoveryRotation(const std::string &reason) {
    if (!recovery_rotation.isActive()) {
        return;
    }

    recovery_rotation.phase = RecoveryPhase::Inactive;
    recovery_rotation.reacquired_stamp = rclcpp::Time{};
    publishTrackingCommand(0.0, 0.0);
    RCLCPP_INFO(node.get_logger(), "Stopped visual recovery rotation: %s", reason.c_str());
}

void LCVision::Impl::startRecoveryRotation(
    const int direction, const std::vector<DetectionDepthResult> &results, const rclcpp::Time &stamp) {
    node.cancelCurrentNavigationGoal("vision target lost, starting visual recovery rotation");
    tracked_target.goal_dispatched = false;
    recovery_rotation.phase = RecoveryPhase::Searching;
    recovery_rotation.direction = direction;
    recovery_rotation.start_stamp = stamp;
    recovery_rotation.reacquired_stamp = rclcpp::Time{};
    RCLCPP_INFO(
        node.get_logger(),
        "Started visual recovery rotation: direction=%d phase=%s remembered_offset_px=%.3f remembered_lateral_m=%.3f "
        "remembered_depth_m=%.3f tracked_id=%d lost_frames=%d detections=%s",
        direction, toString(recovery_rotation.phase), static_cast<double>(remembered_target.image_offset_px),
        static_cast<double>(remembered_target.camera_lateral_m), static_cast<double>(remembered_target.depth_m),
        tracked_target.current_track_id, tracked_target.lost_frames, summarizeDetections(results).c_str());
}

void LCVision::Impl::markRecoveryTargetReacquired(
    const DetectionDepthResult &selected, const int width, const rclcpp::Time &stamp) {
    if (!recovery_rotation.isSearching()) {
        return;
    }

    recovery_rotation.phase = RecoveryPhase::ReacquiredAligning;
    recovery_rotation.reacquired_stamp = stamp;
    RCLCPP_INFO(
        node.get_logger(),
        "Recovery target reacquired: direction=%d phase=%s offset_px=%.3f depth_m=%.3f waiting_for_alignment_before_stop=true",
        recovery_rotation.direction, toString(recovery_rotation.phase),
        static_cast<double>(detectionCenter(selected).x - static_cast<float>(width) * 0.5f),
        static_cast<double>(selected.depth_mm / 1000.0f));
}

bool LCVision::Impl::shouldKeepVisualServo(const rclcpp::Time &stamp) const {
    if (!tracking.visual_servo.enable || !visual_servo.active || !remembered_target.available) {
        return false;
    }

    const double memory_age = (stamp - remembered_target.stamp).seconds();
    if (memory_age > static_cast<double>(tracking.recovery.memory_timeout_sec)) {
        return false;
    }

    return remembered_target.depth_m <= tracking.visual_servo.lost_target_distance_m;
}

bool LCVision::Impl::isTargetAlignedForForwardMotion(const DetectionDepthResult &selected, const int width) const {
    const float image_center_x = static_cast<float>(width) * 0.5f;
    const float box_center_x = detectionCenter(selected).x;
    const float half_band = std::max(
        1.0f,
        std::min(
            static_cast<float>(tracking.visual_servo.align_deadband_px),
            static_cast<float>(selected.detection.box.width) * tracking.visual_servo.align_band_ratio));
    const float band_left = box_center_x - half_band;
    const float band_right = box_center_x + half_band;
    return image_center_x >= band_left && image_center_x <= band_right;
}

bool LCVision::Impl::maybeRunVisualServo(
    const DetectionDepthResult &selected, const int width, const rclcpp::Time &stamp) {
    if (!tracking.visual_servo.enable || !selected.control_distance_valid) {
        if (visual_servo.active) {
            visual_servo.active = false;
        }
        return false;
    }

    const float target_distance_m = selected.control_distance_mm / 1000.0f;
    if (!visual_servo.active && target_distance_m <= tracking.visual_servo.engage_distance_m) {
        visual_servo.active = true;
        visual_servo.activation_stamp = stamp;
        RCLCPP_INFO(
            node.get_logger(), "Entering pure visual tracking mode at %.2fm (%s)", static_cast<double>(target_distance_m),
            toString(selected.depth_quality_state));
    }

    if (!visual_servo.active) {
        return false;
    }

    if (target_distance_m > tracking.visual_servo.lost_target_distance_m) {
        stopTrackingMotion("visual target moved beyond visual-servo range", true);
        return false;
    }

    if (navigation_goal_active.load(std::memory_order_relaxed)) {
        node.cancelCurrentNavigationGoal("switching to pure visual tracking");
    }

    tracked_target.goal_dispatched = false;

    const float image_center_x = static_cast<float>(width) * 0.5f;
    const float offset_px = detectionCenter(selected).x - image_center_x;
    const float normalized_error = offset_px / std::max(1.0f, image_center_x);
    const float abs_offset_px = std::abs(offset_px);
    const bool aligned_for_forward = isTargetAlignedForForwardMotion(selected, width);
    const bool allow_recovery_stop = recovery_rotation.isReacquiredAligning() &&
                                     (stamp - recovery_rotation.reacquired_stamp).seconds() >=
                                         static_cast<double>(tracking.recovery.lost_delay_sec);

    double angular_velocity = 0.0;
    if (!aligned_for_forward && abs_offset_px > tracking.visual_servo.forward_deadband_px) {
        angular_velocity = std::clamp(
            -static_cast<double>(normalized_error) * static_cast<double>(tracking.visual_servo.turn_gain),
            -static_cast<double>(tracking.visual_servo.max_turn_speed_rad_s),
            static_cast<double>(tracking.visual_servo.max_turn_speed_rad_s));
        if (std::abs(angular_velocity) > 1.0e-4) {
            const double min_turn_speed = static_cast<double>(tracking.visual_servo.min_turn_speed_rad_s);
            angular_velocity = std::copysign(std::max(std::abs(angular_velocity), min_turn_speed), angular_velocity);
        }
    }

    double linear_velocity = 0.0;
    const double distance_error_m =
        static_cast<double>(target_distance_m) - static_cast<double>(goal.standoff_distance_m);
    const auto forward_safety = evaluateForwardSafety(selected, stamp);
    const bool forward_blocked = forward_safety.blocked;
    if (selected.depth_quality_state == DepthQualityState::Reliable && distance_error_m > 0.0 && aligned_for_forward &&
        !forward_blocked) {
        linear_velocity = std::clamp(
            distance_error_m * static_cast<double>(tracking.visual_servo.linear_gain),
            static_cast<double>(tracking.visual_servo.min_linear_speed_m_s),
            static_cast<double>(tracking.visual_servo.max_linear_speed_m_s));
    }

    if (aligned_for_forward && distance_error_m <= 0.05) {
        linear_velocity = 0.0;
        angular_velocity = 0.0;
    }

    if (aligned_for_forward && allow_recovery_stop) {
        stopRecoveryRotation("target alignment band held after reacquisition");
    }

    publishTrackingCommand(linear_velocity, angular_velocity);
    visual_servo.command_active = true;
    if (tracking.debug.enable_verbose_logs) {
        RCLCPP_INFO(
            node.get_logger(),
            "Visual servo command: offset_px=%.3f normalized_error=%.3f aligned_for_forward=%s linear=%.3f angular=%.3f "
            "depth_quality=%s control_distance_m=%.3f forward_blocked=%s target_forward_limit_m=%.3f scan_available=%s "
            "nearest_obstacle_x_m=%.3f recovery_phase=%s recovery_direction=%d allow_recovery_stop=%s",
            static_cast<double>(offset_px), static_cast<double>(normalized_error),
            aligned_for_forward ? "true" : "false", linear_velocity, angular_velocity,
            toString(selected.depth_quality_state), static_cast<double>(target_distance_m),
            forward_blocked ? "true" : "false", static_cast<double>(forward_safety.target_forward_limit_m),
            forward_safety.scan_available ? "true" : "false", static_cast<double>(forward_safety.nearest_obstacle_x_m),
            toString(recovery_rotation.phase), recovery_rotation.direction, allow_recovery_stop ? "true" : "false");
    }
    return true;
}

bool LCVision::Impl::tryHandleSelectedTargetWithVisualServo(
    const DetectionDepthResult &selected, const int width, const rclcpp::Time &stamp) {
    markRecoveryTargetReacquired(selected, width, stamp);
    if (!maybeRunVisualServo(selected, width, stamp)) {
        return false;
    }

    if (tracking.debug.enable_verbose_logs) {
        RCLCPP_INFO(
            node.get_logger(), "Visual servo handled selected target. track_id=%d detection=%s",
            tracked_target.current_track_id, summarizeDetectionResult(selected).c_str());
    }
    return true;
}

bool LCVision::Impl::tryPopulateSelectedTargetMapPoint(
    const DetectionDepthResult &selected, const geometry_msgs::msg::PointStamped &camera_point, const int width,
    const rclcpp::Time &stamp, geometry_msgs::msg::PointStamped &map_point) {
    if (!transformPointToFrame(camera_point, goal.global_frame_id, map_point)) {
        if (tracking.debug.enable_verbose_logs) {
            RCLCPP_INFO(
                node.get_logger(),
                "Selected detection could not transform to global frame. selected=%s",
                summarizeDetectionResult(selected).c_str());
        }
        return false;
    }

    rememberTargetObservation(selected, width, stamp, map_point);
    tracked_target.last_map_point = map_point;
    tracked_target.last_map_point_valid = true;
    return true;
}

void LCVision::Impl::maybeRecoverLostTarget(
    const std::vector<DetectionDepthResult> &results, const rclcpp::Time &stamp) {
    constexpr double kDetectorBlankGraceSec = 0.6;

    if (!tracking.recovery.enable) {
        return;
    }

    if (external_navigation_active.load(std::memory_order_relaxed)) {
        if (recovery_rotation.isActive()) {
            stopTrackingMotion("external navigation is active", false);
        }
        return;
    }

    if (visual_servo.active && !shouldKeepVisualServo(stamp)) {
        stopTrackingMotion("visual servo memory expired or target moved too far", true);
    }

    const bool keep_visual_servo = shouldKeepVisualServo(stamp);
    if (!keep_visual_servo && navigation_goal_active.load(std::memory_order_relaxed)) {
        if (recovery_rotation.isActive()) {
            stopRecoveryRotation("navigation is active");
        }
        return;
    }

    if (!remembered_target.available) {
        if (recovery_rotation.isActive()) {
            stopRecoveryRotation("no remembered target");
        }
        if (tracking.debug.enable_verbose_logs) {
            RCLCPP_INFO(
                node.get_logger(), "Recovery skipped: no remembered target. detections=%s",
                summarizeDetections(results).c_str());
        }
        return;
    }

    const double memory_age =
        (stamp - remembered_target.stamp).seconds();
    if (memory_age > static_cast<double>(tracking.recovery.memory_timeout_sec)) {
        if (recovery_rotation.isActive()) {
            stopRecoveryRotation("remembered target expired");
        }
        remembered_target.available = false;
        if (tracking.debug.enable_verbose_logs) {
            RCLCPP_INFO(
                node.get_logger(),
                "Recovery skipped: remembered target expired. memory_age=%.3f timeout=%.3f detections=%s",
                memory_age, static_cast<double>(tracking.recovery.memory_timeout_sec),
                summarizeDetections(results).c_str());
        }
        return;
    }

    // Near-range visual tracking frequently suffers short detector blanking, especially when the target
    // is large and close to image borders. When the detector returns no boxes at all, keep the robot
    // out of recovery scan for a short grace period and stop issuing motion until detections return.
    if (results.empty() && keep_visual_servo && memory_age < kDetectorBlankGraceSec) {
        if (recovery_rotation.isActive()) {
            stopRecoveryRotation("detector blank grace period while close-range visual servo is active");
        }
        if (visual_servo.command_active) {
            publishTrackingCommand(0.0, 0.0);
            visual_servo.command_active = false;
        }
        if (tracking.debug.enable_verbose_logs) {
            RCLCPP_INFO(
                node.get_logger(),
                "Recovery suppressed during detector blank grace period: memory_age=%.3f blank_grace=%.3f "
                "remembered_offset_px=%.3f remembered_lateral_m=%.3f remembered_depth_m=%.3f",
                memory_age, kDetectorBlankGraceSec, static_cast<double>(remembered_target.image_offset_px),
                static_cast<double>(remembered_target.camera_lateral_m), static_cast<double>(remembered_target.depth_m));
        }
        return;
    }

    if (!recovery_rotation.isActive() &&
        memory_age < static_cast<double>(tracking.recovery.lost_delay_sec)) {
        if (tracking.debug.enable_verbose_logs) {
            RCLCPP_INFO(
                node.get_logger(),
                "Recovery delayed: memory_age=%.3f lost_delay=%.3f remembered_offset_px=%.3f remembered_lateral_m=%.3f "
                "remembered_depth_m=%.3f detections=%s",
                memory_age, static_cast<double>(tracking.recovery.lost_delay_sec),
                static_cast<double>(remembered_target.image_offset_px),
                static_cast<double>(remembered_target.camera_lateral_m),
                static_cast<double>(remembered_target.depth_m), summarizeDetections(results).c_str());
        }
        return;
    }

    if (!recovery_rotation.isActive()) {
        int direction = 1;
        if (std::abs(remembered_target.image_offset_px) > 1.0f) {
            direction = remembered_target.image_offset_px > 0.0f ? -1 : 1;
        } else if (std::abs(remembered_target.camera_lateral_m) > 1.0e-3f) {
            direction = remembered_target.camera_lateral_m > 0.0f ? -1 : 1;
        }
        startRecoveryRotation(direction, results, stamp);
    }

    bool has_visible_candidate = false;
    for (const auto &result : results) {
        if (result.depth_valid && result.camera_point_valid) {
            has_visible_candidate = true;
            break;
        }
    }

    const double recovery_age = (stamp - recovery_rotation.start_stamp).seconds();
    if (recovery_age > static_cast<double>(tracking.recovery.search_timeout_sec)) {
        stopTrackingMotion("recovery timeout", true);
        clearTrackedTarget();
        remembered_target.available = false;
        RCLCPP_INFO(
            node.get_logger(),
            "Recovery timeout: recovery_age=%.3f timeout=%.3f detections=%s",
            recovery_age, static_cast<double>(tracking.recovery.search_timeout_sec), summarizeDetections(results).c_str());
        return;
    }

    double turn_speed = static_cast<double>(tracking.recovery.turn_speed_rad_s);
    if (has_visible_candidate) {
        turn_speed = std::min(
            turn_speed, static_cast<double>(tracking.recovery.reacquire_turn_speed_rad_s));
    }
    if (keep_visual_servo) {
        turn_speed = std::min(turn_speed, static_cast<double>(tracking.visual_servo.max_turn_speed_rad_s));
    }

    if (tracking.debug.enable_verbose_logs) {
        RCLCPP_INFO(
            node.get_logger(),
            "Recovery command: phase=%s direction=%d turn_speed=%.3f keep_visual_servo=%s has_visible_candidate=%s recovery_age=%.3f "
            "memory_age=%.3f tracked_id=%d lost_frames=%d detections=%s",
            toString(recovery_rotation.phase), recovery_rotation.direction, turn_speed,
            keep_visual_servo ? "true" : "false", has_visible_candidate ? "true" : "false", recovery_age, memory_age,
            tracked_target.current_track_id, tracked_target.lost_frames, summarizeDetections(results).c_str());
    }
    publishTrackingCommand(0.0, static_cast<double>(recovery_rotation.direction) * turn_speed);
}

int LCVision::Impl::selectTrackedDetection(
    std::vector<DetectionDepthResult> &results, const int width, const int height, const rclcpp::Time &stamp) {
    if (!tracking_runtime_enabled.load(std::memory_order_relaxed)) {
        if (tracked_target.active) {
            clearTrackedTarget();
        }
        return -1;
    }

    // An active target with an invalid track id cannot be matched reliably and causes visible detections
    // to be treated as "lost". Drop that broken state and reacquire from current detections instead.
    if (tracked_target.active && tracked_target.current_track_id < 0) {
        RCLCPP_WARN(
            node.get_logger(),
            "Tracked target entered invalid state with track_id=%d; clearing target and reacquiring from detections",
            tracked_target.current_track_id);
        clearTrackedTarget();
    }

    std::vector<int> candidates;
    candidates.reserve(results.size());
    for (size_t i = 0; i < results.size(); ++i) {
        if (results[i].depth_valid && results[i].camera_point_valid) {
            candidates.push_back(static_cast<int>(i));
        }
    }

    if (tracking.debug.enable_verbose_logs) {
        RCLCPP_INFO(
            node.get_logger(),
            "Tracking input: frame_size=%dx%d tracked_active=%s tracked_id=%d stable_frames=%d lost_frames=%d detections=%s",
            width, height, tracked_target.active ? "true" : "false", tracked_target.current_track_id,
            tracked_target.stable_frames, tracked_target.lost_frames, summarizeDetections(results).c_str());
    }

    if (candidates.empty()) {
        if (tracked_target.active) {
            ++tracked_target.lost_frames;
            if (tracked_target.lost_frames > tracking.max_lost_frames) {
                clearTrackedTarget();
            }
        }
        if (tracking.debug.enable_verbose_logs) {
            RCLCPP_INFO(
                node.get_logger(),
                "Tracking found no valid depth candidates. tracked_active=%s tracked_id=%d lost_frames=%d max_lost_frames=%d",
                tracked_target.active ? "true" : "false", tracked_target.current_track_id, tracked_target.lost_frames,
                tracking.max_lost_frames);
        }
        return -1;
    }

    assignTrackIds(results, stamp);

    if (tracked_target.active) {
        const auto commitTrackedMatch =
            [&](const int best_index, const char *source, const std::string &details) {
                tracked_target.lost_frames = 0;
                ++tracked_target.stable_frames;
                refreshTrackMemoryFromDetection(
                    results[static_cast<size_t>(best_index)], tracked_target.current_track_id, stamp);
                tracked_target.last_result = results[static_cast<size_t>(best_index)];
                tracked_target.last_stamp = stamp;
                results[static_cast<size_t>(best_index)].selected = true;
                results[static_cast<size_t>(best_index)].track_id = tracked_target.current_track_id;
                tracked_target.last_result.selected = true;
                tracked_target.last_result.track_id = tracked_target.current_track_id;
                if (tracking.debug.enable_verbose_logs) {
                    RCLCPP_INFO(
                        node.get_logger(),
                        "Tracking matched existing target: source=%s index=%d track_id=%d stable_frames=%d %s detection=%s",
                        source, best_index, tracked_target.current_track_id, tracked_target.stable_frames, details.c_str(),
                        summarizeDetectionResult(results[static_cast<size_t>(best_index)]).c_str());
                }
                return best_index;
            };

        int best_index = -1;
        for (const int candidate_index : candidates) {
            if (results[static_cast<size_t>(candidate_index)].track_id == tracked_target.current_track_id) {
                best_index = candidate_index;
                break;
            }
        }

        if (best_index >= 0) {
            return commitTrackedMatch(best_index, "track_id", "match=exact_track_id");
        }

        int                    continuity_best_index = -1;
        ContinuityMatchMetrics continuity_best_metrics;
        bool                   have_continuity_candidate = false;
        for (const int candidate_index : candidates) {
            const auto &candidate = results[static_cast<size_t>(candidate_index)];
            const auto  metrics = computeContinuityMatchMetrics(tracked_target.last_result, candidate, tracking);
            if (tracking.debug.enable_verbose_logs) {
                RCLCPP_INFO(
                    node.get_logger(),
                    "Tracking continuity candidate: index=%d tracked_id=%d observed_track_id=%d %s detection=%s",
                    candidate_index, tracked_target.current_track_id, candidate.track_id,
                    summarizeContinuityMatchMetrics(metrics).c_str(), summarizeDetectionResult(candidate).c_str());
            }

            if (!metrics.accepted) {
                continue;
            }

            const bool better_match =
                !have_continuity_candidate || metrics.iou > continuity_best_metrics.iou + 1.0e-6 ||
                (std::abs(metrics.iou - continuity_best_metrics.iou) <= 1.0e-6 &&
                 metrics.center_distance_px + 1.0e-6 < continuity_best_metrics.center_distance_px) ||
                (std::abs(metrics.iou - continuity_best_metrics.iou) <= 1.0e-6 &&
                 std::abs(metrics.center_distance_px - continuity_best_metrics.center_distance_px) <= 1.0e-6 &&
                 metrics.depth_delta_mm + 1.0e-6 < continuity_best_metrics.depth_delta_mm) ||
                (std::abs(metrics.iou - continuity_best_metrics.iou) <= 1.0e-6 &&
                 std::abs(metrics.center_distance_px - continuity_best_metrics.center_distance_px) <= 1.0e-6 &&
                 std::abs(metrics.depth_delta_mm - continuity_best_metrics.depth_delta_mm) <= 1.0e-6 &&
                 candidate.detection.score >
                     results[static_cast<size_t>(continuity_best_index)].detection.score);
            if (!better_match) {
                continue;
            }

            continuity_best_index = candidate_index;
            continuity_best_metrics = metrics;
            have_continuity_candidate = true;
        }

        if (have_continuity_candidate) {
            return commitTrackedMatch(
                continuity_best_index, "continuity",
                "match=" + summarizeContinuityMatchMetrics(continuity_best_metrics));
        }

        ++tracked_target.lost_frames;
        if (tracked_target.lost_frames > tracking.max_lost_frames) {
            clearTrackedTarget();
        }
        std::string last_continuity_summary = "none";
        if (have_continuity_candidate) {
            last_continuity_summary = summarizeContinuityMatchMetrics(continuity_best_metrics);
        }
        if (tracking.debug.enable_verbose_logs) {
            RCLCPP_INFO(
                node.get_logger(),
                "Tracking failed to match existing target. tracked_id=%d candidate_count=%zu lost_frames=%d max_lost_frames=%d "
                "best_continuity=%s",
                tracked_target.current_track_id, candidates.size(), tracked_target.lost_frames, tracking.max_lost_frames,
                last_continuity_summary.c_str());
        }
        return -1;
    }

    const cv::Point2f image_center(static_cast<float>(width) * 0.5f, static_cast<float>(height) * 0.5f);
    double best_center_distance = std::numeric_limits<double>::infinity();
    int    best_index = -1;

    for (const int candidate_index : candidates) {
        const auto  &candidate = results[static_cast<size_t>(candidate_index)];
        const double center_distance = cv::norm(detectionCenter(candidate) - image_center);
        if (center_distance < best_center_distance ||
            (std::abs(center_distance - best_center_distance) < 1.0e-6 &&
             candidate.detection.score > results[static_cast<size_t>(best_index)].detection.score)) {
            best_center_distance = center_distance;
            best_index = candidate_index;
        }
    }

    if (best_index < 0) {
        return -1;
    }

    if (results[static_cast<size_t>(best_index)].track_id < 0) {
        results[static_cast<size_t>(best_index)].track_id = allocateTrackId();
        if (tracking.debug.enable_verbose_logs) {
            RCLCPP_INFO(
                node.get_logger(), "Assigned new track_id=%d to acquired detection without an existing history match",
                results[static_cast<size_t>(best_index)].track_id);
        }
    }
    refreshTrackMemoryFromDetection(results[static_cast<size_t>(best_index)], results[static_cast<size_t>(best_index)].track_id, stamp);

    tracked_target.active = true;
    tracked_target.current_track_id = results[static_cast<size_t>(best_index)].track_id;
    tracked_target.stable_frames = 1;
    tracked_target.lost_frames = 0;
    tracked_target.goal_dispatched = false;
    tracked_target.last_result = results[static_cast<size_t>(best_index)];
    tracked_target.last_stamp = stamp;
    results[static_cast<size_t>(best_index)].selected = true;
    results[static_cast<size_t>(best_index)].track_id = tracked_target.current_track_id;
    tracked_target.last_result.selected = true;
    tracked_target.last_result.track_id = tracked_target.current_track_id;
    RCLCPP_INFO(
        node.get_logger(), "Tracking acquired new target: index=%d track_id=%d",
        best_index, tracked_target.current_track_id);
    return best_index;
}

void LCVision::Impl::storeCostmap(const nav2_msgs::msg::Costmap::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(costmap_cache.mutex);
    costmap_cache.costmap = *msg;
    costmap_cache.available = true;
}

bool LCVision::Impl::copyLatestCostmap(nav2_msgs::msg::Costmap &costmap) const {
    std::lock_guard<std::mutex> lock(costmap_cache.mutex);
    if (!costmap_cache.available) {
        return false;
    }

    costmap = costmap_cache.costmap;
    return true;
}

void LCVision::Impl::storeLaserScan(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(laser_scan_cache.mutex);
    laser_scan_cache.scan = *msg;
    laser_scan_cache.available = true;
}

bool LCVision::Impl::copyLatestLaserScan(sensor_msgs::msg::LaserScan &scan) const {
    std::lock_guard<std::mutex> lock(laser_scan_cache.mutex);
    if (!laser_scan_cache.available) {
        return false;
    }

    scan = laser_scan_cache.scan;
    return true;
}

bool LCVision::Impl::transformPointToFrame(
    const geometry_msgs::msg::PointStamped &input, const std::string &target_frame,
    geometry_msgs::msg::PointStamped &output) const {
    try {
        output = tf_buffer->transform(input, target_frame);
        return true;
    } catch (const tf2::TransformException &ex) {
        RCLCPP_WARN_THROTTLE(
            node.get_logger(), *node.get_clock(), 2000, "Failed to transform point from %s to %s: %s",
            input.header.frame_id.c_str(), target_frame.c_str(), ex.what());
        return false;
    }
}

ForwardSafetyDecision LCVision::Impl::evaluateForwardSafety(
    const DetectionDepthResult &selected, const rclcpp::Time &stamp) const {
    ForwardSafetyDecision decision;
    if (!tracking.lidar.enable || !selected.camera_point_valid) {
        return decision;
    }

    geometry_msgs::msg::PointStamped camera_point;
    camera_point.header.stamp = toBuiltinTime(stamp);
    camera_point.header.frame_id = depth.frame_id;
    camera_point.point = selected.camera_point;

    geometry_msgs::msg::PointStamped base_point;
    if (!transformPointToFrame(camera_point, goal.robot_frame_id, base_point)) {
        return decision;
    }

    decision.target_distance_valid = true;
    decision.target_forward_limit_m =
        std::max(0.0f, static_cast<float>(base_point.point.x) - goal.standoff_distance_m);
    if (decision.target_forward_limit_m <= 1.0e-3f) {
        return decision;
    }

    sensor_msgs::msg::LaserScan scan;
    if (!copyLatestLaserScan(scan)) {
        return decision;
    }
    decision.scan_available = true;

    geometry_msgs::msg::TransformStamped transform;
    try {
        transform = tf_buffer->lookupTransform(
            goal.robot_frame_id, scan.header.frame_id, tf2::TimePointZero, tf2::durationFromSec(0.05));
    } catch (const tf2::TransformException &ex) {
        RCLCPP_WARN_THROTTLE(
            node.get_logger(), *node.get_clock(), 2000, "Failed to transform lidar scan from %s to %s: %s",
            scan.header.frame_id.c_str(), goal.robot_frame_id.c_str(), ex.what());
        return decision;
    }

    const tf2::Quaternion rotation_quaternion(
        transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z,
        transform.transform.rotation.w);
    const tf2::Matrix3x3 rotation(rotation_quaternion);
    const tf2::Vector3 translation(
        transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z);

    const float half_width_m = std::max(0.05f, goal.clearance_radius_m);
    for (size_t index = 0; index < scan.ranges.size(); ++index) {
        const float range = scan.ranges[index];
        if (!std::isfinite(range) || range < scan.range_min || range > scan.range_max) {
            continue;
        }

        const double angle =
            static_cast<double>(scan.angle_min) + static_cast<double>(index) * static_cast<double>(scan.angle_increment);
        const tf2::Vector3 point_in_scan(
            static_cast<double>(range) * std::cos(angle), static_cast<double>(range) * std::sin(angle), 0.0);
        const tf2::Vector3 point_in_base = rotation * point_in_scan + translation;

        const float point_x_m = static_cast<float>(point_in_base.x());
        const float point_y_m = static_cast<float>(point_in_base.y());
        if (point_x_m < 0.0f || point_x_m > decision.target_forward_limit_m) {
            continue;
        }
        if (std::abs(point_y_m) > half_width_m) {
            continue;
        }

        decision.blocked = true;
        decision.nearest_obstacle_x_m = std::min(decision.nearest_obstacle_x_m, point_x_m);
    }

    return decision;
}

bool LCVision::Impl::worldToCostmapCell(
    const nav2_msgs::msg::Costmap &costmap, const double world_x, const double world_y, int &cell_x,
    int &cell_y) const {
    const double origin_x = costmap.metadata.origin.position.x;
    const double origin_y = costmap.metadata.origin.position.y;
    const double resolution = std::max(1.0e-6f, costmap.metadata.resolution);
    const double map_x = (world_x - origin_x) / resolution;
    const double map_y = (world_y - origin_y) / resolution;
    cell_x = static_cast<int>(std::floor(map_x));
    cell_y = static_cast<int>(std::floor(map_y));

    return cell_x >= 0 && cell_y >= 0 && cell_x < static_cast<int>(costmap.metadata.size_x) &&
           cell_y < static_cast<int>(costmap.metadata.size_y);
}

bool LCVision::Impl::isFreeCost(const uint8_t cost) const {
    return cost != 255U && cost < static_cast<uint8_t>(std::clamp(goal.max_cell_cost, 0, 254));
}

bool LCVision::Impl::isCellFree(const nav2_msgs::msg::Costmap &costmap, const int cell_x, const int cell_y) const {
    if (cell_x < 0 || cell_y < 0 || cell_x >= static_cast<int>(costmap.metadata.size_x) ||
        cell_y >= static_cast<int>(costmap.metadata.size_y)) {
        return false;
    }

    const size_t index =
        static_cast<size_t>(cell_y) * static_cast<size_t>(costmap.metadata.size_x) + static_cast<size_t>(cell_x);
    if (index >= costmap.data.size()) {
        return false;
    }

    return isFreeCost(costmap.data[index]);
}

bool LCVision::Impl::hasLocalClearance(
    const nav2_msgs::msg::Costmap &costmap, const double world_x, const double world_y,
    const double clearance_radius_m) const {
    int center_x = 0;
    int center_y = 0;
    if (!worldToCostmapCell(costmap, world_x, world_y, center_x, center_y)) {
        return false;
    }

    const double resolution = std::max(1.0e-6f, costmap.metadata.resolution);
    const int radius_cells = std::max(0, static_cast<int>(std::ceil(clearance_radius_m / resolution)));
    for (int dy = -radius_cells; dy <= radius_cells; ++dy) {
        for (int dx = -radius_cells; dx <= radius_cells; ++dx) {
            const double cell_distance = std::hypot(static_cast<double>(dx), static_cast<double>(dy)) * resolution;
            if (cell_distance > clearance_radius_m) {
                continue;
            }
            if (!isCellFree(costmap, center_x + dx, center_y + dy)) {
                return false;
            }
        }
    }

    return true;
}

bool LCVision::Impl::isRayFree(
    const nav2_msgs::msg::Costmap &costmap, const geometry_msgs::msg::Point &start,
    const geometry_msgs::msg::Point &end) const {
    const double distance = std::hypot(end.x - start.x, end.y - start.y);
    const double resolution = std::max(1.0e-3f, costmap.metadata.resolution);
    const int    samples = std::max(1, static_cast<int>(std::ceil(distance / resolution)));
    for (int i = 0; i <= samples; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(samples);
        const double sample_x = start.x + (end.x - start.x) * t;
        const double sample_y = start.y + (end.y - start.y) * t;
        int          cell_x = 0;
        int          cell_y = 0;
        if (!worldToCostmapCell(costmap, sample_x, sample_y, cell_x, cell_y) || !isCellFree(costmap, cell_x, cell_y)) {
            return false;
        }
    }
    return true;
}

std::optional<geometry_msgs::msg::PoseStamped> LCVision::Impl::buildSafeGoalPose(
    const geometry_msgs::msg::PointStamped &target_map_point, const rclcpp::Time &stamp) const {
    nav2_msgs::msg::Costmap costmap;
    if (!copyLatestCostmap(costmap)) {
        RCLCPP_WARN_THROTTLE(
            node.get_logger(), *node.get_clock(), 2000, "Waiting for %s before computing a safe navigation goal",
            goal.costmap_topic.c_str());
        return std::nullopt;
    }

    geometry_msgs::msg::PointStamped robot_origin;
    robot_origin.header.stamp = toBuiltinTime(stamp);
    robot_origin.header.frame_id = goal.robot_frame_id;
    robot_origin.point.x = 0.0;
    robot_origin.point.y = 0.0;
    robot_origin.point.z = 0.0;

    geometry_msgs::msg::PointStamped robot_map_point;
    if (!transformPointToFrame(robot_origin, costmap.header.frame_id, robot_map_point)) {
        return std::nullopt;
    }

    geometry_msgs::msg::PointStamped target_costmap_point = target_map_point;
    if (target_map_point.header.frame_id != costmap.header.frame_id &&
        !transformPointToFrame(target_map_point, costmap.header.frame_id, target_costmap_point)) {
        return std::nullopt;
    }

    const double delta_x = target_costmap_point.point.x - robot_map_point.point.x;
    const double delta_y = target_costmap_point.point.y - robot_map_point.point.y;
    const double distance = std::hypot(delta_x, delta_y);
    if (distance <= static_cast<double>(goal.standoff_distance_m)) {
        return std::nullopt;
    }

    const double dir_x = delta_x / distance;
    const double dir_y = delta_y / distance;
    const double lateral_x = -dir_y;
    const double lateral_y = dir_x;
    const double max_target_distance = std::max(
        static_cast<double>(goal.standoff_distance_m), static_cast<double>(goal.max_target_distance_m));
    const int    lateral_steps = std::max(
        0, static_cast<int>(std::floor(goal.max_lateral_offset_m / std::max(goal.lateral_search_step_m, 1.0e-3f))));

    const auto try_candidate =
        [&](const double backoff_distance, const double lateral_offset,
            const bool require_ray_free) -> std::optional<geometry_msgs::msg::PoseStamped> {
        geometry_msgs::msg::Point candidate;
        candidate.x = target_costmap_point.point.x - dir_x * backoff_distance + lateral_x * lateral_offset;
        candidate.y = target_costmap_point.point.y - dir_y * backoff_distance + lateral_y * lateral_offset;
        candidate.z = 0.0;

        if (!hasLocalClearance(costmap, candidate.x, candidate.y, goal.clearance_radius_m)) {
            return std::nullopt;
        }

        if (require_ray_free && !isRayFree(costmap, robot_map_point.point, candidate)) {
            return std::nullopt;
        }

        geometry_msgs::msg::PoseStamped pose;
        pose.header.stamp = toBuiltinTime(stamp);
        pose.header.frame_id = costmap.header.frame_id;
        pose.pose.position.x = candidate.x;
        pose.pose.position.y = candidate.y;
        pose.pose.position.z = 0.0;
        pose.pose.orientation = quaternionFromYaw(
            std::atan2(target_costmap_point.point.y - candidate.y, target_costmap_point.point.x - candidate.x));
        return pose;
    };

    const auto search_candidates =
        [&](const bool require_ray_free) -> std::optional<geometry_msgs::msg::PoseStamped> {
        for (double backoff = goal.standoff_distance_m; backoff <= max_target_distance + 1.0e-6;
             backoff += std::max(goal.longitudinal_search_step_m, 1.0e-3f)) {
            if (const auto center_pose = try_candidate(backoff, 0.0, require_ray_free); center_pose.has_value()) {
                return center_pose;
            }

            for (int step = 1; step <= lateral_steps; ++step) {
                const double offset = static_cast<double>(step) * goal.lateral_search_step_m;
                if (const auto left_pose = try_candidate(backoff, offset, require_ray_free); left_pose.has_value()) {
                    return left_pose;
                }
                if (const auto right_pose = try_candidate(backoff, -offset, require_ray_free); right_pose.has_value()) {
                    return right_pose;
                }
            }
        }

        return std::nullopt;
    };

    if (const auto strict_pose = search_candidates(true); strict_pose.has_value()) {
        return strict_pose;
    }

    if (const auto relaxed_pose = search_candidates(false); relaxed_pose.has_value()) {
        RCLCPP_INFO_THROTTLE(
            node.get_logger(), *node.get_clock(), 2000,
            "Using relaxed vision goal search within %.2fm of the target", goal.max_target_distance_m);
        return relaxed_pose;
    }

    return std::nullopt;
}

void LCVision::Impl::publishTrackedTargetDebug(
    const geometry_msgs::msg::PointStamped &camera_point, const geometry_msgs::msg::PointStamped &map_point,
    const std::optional<geometry_msgs::msg::PoseStamped> &goal_pose) {
    if (!goal.publish_debug_topics) {
        return;
    }

    if (selected_target_camera_point_pub != nullptr) {
        selected_target_camera_point_pub->publish(camera_point);
    }
    if (selected_target_map_point_pub != nullptr) {
        selected_target_map_point_pub->publish(map_point);
    }
    if (goal_pose.has_value() && selected_goal_pose_pub != nullptr) {
        selected_goal_pose_pub->publish(*goal_pose);
    }
}

void LCVision::Impl::publishGoalDebugMarkers(
    const std::vector<DetectionDepthResult> &results, const rclcpp::Time &stamp) {
    if (goal_debug_marker_pub == nullptr || !goal.debug_marker.enable) {
        return;
    }

    visualization_msgs::msg::MarkerArray marker_array;
    visualization_msgs::msg::Marker      clear_marker;
    clear_marker.header.stamp = toBuiltinTime(stamp);
    clear_marker.header.frame_id = goal.debug_marker.frame_id;
    clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.push_back(clear_marker);

    int marker_id = 0;
    for (const auto &result : results) {
        if (!result.camera_point_valid) {
            continue;
        }

        geometry_msgs::msg::PointStamped camera_point;
        camera_point.header.stamp = toBuiltinTime(stamp);
        camera_point.header.frame_id = depth.frame_id;
        camera_point.point = result.camera_point;

        geometry_msgs::msg::PointStamped map_point;
        if (!transformPointToFrame(camera_point, goal.debug_marker.frame_id, map_point)) {
            continue;
        }

        const bool is_selected = result.selected;
        const float alpha = std::clamp(goal.debug_marker.alpha, 0.0f, 1.0f);

        visualization_msgs::msg::Marker point_marker;
        point_marker.header.stamp = toBuiltinTime(stamp);
        point_marker.header.frame_id = goal.debug_marker.frame_id;
        point_marker.ns = "goal_debug_points";
        point_marker.id = marker_id++;
        point_marker.type = visualization_msgs::msg::Marker::SPHERE;
        point_marker.action = visualization_msgs::msg::Marker::ADD;
        point_marker.pose.position = map_point.point;
        point_marker.pose.position.z = goal.debug_marker.z;
        point_marker.pose.orientation.w = 1.0;
        point_marker.scale.x = std::max(1.0e-3f, goal.debug_marker.point_scale);
        point_marker.scale.y = std::max(1.0e-3f, goal.debug_marker.point_scale);
        point_marker.scale.z = std::max(1.0e-3f, goal.debug_marker.point_scale);
        point_marker.color.r = is_selected ? 0.95f : 0.15f;
        point_marker.color.g = is_selected ? 0.35f : 0.85f;
        point_marker.color.b = is_selected ? 0.20f : 0.95f;
        point_marker.color.a = alpha;
        marker_array.markers.push_back(std::move(point_marker));

        visualization_msgs::msg::Marker text_marker;
        text_marker.header.stamp = toBuiltinTime(stamp);
        text_marker.header.frame_id = goal.debug_marker.frame_id;
        text_marker.ns = "goal_debug_labels";
        text_marker.id = marker_id++;
        text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        text_marker.action = visualization_msgs::msg::Marker::ADD;
        text_marker.pose.position = map_point.point;
        text_marker.pose.position.z = goal.debug_marker.z + goal.debug_marker.text_z_offset;
        text_marker.pose.orientation.w = 1.0;
        text_marker.scale.z = std::max(1.0e-3f, goal.debug_marker.text_scale);
        text_marker.color.r = 1.0f;
        text_marker.color.g = 1.0f;
        text_marker.color.b = 1.0f;
        text_marker.color.a = alpha;

        const double distance_m = static_cast<double>(result.depth_mm) / 1000.0;
        std::ostringstream label_stream;
        label_stream.setf(std::ios::fixed);
        label_stream.precision(2);
        label_stream << result.detection.label << " " << distance_m << "m";
        if (result.track_id >= 0) {
            label_stream << " id=" << result.track_id;
        }
        if (is_selected) {
            label_stream << " selected";
        }
        text_marker.text = label_stream.str();
        marker_array.markers.push_back(std::move(text_marker));
    }

    goal_debug_marker_pub->publish(marker_array);
}

void LCVision::Impl::maybeDispatchNavigationGoal(
    std::vector<DetectionDepthResult> &results, const int width, const int height, const rclcpp::Time &stamp) {
    if (!goal_runtime_enabled.load(std::memory_order_relaxed) ||
        !tracking_runtime_enabled.load(std::memory_order_relaxed)) {
        publishSelectedTargetStatus(std::nullopt, stamp);
        stopTrackingMotion("tracking or goal runtime disabled", true);
        return;
    }

    const int selected_index = selectTrackedDetection(results, width, height, stamp);
    if (selected_index < 0) {
        if (tracking.debug.enable_verbose_logs) {
            RCLCPP_INFO(
                node.get_logger(),
                "No tracked detection selected. tracked_active=%s tracked_id=%d stable_frames=%d lost_frames=%d",
                tracked_target.active ? "true" : "false", tracked_target.current_track_id, tracked_target.stable_frames,
                tracked_target.lost_frames);
        }
        publishSelectedTargetStatus(std::nullopt, stamp);
        maybeRecoverLostTarget(results, stamp);
        return;
    }

    auto &selected = results[static_cast<size_t>(selected_index)];
    selected.depth_quality_state = evaluateSelectedTargetDepthQuality(selected, stamp);
    const auto control_decision = buildDepthControlDecision(selected, selected.depth_quality_state, stamp);
    selected.control_distance_valid = control_decision.control_distance_valid;
    selected.control_distance_mm =
        control_decision.control_distance_valid ? control_decision.control_distance_m * 1000.0f : 0.0f;
    tracked_target.last_result.depth_quality_state = selected.depth_quality_state;
    tracked_target.last_result.control_distance_valid = selected.control_distance_valid;
    tracked_target.last_result.control_distance_mm = selected.control_distance_mm;
    rememberTargetObservation(selected, width, stamp);
    publishSelectedTargetStatus(std::optional<DetectionDepthResult>(selected), stamp);
    geometry_msgs::msg::PointStamped camera_point;
    camera_point.header.stamp = toBuiltinTime(stamp);
    camera_point.header.frame_id = depth.frame_id;
    camera_point.point = selected.camera_point;

    if (tryHandleSelectedTargetWithVisualServo(selected, width, stamp)) {
        return;
    }

    if (selected.depth_quality_state != DepthQualityState::Reliable) {
        if (tracking.debug.enable_verbose_logs) {
            RCLCPP_INFO(
                node.get_logger(),
                "Navigation goal suppressed because selected depth is not reliable. depth_quality=%s selected=%s",
                toString(selected.depth_quality_state), summarizeDetectionResult(selected).c_str());
        }
        return;
    }

    geometry_msgs::msg::PointStamped map_point;
    if (!tryPopulateSelectedTargetMapPoint(selected, camera_point, width, stamp, map_point)) {
        return;
    }

    const auto goal_pose = buildSafeGoalPose(map_point, stamp);
    publishTrackedTargetDebug(camera_point, map_point, goal_pose);

    if (visual_servo.command_active) {
        stopTrackingMotion("falling back to navigation-guided tracking", false);
    }

    if (!goal_pose.has_value() || tracked_target.goal_dispatched || tracked_target.stable_frames < goal.min_stable_frames ||
        navigation_goal_active.load() ||
        external_navigation_active.load(std::memory_order_relaxed)) {
        if (tracking.debug.enable_verbose_logs) {
            RCLCPP_INFO(
                node.get_logger(),
                "Navigation goal not dispatched. goal_pose=%s goal_dispatched=%s stable_frames=%d min_stable_frames=%d "
                "navigation_goal_active=%s external_navigation_active=%s selected=%s",
                goal_pose.has_value() ? "true" : "false", tracked_target.goal_dispatched ? "true" : "false",
                tracked_target.stable_frames, goal.min_stable_frames,
                navigation_goal_active.load() ? "true" : "false",
                external_navigation_active.load(std::memory_order_relaxed) ? "true" : "false",
                summarizeDetectionResult(selected).c_str());
        }
        return;
    }

    node.sendGoal(*goal_pose);
    tracked_target.goal_dispatched = true;
    RCLCPP_INFO(
        node.get_logger(), "Navigation goal dispatched for selected target. track_id=%d selected=%s",
        tracked_target.current_track_id, summarizeDetectionResult(selected).c_str());
}

} // namespace lc_vision
