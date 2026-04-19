#include "lc_vision_internal.hpp"

namespace lc_vision {

void LCVision::Impl::clearTrackedTarget() {
    const int32_t next_track_id = tracked_target.next_track_id;
    tracked_target.active = false;
    tracked_target.current_track_id = -1;
    tracked_target.stable_frames = 0;
    tracked_target.lost_frames = 0;
    tracked_target.goal_dispatched = false;
    tracked_target.last_result = DetectionDepthResult();
    tracked_target.last_stamp = rclcpp::Time{};
    tracked_target.next_track_id = next_track_id;
}

cv::Point2f LCVision::Impl::detectionCenter(const DetectionDepthResult &result) {
    return cv::Point2f(
        static_cast<float>(result.detection.box.x) + static_cast<float>(result.detection.box.width) * 0.5f,
        static_cast<float>(result.detection.box.y) + static_cast<float>(result.detection.box.height) * 0.5f);
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

void LCVision::Impl::rememberTargetObservation(
    const DetectionDepthResult &result, const int width, const rclcpp::Time &stamp,
    const std::optional<geometry_msgs::msg::PointStamped> &map_point) {
    remembered_target.available = true;
    remembered_target.stamp = stamp;
    remembered_target.image_offset_px = detectionCenter(result).x - static_cast<float>(width) * 0.5f;
    remembered_target.camera_lateral_m =
        result.camera_point_valid ? static_cast<float>(result.camera_point.x) : 0.0f;
    remembered_target.map_point_valid = map_point.has_value();
    if (map_point.has_value()) {
        remembered_target.map_point = *map_point;
    }
}

void LCVision::Impl::publishRecoveryCommand(const double angular_velocity) {
    if (recovery_cmd_vel_pub == nullptr) {
        return;
    }

    geometry_msgs::msg::Twist cmd;
    cmd.angular.z = angular_velocity;
    recovery_cmd_vel_pub->publish(cmd);
}

void LCVision::Impl::stopRecoveryRotation(const std::string &reason) {
    if (!recovery_rotation.active) {
        return;
    }

    recovery_rotation.active = false;
    publishRecoveryCommand(0.0);
    RCLCPP_INFO(node.get_logger(), "Stopped visual recovery rotation: %s", reason.c_str());
}

void LCVision::Impl::maybeRecoverLostTarget(const rclcpp::Time &stamp) {
    if (!tracking.recovery.enable) {
        return;
    }

    if (external_navigation_active.load(std::memory_order_relaxed)) {
        if (recovery_rotation.active) {
            stopRecoveryRotation("external navigation is active");
        }
        return;
    }

    if (navigation_goal_active.load(std::memory_order_relaxed)) {
        if (recovery_rotation.active) {
            stopRecoveryRotation("navigation is active");
        }
        return;
    }

    if (!remembered_target.available) {
        if (recovery_rotation.active) {
            stopRecoveryRotation("no remembered target");
        }
        return;
    }

    const double memory_age =
        (stamp - remembered_target.stamp).seconds();
    if (memory_age > static_cast<double>(tracking.recovery.memory_timeout_sec)) {
        if (recovery_rotation.active) {
            stopRecoveryRotation("remembered target expired");
        }
        remembered_target.available = false;
        return;
    }

    if (!recovery_rotation.active &&
        memory_age < static_cast<double>(tracking.recovery.lost_delay_sec)) {
        return;
    }

    if (!recovery_rotation.active) {
        int direction = 1;
        if (std::abs(remembered_target.image_offset_px) > 1.0f) {
            direction = remembered_target.image_offset_px > 0.0f ? -1 : 1;
        } else if (std::abs(remembered_target.camera_lateral_m) > 1.0e-3f) {
            direction = remembered_target.camera_lateral_m > 0.0f ? -1 : 1;
        }

        node.cancelCurrentNavigationGoal("vision target lost, starting visual recovery rotation");
        tracked_target.goal_dispatched = false;
        recovery_rotation.active = true;
        recovery_rotation.direction = direction;
        recovery_rotation.start_stamp = stamp;
        RCLCPP_INFO(
            node.get_logger(), "Started visual recovery rotation: direction=%d memory_age=%.2fs", direction, memory_age);
    }

    const double recovery_age = (stamp - recovery_rotation.start_stamp).seconds();
    if (recovery_age > static_cast<double>(tracking.recovery.search_timeout_sec)) {
        stopRecoveryRotation("recovery timeout");
        clearTrackedTarget();
        remembered_target.available = false;
        return;
    }

    publishRecoveryCommand(
        static_cast<double>(recovery_rotation.direction) * static_cast<double>(tracking.recovery.turn_speed_rad_s));
}

int LCVision::Impl::selectTrackedDetection(
    std::vector<DetectionDepthResult> &results, const int width, const int height, const rclcpp::Time &stamp) {
    if (!tracking_runtime_enabled.load(std::memory_order_relaxed)) {
        if (tracked_target.active) {
            clearTrackedTarget();
        }
        return -1;
    }

    std::vector<int> candidates;
    candidates.reserve(results.size());
    for (size_t i = 0; i < results.size(); ++i) {
        if (results[i].depth_valid && results[i].camera_point_valid) {
            candidates.push_back(static_cast<int>(i));
        }
    }

    if (candidates.empty()) {
        if (tracked_target.active) {
            ++tracked_target.lost_frames;
            if (tracked_target.lost_frames > tracking.max_lost_frames) {
                clearTrackedTarget();
            }
        }
        return -1;
    }

    if (tracked_target.active) {
        const cv::Point2f previous_center = detectionCenter(tracked_target.last_result);
        const float       previous_depth = tracked_target.last_result.depth_mm;
        double            best_cost = std::numeric_limits<double>::infinity();
        int               best_index = -1;

        for (const int candidate_index : candidates) {
            auto &candidate = results[static_cast<size_t>(candidate_index)];
            const cv::Point2f center = detectionCenter(candidate);
            const double      center_distance = cv::norm(center - previous_center);
            if (center_distance > static_cast<double>(tracking.max_center_distance_px)) {
                continue;
            }

            const double iou = rectIou(tracked_target.last_result.detection.box, candidate.detection.box);
            const double depth_delta = std::abs(static_cast<double>(candidate.depth_mm - previous_depth));
            if (depth_delta > static_cast<double>(tracking.max_depth_delta_mm)) {
                continue;
            }

            if (iou < static_cast<double>(tracking.min_iou_for_match) &&
                center_distance > static_cast<double>(tracking.max_center_distance_px) * 0.5) {
                continue;
            }

            const double cost =
                center_distance / std::max(1.0f, tracking.max_center_distance_px) +
                depth_delta / std::max(1.0f, tracking.max_depth_delta_mm) + (1.0 - iou);
            if (cost < best_cost) {
                best_cost = cost;
                best_index = candidate_index;
            }
        }

        if (best_index >= 0) {
            stopRecoveryRotation("target reacquired");
            tracked_target.lost_frames = 0;
            ++tracked_target.stable_frames;
            tracked_target.last_result = results[static_cast<size_t>(best_index)];
            tracked_target.last_stamp = stamp;
            results[static_cast<size_t>(best_index)].selected = true;
            results[static_cast<size_t>(best_index)].track_id = tracked_target.current_track_id;
            tracked_target.last_result.selected = true;
            tracked_target.last_result.track_id = tracked_target.current_track_id;
            return best_index;
        }

        ++tracked_target.lost_frames;
        if (tracked_target.lost_frames > tracking.max_lost_frames) {
            clearTrackedTarget();
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

    tracked_target.active = true;
    tracked_target.current_track_id = tracked_target.next_track_id++;
    tracked_target.stable_frames = 1;
    tracked_target.lost_frames = 0;
    tracked_target.goal_dispatched = false;
    stopRecoveryRotation("target acquired");
    tracked_target.last_result = results[static_cast<size_t>(best_index)];
    tracked_target.last_stamp = stamp;
    results[static_cast<size_t>(best_index)].selected = true;
    results[static_cast<size_t>(best_index)].track_id = tracked_target.current_track_id;
    tracked_target.last_result.selected = true;
    tracked_target.last_result.track_id = tracked_target.current_track_id;
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
        stopRecoveryRotation("tracking or goal runtime disabled");
        return;
    }

    const int selected_index = selectTrackedDetection(results, width, height, stamp);
    if (selected_index < 0) {
        maybeRecoverLostTarget(stamp);
        return;
    }

    auto &selected = results[static_cast<size_t>(selected_index)];
    rememberTargetObservation(selected, width, stamp);
    geometry_msgs::msg::PointStamped camera_point;
    camera_point.header.stamp = toBuiltinTime(stamp);
    camera_point.header.frame_id = depth.frame_id;
    camera_point.point = selected.camera_point;

    geometry_msgs::msg::PointStamped map_point;
    if (!transformPointToFrame(camera_point, goal.global_frame_id, map_point)) {
        return;
    }

    rememberTargetObservation(selected, width, stamp, map_point);

    const auto goal_pose = buildSafeGoalPose(map_point, stamp);
    publishTrackedTargetDebug(camera_point, map_point, goal_pose);

    if (!goal_pose.has_value() || tracked_target.goal_dispatched || tracked_target.stable_frames < goal.min_stable_frames ||
        navigation_goal_active.load() ||
        external_navigation_active.load(std::memory_order_relaxed)) {
        return;
    }

    node.sendGoal(*goal_pose);
    tracked_target.goal_dispatched = true;
}

} // namespace lc_vision
