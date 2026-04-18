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
    const double      center_gate =
        static_cast<double>(std::min(width, height)) * static_cast<double>(tracking.initial_center_gate_ratio);
    double best_center_distance = std::numeric_limits<double>::infinity();
    int    best_index = -1;

    for (const int candidate_index : candidates) {
        const auto  &candidate = results[static_cast<size_t>(candidate_index)];
        const double center_distance = cv::norm(detectionCenter(candidate) - image_center);
        if (center_distance > center_gate) {
            continue;
        }

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
    const int    lateral_steps = std::max(
        0, static_cast<int>(std::floor(goal.max_lateral_offset_m / std::max(goal.lateral_search_step_m, 1.0e-3f))));

    const auto try_candidate =
        [&](const double backoff_distance, const double lateral_offset) -> std::optional<geometry_msgs::msg::PoseStamped> {
        geometry_msgs::msg::Point candidate;
        candidate.x = target_costmap_point.point.x - dir_x * backoff_distance + lateral_x * lateral_offset;
        candidate.y = target_costmap_point.point.y - dir_y * backoff_distance + lateral_y * lateral_offset;
        candidate.z = 0.0;

        if (!hasLocalClearance(costmap, candidate.x, candidate.y, goal.clearance_radius_m)) {
            return std::nullopt;
        }

        if (!isRayFree(costmap, robot_map_point.point, candidate)) {
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

    for (double backoff = goal.standoff_distance_m;
         backoff <= goal.standoff_distance_m + goal.max_backoff_distance_m + 1.0e-6;
         backoff += std::max(goal.longitudinal_search_step_m, 1.0e-3f)) {
        if (const auto center_pose = try_candidate(backoff, 0.0); center_pose.has_value()) {
            return center_pose;
        }

        for (int step = 1; step <= lateral_steps; ++step) {
            const double offset = static_cast<double>(step) * goal.lateral_search_step_m;
            if (const auto left_pose = try_candidate(backoff, offset); left_pose.has_value()) {
                return left_pose;
            }
            if (const auto right_pose = try_candidate(backoff, -offset); right_pose.has_value()) {
                return right_pose;
            }
        }
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

void LCVision::Impl::maybeDispatchNavigationGoal(
    std::vector<DetectionDepthResult> &results, const int width, const int height, const rclcpp::Time &stamp) {
    if (!goal_runtime_enabled.load(std::memory_order_relaxed) ||
        !tracking_runtime_enabled.load(std::memory_order_relaxed)) {
        return;
    }

    const int selected_index = selectTrackedDetection(results, width, height, stamp);
    if (selected_index < 0) {
        return;
    }

    auto &selected = results[static_cast<size_t>(selected_index)];
    geometry_msgs::msg::PointStamped camera_point;
    camera_point.header.stamp = toBuiltinTime(stamp);
    camera_point.header.frame_id = depth.frame_id;
    camera_point.point = selected.camera_point;

    geometry_msgs::msg::PointStamped map_point;
    if (!transformPointToFrame(camera_point, goal.global_frame_id, map_point)) {
        return;
    }

    const auto goal_pose = buildSafeGoalPose(map_point, stamp);
    publishTrackedTargetDebug(camera_point, map_point, goal_pose);

    if (!goal_pose.has_value() || tracked_target.goal_dispatched || tracked_target.stable_frames < goal.min_stable_frames ||
        navigation_goal_active.load()) {
        return;
    }

    node.sendGoal(*goal_pose);
    tracked_target.goal_dispatched = true;
}

} // namespace lc_vision
