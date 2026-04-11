#pragma once

#include <memory>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

namespace lc_vision {

class LCVision : public rclcpp::Node {
public:
    explicit LCVision(const rclcpp::NodeOptions &options);
    ~LCVision() override;

private:
    struct Impl;

    void getParams();
    void prepareModel();
    void sendGoal(const geometry_msgs::msg::PoseStamped &goal);

    using NavigateToPose = nav2_msgs::action::NavigateToPose;

    std::unique_ptr<Impl> impl_;
    rclcpp_action::Client<NavigateToPose>::SharedPtr nav_to_pose_;
};

} // namespace lc_vision
