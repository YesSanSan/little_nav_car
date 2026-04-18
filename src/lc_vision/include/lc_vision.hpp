#pragma once

#include <memory>
#include <string>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
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
    void cancelCurrentNavigationGoal(const std::string &reason);
    rcl_interfaces::msg::SetParametersResult
    handleRuntimeParameters(const std::vector<rclcpp::Parameter> &parameters);

    using NavigateToPose = nav2_msgs::action::NavigateToPose;

    std::unique_ptr<Impl> impl_;
    rclcpp_action::Client<NavigateToPose>::SharedPtr nav_to_pose_;
};

} // namespace lc_vision
