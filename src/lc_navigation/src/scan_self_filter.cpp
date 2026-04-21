#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/point.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <tf2/exceptions.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/time.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace {

constexpr double kFixedResolutionAngleTolerance = 1e-6;
constexpr double kFixedResolutionTieTolerance = 1e-12;

bool nearlyEqual(const double lhs, const double rhs, const double tolerance)
{
  return std::abs(lhs - rhs) <= tolerance;
}

}  // namespace

class ScanSelfFilter : public rclcpp::Node
{
public:
  ScanSelfFilter()
  : Node("scan_self_filter"),
    tf_buffer_(this->get_clock())
  {
    declareParameters();
    loadParameters();

    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(
      tf_buffer_,
      this->get_node_base_interface(),
      this->get_node_logging_interface(),
      this->get_node_parameters_interface(),
      this->get_node_topics_interface(),
      true);

    publisher_ = this->create_publisher<sensor_msgs::msg::LaserScan>(
      output_scan_topic_, rclcpp::SensorDataQoS());
    marker_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      marker_topic_, rclcpp::QoS(10));
    subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
      input_scan_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&ScanSelfFilter::scanCallback, this, std::placeholders::_1));

    const double publish_period_seconds =
      marker_publish_rate_ > 0.0 ? (1.0 / marker_publish_rate_) : 0.2;
    marker_timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(publish_period_seconds)),
      std::bind(&ScanSelfFilter::publishFilterMarkers, this));

    parameter_callback_handle_ = this->add_on_set_parameters_callback(
      std::bind(&ScanSelfFilter::onSetParameters, this, std::placeholders::_1));

    RCLCPP_INFO(
      this->get_logger(),
      "Filtering %s -> %s in %s with bounds x:[%.4f, %.4f] y:[%.4f, %.4f] "
      "(fixed_resolution_enabled=%s, fixed_resolution_bins=%ld, "
      "fixed_resolution_use_full_circle=%s)",
      input_scan_topic_.c_str(),
      output_scan_topic_.c_str(),
      base_frame_.c_str(),
      x_min_,
      x_max_,
      y_min_,
      y_max_,
      fixed_resolution_enabled_ ? "true" : "false",
      fixed_resolution_bins_,
      fixed_resolution_use_full_circle_ ? "true" : "false");
  }

private:
  void declareParameters()
  {
    this->declare_parameter<std::string>("input_scan_topic", "/scan");
    this->declare_parameter<std::string>("output_scan_topic", "/scan_filtered");
    this->declare_parameter<std::string>("base_frame", "base_link");
    this->declare_parameter<double>("x_min", -0.1295);
    this->declare_parameter<double>("x_max", 0.1295);
    this->declare_parameter<double>("y_min", -0.1145);
    this->declare_parameter<double>("y_max", 0.1145);
    this->declare_parameter<double>("transform_timeout_sec", 0.05);
    this->declare_parameter<std::string>("marker_topic", "/scan_self_filter/marker");
    this->declare_parameter<double>("marker_publish_rate", 5.0);
    this->declare_parameter<double>("marker_z", 0.03);
    this->declare_parameter<double>("marker_line_width", 0.01);
    this->declare_parameter<double>("marker_alpha", 0.2);
    this->declare_parameter<bool>("fixed_resolution_enabled", false);
    this->declare_parameter<int64_t>("fixed_resolution_bins", 460);
    this->declare_parameter<double>("fixed_resolution_angle_min", 0.0);
    this->declare_parameter<bool>("fixed_resolution_use_full_circle", false);
  }

  void loadParameters()
  {
    input_scan_topic_ = this->get_parameter("input_scan_topic").as_string();
    output_scan_topic_ = this->get_parameter("output_scan_topic").as_string();
    base_frame_ = this->get_parameter("base_frame").as_string();
    x_min_ = this->get_parameter("x_min").as_double();
    x_max_ = this->get_parameter("x_max").as_double();
    y_min_ = this->get_parameter("y_min").as_double();
    y_max_ = this->get_parameter("y_max").as_double();
    transform_timeout_sec_ = this->get_parameter("transform_timeout_sec").as_double();
    marker_topic_ = this->get_parameter("marker_topic").as_string();
    marker_publish_rate_ = this->get_parameter("marker_publish_rate").as_double();
    marker_z_ = this->get_parameter("marker_z").as_double();
    marker_line_width_ = this->get_parameter("marker_line_width").as_double();
    marker_alpha_ = this->get_parameter("marker_alpha").as_double();
    fixed_resolution_enabled_ = this->get_parameter("fixed_resolution_enabled").as_bool();
    fixed_resolution_bins_ = this->get_parameter("fixed_resolution_bins").as_int();
    fixed_resolution_angle_min_config_ =
      this->get_parameter("fixed_resolution_angle_min").as_double();
    fixed_resolution_use_full_circle_ =
      this->get_parameter("fixed_resolution_use_full_circle").as_bool();
  }

  void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
  {
    sensor_msgs::msg::LaserScan filtered_scan = *msg;

    if (msg->header.frame_id.empty()) {
      warnThrottled(
        "LaserScan has empty frame_id; bypassing self filter for this frame.",
        5.0,
        "missing_frame_id");
    } else {
      const auto transform = lookupTransform(msg->header.frame_id);
      if (transform.has_value()) {
        const auto & transform_msg = transform->transform;
        const tf2::Quaternion rotation_quaternion(
          transform_msg.rotation.x,
          transform_msg.rotation.y,
          transform_msg.rotation.z,
          transform_msg.rotation.w);
        const tf2::Matrix3x3 rotation(rotation_quaternion);
        const tf2::Vector3 translation(
          transform_msg.translation.x,
          transform_msg.translation.y,
          transform_msg.translation.z);

        for (std::size_t index = 0; index < msg->ranges.size(); ++index) {
          const float scan_range = msg->ranges[index];
          if (!std::isfinite(scan_range)) {
            continue;
          }
          if (scan_range < msg->range_min || scan_range > msg->range_max) {
            continue;
          }

          const double angle =
            static_cast<double>(msg->angle_min) +
            static_cast<double>(index) * static_cast<double>(msg->angle_increment);
          const tf2::Vector3 point_in_laser(
            static_cast<double>(scan_range) * std::cos(angle),
            static_cast<double>(scan_range) * std::sin(angle),
            0.0);
          const tf2::Vector3 point_in_base = rotation * point_in_laser + translation;

          if (isInsideFilterBox(point_in_base.x(), point_in_base.y())) {
            filtered_scan.ranges[index] = std::numeric_limits<float>::infinity();
          }
        }
      }
    }

    sensor_msgs::msg::LaserScan output_scan = filtered_scan;
    if (fixed_resolution_enabled_) {
      output_scan = resampleScanToFixedResolution(filtered_scan);
    }

    publisher_->publish(output_scan);
  }

  sensor_msgs::msg::LaserScan resampleScanToFixedResolution(
    const sensor_msgs::msg::LaserScan & msg)
  {
    if (fixed_resolution_bins_ < 2) {
      warnThrottled(
        "fixed_resolution_bins must be at least 2; publishing the filtered scan unchanged.",
        5.0,
        "invalid_fixed_resolution_bins");
      return msg;
    }

    if (!lockFixedResolutionGrid(msg)) {
      return msg;
    }

    if (!fixed_resolution_angle_min_.has_value() ||
      !fixed_resolution_angle_max_.has_value() ||
      !fixed_resolution_angle_increment_.has_value())
    {
      return msg;
    }

    if (!fixed_resolution_use_full_circle_ &&
      (!nearlyEqual(
          static_cast<double>(msg.angle_min),
          *fixed_resolution_angle_min_,
          kFixedResolutionAngleTolerance) ||
      !nearlyEqual(
          static_cast<double>(msg.angle_max),
          *fixed_resolution_angle_max_,
          kFixedResolutionAngleTolerance)))
    {
      warnThrottled(
        "LaserScan angle range changed after fixed-resolution grid was locked; "
        "continuing to publish on the original fixed grid.",
        5.0,
        "fixed_resolution_angle_range_changed");
    }

    sensor_msgs::msg::LaserScan output_scan;
    output_scan.header = msg.header;
    output_scan.angle_min = static_cast<float>(*fixed_resolution_angle_min_);
    output_scan.angle_max = static_cast<float>(*fixed_resolution_angle_max_);
    output_scan.angle_increment = static_cast<float>(*fixed_resolution_angle_increment_);
    output_scan.time_increment = computeFixedResolutionTimeIncrement(msg);
    output_scan.scan_time = msg.scan_time;
    output_scan.range_min = msg.range_min;
    output_scan.range_max = msg.range_max;
    output_scan.ranges.assign(
      static_cast<std::size_t>(fixed_resolution_bins_),
      std::numeric_limits<float>::infinity());
    output_scan.intensities.assign(
      static_cast<std::size_t>(fixed_resolution_bins_),
      0.0F);

    std::vector<double> best_angle_errors(
      static_cast<std::size_t>(fixed_resolution_bins_),
      std::numeric_limits<double>::infinity());
    std::vector<float> best_ranges(
      static_cast<std::size_t>(fixed_resolution_bins_),
      std::numeric_limits<float>::infinity());

    for (std::size_t index = 0; index < msg.ranges.size(); ++index) {
      const float scan_range = msg.ranges[index];
      if (!std::isfinite(scan_range)) {
        continue;
      }
      if (scan_range < msg.range_min || scan_range > msg.range_max) {
        continue;
      }

      const double angle =
        static_cast<double>(msg.angle_min) +
        static_cast<double>(index) * static_cast<double>(msg.angle_increment);
      const double mapped_index_unrounded =
        (angle - *fixed_resolution_angle_min_) / *fixed_resolution_angle_increment_;
      const auto mapped_index = static_cast<int64_t>(std::llround(mapped_index_unrounded));
      if (mapped_index < 0 || mapped_index >= fixed_resolution_bins_) {
        continue;
      }

      const double mapped_angle =
        *fixed_resolution_angle_min_ +
        static_cast<double>(mapped_index) * *fixed_resolution_angle_increment_;
      const double angle_error = std::abs(angle - mapped_angle);
      const std::size_t output_index = static_cast<std::size_t>(mapped_index);
      bool should_replace =
        angle_error + kFixedResolutionTieTolerance < best_angle_errors[output_index];
      should_replace = should_replace || (
        nearlyEqual(
          angle_error,
          best_angle_errors[output_index],
          kFixedResolutionTieTolerance) &&
        scan_range < best_ranges[output_index]);

      if (!should_replace) {
        continue;
      }

      output_scan.ranges[output_index] = scan_range;
      output_scan.intensities[output_index] =
        index < msg.intensities.size() ? msg.intensities[index] : 0.0F;
      best_angle_errors[output_index] = angle_error;
      best_ranges[output_index] = scan_range;
    }

    return output_scan;
  }

  bool lockFixedResolutionGrid(const sensor_msgs::msg::LaserScan & msg)
  {
    if (fixed_resolution_angle_min_.has_value() &&
      fixed_resolution_angle_max_.has_value() &&
      fixed_resolution_angle_increment_.has_value())
    {
      return true;
    }

    double angle_min = 0.0;
    double angle_max = 0.0;
    double angle_increment = 0.0;

    if (fixed_resolution_use_full_circle_) {
      angle_min = fixed_resolution_angle_min_config_;
      angle_increment = (2.0 * M_PI) / static_cast<double>(fixed_resolution_bins_);
      angle_max =
        angle_min + angle_increment * static_cast<double>(fixed_resolution_bins_ - 1);
    } else {
      angle_min = static_cast<double>(msg.angle_min);
      angle_increment =
        (static_cast<double>(msg.angle_max) - static_cast<double>(msg.angle_min)) /
        static_cast<double>(fixed_resolution_bins_ - 1);
      angle_max = static_cast<double>(msg.angle_max);
    }

    if (!std::isfinite(angle_increment) || std::abs(angle_increment) <= 1e-12) {
      warnThrottled(
        "Unable to lock the fixed-resolution grid because the incoming scan angle range is "
        "invalid; publishing the filtered scan unchanged.",
        5.0,
        "invalid_fixed_resolution_grid");
      return false;
    }

    fixed_resolution_angle_min_ = angle_min;
    fixed_resolution_angle_max_ = angle_max;
    fixed_resolution_angle_increment_ = angle_increment;

    RCLCPP_INFO(
      this->get_logger(),
      "Locked fixed-resolution scan grid to angle_min=%.6f, angle_max=%.6f, bins=%ld, "
      "use_full_circle=%s",
      *fixed_resolution_angle_min_,
      *fixed_resolution_angle_max_,
      fixed_resolution_bins_,
      fixed_resolution_use_full_circle_ ? "true" : "false");
    return true;
  }

  float computeFixedResolutionTimeIncrement(const sensor_msgs::msg::LaserScan & msg) const
  {
    if (msg.scan_time > 0.0F) {
      return static_cast<float>(
        static_cast<double>(msg.scan_time) / static_cast<double>(fixed_resolution_bins_));
    }

    if (msg.time_increment > 0.0F && !msg.ranges.empty()) {
      return static_cast<float>(
        static_cast<double>(msg.time_increment) * static_cast<double>(msg.ranges.size()) /
        static_cast<double>(fixed_resolution_bins_));
    }

    return 0.0F;
  }

  std::optional<geometry_msgs::msg::TransformStamped> lookupTransform(const std::string & scan_frame)
  {
    try {
      return tf_buffer_.lookupTransform(
        base_frame_,
        scan_frame,
        rclcpp::Time(0, 0, RCL_ROS_TIME),
        rclcpp::Duration::from_seconds(transform_timeout_sec_));
    } catch (const tf2::TransformException & exc) {
      warnThrottled(
        "Unable to transform " + scan_frame + " -> " + base_frame_ +
        "; bypassing self filter. Details: " + exc.what());
      return std::nullopt;
    }
  }

  bool isInsideFilterBox(const double x, const double y) const
  {
    return x_min_ <= x && x <= x_max_ && y_min_ <= y && y <= y_max_;
  }

  rcl_interfaces::msg::SetParametersResult onSetParameters(
    const std::vector<rclcpp::Parameter> & parameters)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;

    for (const auto & parameter : parameters) {
      if (parameter.get_name() == "fixed_resolution_bins" && parameter.as_int() < 2) {
        result.successful = false;
        result.reason = "fixed_resolution_bins must be at least 2";
        return result;
      }
    }

    for (const auto & parameter : parameters) {
      const auto & name = parameter.get_name();
      if (name == "base_frame") {
        base_frame_ = parameter.as_string();
      } else if (name == "x_min") {
        x_min_ = parameter.as_double();
      } else if (name == "x_max") {
        x_max_ = parameter.as_double();
      } else if (name == "y_min") {
        y_min_ = parameter.as_double();
      } else if (name == "y_max") {
        y_max_ = parameter.as_double();
      } else if (name == "transform_timeout_sec") {
        transform_timeout_sec_ = parameter.as_double();
      } else if (name == "marker_z") {
        marker_z_ = parameter.as_double();
      } else if (name == "marker_line_width") {
        marker_line_width_ = parameter.as_double();
      } else if (name == "marker_alpha") {
        marker_alpha_ = parameter.as_double();
      } else if (name == "fixed_resolution_enabled") {
        fixed_resolution_enabled_ = parameter.as_bool();
        resetFixedResolutionGrid();
      } else if (name == "fixed_resolution_bins") {
        fixed_resolution_bins_ = parameter.as_int();
        resetFixedResolutionGrid();
      } else if (name == "fixed_resolution_angle_min") {
        fixed_resolution_angle_min_config_ = parameter.as_double();
        resetFixedResolutionGrid();
      } else if (name == "fixed_resolution_use_full_circle") {
        fixed_resolution_use_full_circle_ = parameter.as_bool();
        resetFixedResolutionGrid();
      }
    }

    publishFilterMarkers();
    return result;
  }

  void resetFixedResolutionGrid()
  {
    fixed_resolution_angle_min_.reset();
    fixed_resolution_angle_max_.reset();
    fixed_resolution_angle_increment_.reset();
  }

  void publishFilterMarkers()
  {
    visualization_msgs::msg::MarkerArray marker_array;
    const auto now = this->get_clock()->now();

    visualization_msgs::msg::Marker fill_marker;
    fill_marker.header.frame_id = base_frame_;
    fill_marker.header.stamp = now;
    fill_marker.ns = "scan_self_filter";
    fill_marker.id = 0;
    fill_marker.type = visualization_msgs::msg::Marker::CUBE;
    fill_marker.action = visualization_msgs::msg::Marker::ADD;
    fill_marker.pose.orientation.w = 1.0;
    fill_marker.pose.position.x = (x_min_ + x_max_) / 2.0;
    fill_marker.pose.position.y = (y_min_ + y_max_) / 2.0;
    fill_marker.pose.position.z = marker_z_ / 2.0;
    fill_marker.scale.x = std::max(x_max_ - x_min_, 1e-6);
    fill_marker.scale.y = std::max(y_max_ - y_min_, 1e-6);
    fill_marker.scale.z = std::max(marker_z_, 1e-3);
    fill_marker.color.r = 1.0F;
    fill_marker.color.g = 0.2F;
    fill_marker.color.b = 0.2F;
    fill_marker.color.a = static_cast<float>(std::clamp(marker_alpha_, 0.0, 1.0));

    visualization_msgs::msg::Marker outline_marker;
    outline_marker.header.frame_id = base_frame_;
    outline_marker.header.stamp = now;
    outline_marker.ns = "scan_self_filter";
    outline_marker.id = 1;
    outline_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    outline_marker.action = visualization_msgs::msg::Marker::ADD;
    outline_marker.pose.orientation.w = 1.0;
    outline_marker.scale.x = std::max(marker_line_width_, 1e-4);
    outline_marker.color.r = 1.0F;
    outline_marker.color.g = 0.95F;
    outline_marker.color.b = 0.2F;
    outline_marker.color.a = 1.0F;
    outline_marker.points = {
      makePoint(x_min_, y_min_),
      makePoint(x_max_, y_min_),
      makePoint(x_max_, y_max_),
      makePoint(x_min_, y_max_),
      makePoint(x_min_, y_min_)};

    marker_array.markers.push_back(fill_marker);
    marker_array.markers.push_back(outline_marker);
    marker_publisher_->publish(marker_array);
  }

  geometry_msgs::msg::Point makePoint(const double x, const double y) const
  {
    geometry_msgs::msg::Point point;
    point.x = x;
    point.y = y;
    point.z = marker_z_;
    return point;
  }

  void warnThrottled(
    const std::string & message,
    const double throttle_sec = 5.0,
    const std::string & key = "")
  {
    const int64_t now_ns = this->get_clock()->now().nanoseconds();
    const std::string warning_key = key.empty() ? message : key;
    const auto it = last_warning_time_ns_by_key_.find(warning_key);
    const int64_t last_warning_time_ns =
      it == last_warning_time_ns_by_key_.end() ? 0 : it->second;
    if (now_ns - last_warning_time_ns >= static_cast<int64_t>(throttle_sec * 1e9)) {
      RCLCPP_WARN(this->get_logger(), "%s", message.c_str());
      last_warning_time_ns_by_key_[warning_key] = now_ns;
    }
  }

  std::string input_scan_topic_;
  std::string output_scan_topic_;
  std::string base_frame_;
  std::string marker_topic_;

  double x_min_ = -0.1295;
  double x_max_ = 0.1295;
  double y_min_ = -0.1145;
  double y_max_ = 0.1145;
  double transform_timeout_sec_ = 0.05;
  double marker_publish_rate_ = 5.0;
  double marker_z_ = 0.03;
  double marker_line_width_ = 0.01;
  double marker_alpha_ = 0.2;

  bool fixed_resolution_enabled_ = false;
  int64_t fixed_resolution_bins_ = 460;
  double fixed_resolution_angle_min_config_ = 0.0;
  bool fixed_resolution_use_full_circle_ = false;
  std::optional<double> fixed_resolution_angle_min_;
  std::optional<double> fixed_resolution_angle_max_;
  std::optional<double> fixed_resolution_angle_increment_;

  tf2_ros::Buffer tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr subscription_;
  rclcpp::TimerBase::SharedPtr marker_timer_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_handle_;

  std::unordered_map<std::string, int64_t> last_warning_time_ns_by_key_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ScanSelfFilter>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
