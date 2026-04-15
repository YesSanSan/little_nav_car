#!/usr/bin/env python3

import math
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


def quaternion_to_rotation_matrix(
    x: float, y: float, z: float, w: float
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


class ScanSelfFilter(Node):
    def __init__(self) -> None:
        super().__init__("scan_self_filter")

        self.declare_parameter("input_scan_topic", "/scan")
        self.declare_parameter("output_scan_topic", "/scan_filtered")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("x_min", -0.1295)
        self.declare_parameter("x_max", 0.1295)
        self.declare_parameter("y_min", -0.1145)
        self.declare_parameter("y_max", 0.1145)
        self.declare_parameter("transform_timeout_sec", 0.05)
        self.declare_parameter("marker_topic", "/scan_self_filter/marker")
        self.declare_parameter("marker_publish_rate", 5.0)
        self.declare_parameter("marker_z", 0.03)
        self.declare_parameter("marker_line_width", 0.01)
        self.declare_parameter("marker_alpha", 0.2)
        self.declare_parameter("fixed_resolution_enabled", False)
        self.declare_parameter("fixed_resolution_bins", 460)
        self.declare_parameter("fixed_resolution_angle_min", 0.0)
        self.declare_parameter("fixed_resolution_use_full_circle", False)

        self.input_scan_topic = self.get_parameter("input_scan_topic").get_parameter_value().string_value
        self.output_scan_topic = self.get_parameter("output_scan_topic").get_parameter_value().string_value
        self.base_frame = self.get_parameter("base_frame").get_parameter_value().string_value
        self.x_min = self.get_parameter("x_min").get_parameter_value().double_value
        self.x_max = self.get_parameter("x_max").get_parameter_value().double_value
        self.y_min = self.get_parameter("y_min").get_parameter_value().double_value
        self.y_max = self.get_parameter("y_max").get_parameter_value().double_value
        timeout_sec = self.get_parameter("transform_timeout_sec").get_parameter_value().double_value
        self.transform_timeout = Duration(seconds=timeout_sec)
        self.marker_topic = self.get_parameter("marker_topic").get_parameter_value().string_value
        self.marker_publish_rate = self.get_parameter("marker_publish_rate").get_parameter_value().double_value
        self.marker_z = self.get_parameter("marker_z").get_parameter_value().double_value
        self.marker_line_width = self.get_parameter("marker_line_width").get_parameter_value().double_value
        self.marker_alpha = self.get_parameter("marker_alpha").get_parameter_value().double_value
        self.fixed_resolution_enabled = (
            self.get_parameter("fixed_resolution_enabled").get_parameter_value().bool_value
        )
        self.fixed_resolution_bins = self.get_parameter("fixed_resolution_bins").get_parameter_value().integer_value
        self.fixed_resolution_angle_min_config = (
            self.get_parameter("fixed_resolution_angle_min").get_parameter_value().double_value
        )
        self.fixed_resolution_use_full_circle = (
            self.get_parameter("fixed_resolution_use_full_circle").get_parameter_value().bool_value
        )
        self.fixed_resolution_angle_min: Optional[float] = None
        self.fixed_resolution_angle_max: Optional[float] = None
        self.fixed_resolution_angle_increment: Optional[float] = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        self.publisher = self.create_publisher(LaserScan, self.output_scan_topic, qos_profile_sensor_data)
        self.marker_publisher = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self.subscription = self.create_subscription(
            LaserScan,
            self.input_scan_topic,
            self.scan_callback,
            qos_profile_sensor_data,
        )
        publish_period = 1.0 / self.marker_publish_rate if self.marker_publish_rate > 0.0 else 0.2
        self.marker_timer = self.create_timer(publish_period, self.publish_filter_markers)

        self._last_warning_time_ns_by_key: Dict[str, int] = {}
        self.add_on_set_parameters_callback(self.on_set_parameters)

        self.get_logger().info(
            f"Filtering {self.input_scan_topic} -> {self.output_scan_topic} in {self.base_frame} "
            f"with bounds x:[{self.x_min:.4f}, {self.x_max:.4f}] y:[{self.y_min:.4f}, {self.y_max:.4f}] "
            f"(fixed_resolution_enabled={self.fixed_resolution_enabled}, "
            f"fixed_resolution_bins={self.fixed_resolution_bins}, "
            f"fixed_resolution_use_full_circle={self.fixed_resolution_use_full_circle})"
        )

    def scan_callback(self, msg: LaserScan) -> None:
        filtered_scan = self.clone_scan(msg)

        if not msg.header.frame_id:
            self.warn_throttled(
                "LaserScan has empty frame_id; bypassing self filter for this frame.",
                key="missing_frame_id",
            )
        else:
            transform = self.lookup_transform(msg.header.frame_id)
            if transform is not None:
                rotation = quaternion_to_rotation_matrix(
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w,
                )
                translation = (
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                )

                for index, scan_range in enumerate(msg.ranges):
                    if not math.isfinite(scan_range):
                        continue
                    if scan_range < msg.range_min or scan_range > msg.range_max:
                        continue

                    angle = msg.angle_min + index * msg.angle_increment
                    point_in_laser = (
                        scan_range * math.cos(angle),
                        scan_range * math.sin(angle),
                        0.0,
                    )
                    point_in_base = self.transform_point(rotation, translation, point_in_laser)

                    if self.is_inside_filter_box(point_in_base[0], point_in_base[1]):
                        filtered_scan.ranges[index] = float("inf")

        output_scan = filtered_scan
        if self.fixed_resolution_enabled:
            output_scan = self.resample_scan_to_fixed_resolution(filtered_scan)

        self.publisher.publish(output_scan)
        self.publish_filter_markers()

    def clone_scan(self, msg: LaserScan) -> LaserScan:
        cloned_scan = LaserScan()
        cloned_scan.header = msg.header
        cloned_scan.angle_min = msg.angle_min
        cloned_scan.angle_max = msg.angle_max
        cloned_scan.angle_increment = msg.angle_increment
        cloned_scan.time_increment = msg.time_increment
        cloned_scan.scan_time = msg.scan_time
        cloned_scan.range_min = msg.range_min
        cloned_scan.range_max = msg.range_max
        cloned_scan.ranges = list(msg.ranges)
        cloned_scan.intensities = list(msg.intensities)
        return cloned_scan

    def resample_scan_to_fixed_resolution(self, msg: LaserScan) -> LaserScan:
        if self.fixed_resolution_bins < 2:
            self.warn_throttled(
                "fixed_resolution_bins must be at least 2; publishing the filtered scan unchanged.",
                key="invalid_fixed_resolution_bins",
            )
            return msg

        if not self.lock_fixed_resolution_grid(msg):
            return msg

        angle_min = self.fixed_resolution_angle_min
        angle_max = self.fixed_resolution_angle_max
        angle_increment = self.fixed_resolution_angle_increment
        if angle_min is None or angle_max is None or angle_increment is None:
            return msg

        if not self.fixed_resolution_use_full_circle and (
            not math.isclose(msg.angle_min, angle_min, rel_tol=0.0, abs_tol=1e-6)
            or not math.isclose(msg.angle_max, angle_max, rel_tol=0.0, abs_tol=1e-6)
        ):
            self.warn_throttled(
                "LaserScan angle range changed after fixed-resolution grid was locked; "
                "continuing to publish on the original fixed grid.",
                key="fixed_resolution_angle_range_changed",
            )

        output_scan = LaserScan()
        output_scan.header = msg.header
        output_scan.angle_min = angle_min
        output_scan.angle_max = angle_max
        output_scan.angle_increment = angle_increment
        output_scan.time_increment = self.compute_fixed_resolution_time_increment(msg)
        output_scan.scan_time = msg.scan_time
        output_scan.range_min = msg.range_min
        output_scan.range_max = msg.range_max
        output_scan.ranges = [float("inf")] * self.fixed_resolution_bins
        output_scan.intensities = [0.0] * self.fixed_resolution_bins

        best_angle_errors = [float("inf")] * self.fixed_resolution_bins
        best_ranges = [float("inf")] * self.fixed_resolution_bins
        input_intensities = list(msg.intensities)
        intensity_count = len(input_intensities)
        tie_tolerance = 1e-12

        for index, scan_range in enumerate(msg.ranges):
            if not math.isfinite(scan_range):
                continue
            if scan_range < msg.range_min or scan_range > msg.range_max:
                continue

            angle = msg.angle_min + index * msg.angle_increment
            mapped_index = int(round((angle - angle_min) / angle_increment))
            if mapped_index < 0 or mapped_index >= self.fixed_resolution_bins:
                continue

            mapped_angle = angle_min + mapped_index * angle_increment
            angle_error = abs(angle - mapped_angle)
            should_replace = angle_error + tie_tolerance < best_angle_errors[mapped_index]
            should_replace = should_replace or (
                math.isclose(angle_error, best_angle_errors[mapped_index], rel_tol=0.0, abs_tol=tie_tolerance)
                and scan_range < best_ranges[mapped_index]
            )
            if not should_replace:
                continue

            output_scan.ranges[mapped_index] = scan_range
            output_scan.intensities[mapped_index] = (
                input_intensities[index] if index < intensity_count else 0.0
            )
            best_angle_errors[mapped_index] = angle_error
            best_ranges[mapped_index] = scan_range

        return output_scan

    def lock_fixed_resolution_grid(self, msg: LaserScan) -> bool:
        if (
            self.fixed_resolution_angle_min is not None
            and self.fixed_resolution_angle_max is not None
            and self.fixed_resolution_angle_increment is not None
        ):
            return True

        if self.fixed_resolution_use_full_circle:
            angle_min = self.fixed_resolution_angle_min_config
            angle_increment = (2.0 * math.pi) / float(self.fixed_resolution_bins)
            angle_max = angle_min + angle_increment * float(self.fixed_resolution_bins - 1)
        else:
            angle_min = msg.angle_min
            angle_increment = (msg.angle_max - msg.angle_min) / float(self.fixed_resolution_bins - 1)
            angle_max = msg.angle_max

        if not math.isfinite(angle_increment) or abs(angle_increment) <= 1e-12:
            self.warn_throttled(
                "Unable to lock the fixed-resolution grid because the incoming scan angle range is invalid; "
                "publishing the filtered scan unchanged.",
                key="invalid_fixed_resolution_grid",
            )
            return False

        self.fixed_resolution_angle_min = angle_min
        self.fixed_resolution_angle_max = angle_max
        self.fixed_resolution_angle_increment = angle_increment
        self.get_logger().info(
            f"Locked fixed-resolution scan grid to angle_min={self.fixed_resolution_angle_min:.6f}, "
            f"angle_max={self.fixed_resolution_angle_max:.6f}, bins={self.fixed_resolution_bins}, "
            f"use_full_circle={self.fixed_resolution_use_full_circle}"
        )
        return True

    def compute_fixed_resolution_time_increment(self, msg: LaserScan) -> float:
        if msg.scan_time > 0.0:
            return msg.scan_time / float(self.fixed_resolution_bins)

        if msg.time_increment > 0.0 and len(msg.ranges) > 0:
            return msg.time_increment * float(len(msg.ranges)) / float(self.fixed_resolution_bins)

        return 0.0

    def lookup_transform(self, scan_frame: str):
        try:
            return self.tf_buffer.lookup_transform(
                self.base_frame,
                scan_frame,
                rclpy.time.Time(),
                timeout=self.transform_timeout,
            )
        except TransformException as exc:
            self.warn_throttled(
                f"Unable to transform {scan_frame} -> {self.base_frame}; bypassing self filter. Details: {exc}"
            )
            return None

    def transform_point(
        self,
        rotation: Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]],
        translation: Tuple[float, float, float],
        point: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        return (
            rotation[0][0] * point[0] + rotation[0][1] * point[1] + rotation[0][2] * point[2] + translation[0],
            rotation[1][0] * point[0] + rotation[1][1] * point[1] + rotation[1][2] * point[2] + translation[1],
            rotation[2][0] * point[0] + rotation[2][1] * point[1] + rotation[2][2] * point[2] + translation[2],
        )

    def is_inside_filter_box(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def on_set_parameters(self, parameters: List) -> SetParametersResult:
        for parameter in parameters:
            if parameter.name == "fixed_resolution_bins" and int(parameter.value) < 2:
                return SetParametersResult(
                    successful=False,
                    reason="fixed_resolution_bins must be at least 2",
                )

        for parameter in parameters:
            if parameter.name == "base_frame":
                self.base_frame = parameter.value
            elif parameter.name == "x_min":
                self.x_min = float(parameter.value)
            elif parameter.name == "x_max":
                self.x_max = float(parameter.value)
            elif parameter.name == "y_min":
                self.y_min = float(parameter.value)
            elif parameter.name == "y_max":
                self.y_max = float(parameter.value)
            elif parameter.name == "transform_timeout_sec":
                self.transform_timeout = Duration(seconds=float(parameter.value))
            elif parameter.name == "marker_z":
                self.marker_z = float(parameter.value)
            elif parameter.name == "marker_line_width":
                self.marker_line_width = float(parameter.value)
            elif parameter.name == "marker_alpha":
                self.marker_alpha = float(parameter.value)
            elif parameter.name == "fixed_resolution_enabled":
                self.fixed_resolution_enabled = bool(parameter.value)
                self.reset_fixed_resolution_grid()
            elif parameter.name == "fixed_resolution_bins":
                self.fixed_resolution_bins = int(parameter.value)
                self.reset_fixed_resolution_grid()
            elif parameter.name == "fixed_resolution_angle_min":
                self.fixed_resolution_angle_min_config = float(parameter.value)
                self.reset_fixed_resolution_grid()
            elif parameter.name == "fixed_resolution_use_full_circle":
                self.fixed_resolution_use_full_circle = bool(parameter.value)
                self.reset_fixed_resolution_grid()

        self.publish_filter_markers()
        return SetParametersResult(successful=True)

    def reset_fixed_resolution_grid(self) -> None:
        self.fixed_resolution_angle_min = None
        self.fixed_resolution_angle_max = None
        self.fixed_resolution_angle_increment = None

    def publish_filter_markers(self) -> None:
        now = self.get_clock().now().to_msg()
        marker_array = MarkerArray()

        fill_marker = Marker()
        fill_marker.header.frame_id = self.base_frame
        fill_marker.header.stamp = now
        fill_marker.ns = "scan_self_filter"
        fill_marker.id = 0
        fill_marker.type = Marker.CUBE
        fill_marker.action = Marker.ADD
        fill_marker.pose.orientation.w = 1.0
        fill_marker.pose.position.x = (self.x_min + self.x_max) / 2.0
        fill_marker.pose.position.y = (self.y_min + self.y_max) / 2.0
        fill_marker.pose.position.z = self.marker_z / 2.0
        fill_marker.scale.x = max(self.x_max - self.x_min, 1e-6)
        fill_marker.scale.y = max(self.y_max - self.y_min, 1e-6)
        fill_marker.scale.z = max(self.marker_z, 1e-3)
        fill_marker.color.r = 1.0
        fill_marker.color.g = 0.2
        fill_marker.color.b = 0.2
        fill_marker.color.a = min(max(self.marker_alpha, 0.0), 1.0)

        outline_marker = Marker()
        outline_marker.header.frame_id = self.base_frame
        outline_marker.header.stamp = now
        outline_marker.ns = "scan_self_filter"
        outline_marker.id = 1
        outline_marker.type = Marker.LINE_STRIP
        outline_marker.action = Marker.ADD
        outline_marker.pose.orientation.w = 1.0
        outline_marker.scale.x = max(self.marker_line_width, 1e-4)
        outline_marker.color.r = 1.0
        outline_marker.color.g = 0.95
        outline_marker.color.b = 0.2
        outline_marker.color.a = 1.0
        outline_marker.points = [
            self.make_point(self.x_min, self.y_min),
            self.make_point(self.x_max, self.y_min),
            self.make_point(self.x_max, self.y_max),
            self.make_point(self.x_min, self.y_max),
            self.make_point(self.x_min, self.y_min),
        ]

        marker_array.markers.append(fill_marker)
        marker_array.markers.append(outline_marker)
        self.marker_publisher.publish(marker_array)

    def make_point(self, x: float, y: float) -> Point:
        point = Point()
        point.x = x
        point.y = y
        point.z = self.marker_z
        return point

    def warn_throttled(self, message: str, throttle_sec: float = 5.0, key: Optional[str] = None) -> None:
        now_ns = self.get_clock().now().nanoseconds
        warning_key = key if key is not None else message
        last_warning_time_ns = self._last_warning_time_ns_by_key.get(warning_key, 0)
        if now_ns - last_warning_time_ns >= int(throttle_sec * 1e9):
            self.get_logger().warning(message)
            self._last_warning_time_ns_by_key[warning_key] = now_ns


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = ScanSelfFilter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
