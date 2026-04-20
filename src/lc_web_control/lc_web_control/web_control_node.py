import json
import math
import subprocess
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PointStamped, PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter, parameter_value_to_python
from rclpy.parameter_client import AsyncParameterClient
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


def yaw_from_quaternion(quaternion: Any) -> float:
    siny_cosp = 2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
    cosy_cosp = 1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float) -> Dict[str, float]:
    half_yaw = yaw * 0.5
    return {
        "x": 0.0,
        "y": 0.0,
        "z": math.sin(half_yaw),
        "w": math.cos(half_yaw),
    }


class ControlHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, handler_class, app):
        super().__init__(server_address, handler_class)
        self.app = app


def make_request_handler():
    class RequestHandler(BaseHTTPRequestHandler):
        server_version = "LCWebControl/0.1"

        @property
        def app(self) -> "WebControlNode":
            return self.server.app

        def log_message(self, format_: str, *args) -> None:
            return

        def do_GET(self) -> None:
            route = urlparse(self.path).path
            try:
                if route in ("/", "/index.html"):
                    self._serve_static("index.html", "text/html; charset=utf-8")
                    return
                if route == "/styles.css":
                    self._serve_static("styles.css", "text/css; charset=utf-8")
                    return
                if route == "/app.js":
                    self._serve_static("app.js", "application/javascript; charset=utf-8")
                    return
                if route == "/api/status":
                    self._send_json(HTTPStatus.OK, self.app.build_status_payload())
                    return
                if route == "/api/map":
                    self._send_json(HTTPStatus.OK, self.app.build_map_payload())
                    return

                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "message": f"Unknown route: {route}"})
            except Exception as exc:  # pragma: no cover - defensive HTTP boundary
                self.app.get_logger().error(f"GET {route} failed: {exc}")
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "message": str(exc)})

        def do_POST(self) -> None:
            route = urlparse(self.path).path
            try:
                payload = self._read_json_body()

                if route == "/api/control/start_slam":
                    self._send_json(HTTPStatus.OK, self.app.start_slam())
                    return
                if route == "/api/control/stop_slam":
                    self._send_json(HTTPStatus.OK, self.app.stop_slam())
                    return
                if route == "/api/control/start_tracking":
                    self._send_json(HTTPStatus.OK, self.app.set_tracking_enabled(True))
                    return
                if route == "/api/control/stop_tracking":
                    self._send_json(HTTPStatus.OK, self.app.set_tracking_enabled(False))
                    return
                if route == "/api/control/return_home":
                    self._send_json(HTTPStatus.OK, self.app.return_home())
                    return
                if route == "/api/return_point":
                    self._send_json(
                        HTTPStatus.OK,
                        self.app.set_return_pose_from_map_click(
                            float(payload["x"]),
                            float(payload["y"]),
                            payload.get("yaw"),
                            payload.get("frame_id"),
                        ),
                    )
                    return
                if route == "/api/return_point/current":
                    self._send_json(HTTPStatus.OK, self.app.set_return_pose_to_current())
                    return
                if route == "/api/return_point/startup":
                    self._send_json(HTTPStatus.OK, self.app.set_return_pose_to_startup())
                    return

                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "message": f"Unknown route: {route}"})
            except KeyError as exc:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"ok": False, "message": f"Missing request field: {exc}"},
                )
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "message": str(exc)})
            except RuntimeError as exc:
                self._send_json(HTTPStatus.CONFLICT, {"ok": False, "message": str(exc)})
            except Exception as exc:  # pragma: no cover - defensive HTTP boundary
                self.app.get_logger().error(f"POST {route} failed: {exc}")
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "message": str(exc)})

        def _read_json_body(self) -> Dict[str, Any]:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                return {}

            raw_body = self.rfile.read(content_length)
            if not raw_body:
                return {}

            try:
                return json.loads(raw_body.decode("utf-8"))
            except json.JSONDecodeError as exc:  # pragma: no cover - input validation
                raise ValueError(f"Invalid JSON payload: {exc}") from exc

        def _serve_static(self, filename: str, content_type: str) -> None:
            file_path = self.app.static_dir / filename
            if not file_path.is_file():
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "message": f"Missing asset: {filename}"})
                return

            body = file_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: HTTPStatus, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return RequestHandler


class WebControlNode(Node):
    def __init__(self) -> None:
        super().__init__("lc_web_control")

        default_workspace = Path.cwd()
        default_state_file = Path.home() / ".little_nav_car" / "web_control" / "return_pose.json"

        self.bind_host = self.declare_parameter("bind_host", "0.0.0.0").value
        self.port = int(self.declare_parameter("port", 8080).value)
        self.workspace_dir = Path(
            self.declare_parameter("workspace_dir", str(default_workspace)).value
        ).expanduser()
        self.global_frame = self.declare_parameter("global_frame", "map").value
        self.robot_frame = self.declare_parameter("robot_frame", "base_link").value
        self.map_topic = self.declare_parameter("map_topic", "/map").value
        self.vision_node_name = self.declare_parameter("vision_node_name", "/lc_vision").value
        self.vision_target_topic = self.declare_parameter(
            "vision_target_topic", "/lc_vision/selected_target_map_point"
        ).value
        self.vision_goal_topic = self.declare_parameter(
            "vision_goal_topic", "/lc_vision/selected_goal_pose"
        ).value
        self.command_timeout_sec = float(self.declare_parameter("command_timeout_sec", 20.0).value)
        self.state_file = Path(
            self.declare_parameter("state_file", str(default_state_file)).value
        ).expanduser()

        self.static_dir = Path(get_package_share_directory("lc_web_control")) / "static"
        self.state_lock = threading.Lock()
        self._vision_status_request_pending = False
        self._latest_map: Optional[OccupancyGrid] = None
        self._latest_target_point: Optional[PointStamped] = None
        self._latest_goal_pose: Optional[PoseStamped] = None
        self._startup_pose: Optional[Dict[str, Any]] = None
        self._return_pose: Optional[Dict[str, Any]] = None
        self._vision_status: Dict[str, Any] = {
            "available": False,
            "desired_tracking_enabled": False,
            "effective_tracking_enabled": False,
            "last_applied_tracking_enabled": False,
            "pending_apply": False,
            "last_error": "lc_vision parameter service unavailable",
        }

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self.navigate_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.vision_parameter_client = AsyncParameterClient(self, self.vision_node_name)

        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self._on_map, 10)
        self.target_sub = self.create_subscription(
            PointStamped, self.vision_target_topic, self._on_target_point, 10
        )
        self.goal_sub = self.create_subscription(PoseStamped, self.vision_goal_topic, self._on_goal_pose, 10)
        self.create_timer(1.0, self._refresh_vision_status)

        self._load_return_pose()
        self._start_http_server()

        self.get_logger().info(
            f"Web control ready on http://{self.bind_host}:{self.port} "
            f"(workspace: {self.workspace_dir})"
        )

    def destroy_node(self) -> bool:
        self._shutdown_http_server()
        return super().destroy_node()

    def _start_http_server(self) -> None:
        handler = make_request_handler()
        self.http_server = ControlHTTPServer((self.bind_host, self.port), handler, self)
        self.http_thread = threading.Thread(
            target=self.http_server.serve_forever,
            name="lc_web_control_http",
            daemon=True,
        )
        self.http_thread.start()

    def _shutdown_http_server(self) -> None:
        http_server = getattr(self, "http_server", None)
        if http_server is None:
            return

        http_server.shutdown()
        http_server.server_close()
        self.http_server = None

    def _on_map(self, msg: OccupancyGrid) -> None:
        with self.state_lock:
            self._latest_map = msg

    def _on_target_point(self, msg: PointStamped) -> None:
        with self.state_lock:
            self._latest_target_point = msg

    def _on_goal_pose(self, msg: PoseStamped) -> None:
        with self.state_lock:
            self._latest_goal_pose = msg

    def _refresh_vision_status(self) -> None:
        if self._vision_status_request_pending:
            return

        if not self.vision_parameter_client.wait_for_services(timeout_sec=0.0):
            with self.state_lock:
                self._vision_status = {
                    "available": False,
                    "desired_tracking_enabled": self._vision_status["desired_tracking_enabled"],
                    "effective_tracking_enabled": False,
                    "last_applied_tracking_enabled": self._vision_status["last_applied_tracking_enabled"],
                    "pending_apply": self._vision_status["pending_apply"],
                    "last_error": "lc_vision parameter service unavailable",
                }
            return

        self._vision_status_request_pending = True
        future = self.vision_parameter_client.get_parameters(["tracking.enable", "goal.enable"])
        future.add_done_callback(self._handle_vision_status_response)

    def _handle_vision_status_response(self, future) -> None:
        try:
            response = future.result()
            values = getattr(response, "values", [])
            tracking_enabled = bool(parameter_value_to_python(values[0])) if len(values) > 0 else False
            goal_enabled = bool(parameter_value_to_python(values[1])) if len(values) > 1 else False
            effective_enabled = tracking_enabled and goal_enabled
            status = self._update_vision_status_after_read(effective_enabled)
        except Exception as exc:
            with self.state_lock:
                status = {
                    "available": False,
                    "desired_tracking_enabled": self._vision_status["desired_tracking_enabled"],
                    "effective_tracking_enabled": False,
                    "last_applied_tracking_enabled": self._vision_status["last_applied_tracking_enabled"],
                    "pending_apply": self._vision_status["pending_apply"],
                    "last_error": str(exc),
                }
        finally:
            self._vision_status_request_pending = False

        with self.state_lock:
            self._vision_status = status
        self._save_return_pose()

    def _default_state_payload(self) -> Dict[str, Any]:
        return {
            "startup_pose": None,
            "return_pose": None,
            "vision": {
                "desired_tracking_enabled": False,
                "last_applied_tracking_enabled": False,
                "apply_pending": False,
            },
        }

    def _current_state_payload(self) -> Dict[str, Any]:
        with self.state_lock:
            return {
                "startup_pose": dict(self._startup_pose) if self._startup_pose is not None else None,
                "return_pose": dict(self._return_pose) if self._return_pose is not None else None,
                "vision": {
                    "desired_tracking_enabled": bool(self._vision_status["desired_tracking_enabled"]),
                    "last_applied_tracking_enabled": bool(
                        self._vision_status["last_applied_tracking_enabled"]
                    ),
                    "apply_pending": bool(self._vision_status["pending_apply"]),
                },
            }

    def _parse_saved_state(
        self, payload: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
        state_payload = self._default_state_payload()

        if {"frame_id", "x", "y", "yaw"}.issubset(payload):
            state_payload["startup_pose"] = {
                "frame_id": str(payload["frame_id"]),
                "x": float(payload["x"]),
                "y": float(payload["y"]),
                "yaw": float(payload["yaw"]),
            }
            state_payload["return_pose"] = dict(state_payload["startup_pose"])
            return state_payload["startup_pose"], state_payload["return_pose"], state_payload["vision"]

        startup_pose = payload.get("startup_pose")
        if isinstance(startup_pose, dict) and {"frame_id", "x", "y", "yaw"}.issubset(startup_pose):
            state_payload["startup_pose"] = {
                "frame_id": str(startup_pose["frame_id"]),
                "x": float(startup_pose["x"]),
                "y": float(startup_pose["y"]),
                "yaw": float(startup_pose["yaw"]),
            }

        return_pose = payload.get("return_pose")
        if isinstance(return_pose, dict) and {"frame_id", "x", "y", "yaw"}.issubset(return_pose):
            state_payload["return_pose"] = {
                "frame_id": str(return_pose["frame_id"]),
                "x": float(return_pose["x"]),
                "y": float(return_pose["y"]),
                "yaw": float(return_pose["yaw"]),
            }

        vision = payload.get("vision")
        if isinstance(vision, dict):
            state_payload["vision"] = {
                "desired_tracking_enabled": bool(vision.get("desired_tracking_enabled", False)),
                "last_applied_tracking_enabled": bool(vision.get("last_applied_tracking_enabled", False)),
                "apply_pending": bool(vision.get("apply_pending", False)),
            }

        if state_payload["return_pose"] is None and state_payload["startup_pose"] is not None:
            state_payload["return_pose"] = dict(state_payload["startup_pose"])

        return state_payload["startup_pose"], state_payload["return_pose"], state_payload["vision"]

    def _update_vision_status_after_read(self, effective_enabled: bool) -> Dict[str, Any]:
        with self.state_lock:
            desired_enabled = bool(self._vision_status["desired_tracking_enabled"])
            last_applied_enabled = bool(self._vision_status["last_applied_tracking_enabled"])

        if effective_enabled != desired_enabled:
            try:
                self._apply_tracking_state(desired_enabled)
                message = (
                    "视觉追踪期望状态已自动同步到 lc_vision"
                    if desired_enabled
                    else "视觉追踪关闭状态已自动同步到 lc_vision"
                )
                return {
                    "available": True,
                    "desired_tracking_enabled": desired_enabled,
                    "effective_tracking_enabled": desired_enabled,
                    "last_applied_tracking_enabled": desired_enabled,
                    "pending_apply": False,
                    "last_error": message,
                }
            except RuntimeError as exc:
                return {
                    "available": True,
                    "desired_tracking_enabled": desired_enabled,
                    "effective_tracking_enabled": effective_enabled,
                    "last_applied_tracking_enabled": last_applied_enabled,
                    "pending_apply": True,
                    "last_error": str(exc),
                }

        return {
            "available": True,
            "desired_tracking_enabled": desired_enabled,
            "effective_tracking_enabled": effective_enabled,
            "last_applied_tracking_enabled": effective_enabled,
            "pending_apply": False,
            "last_error": None,
        }

    def _load_return_pose(self) -> None:
        if not self.state_file.is_file():
            return

        try:
            payload = json.loads(self.state_file.read_text(encoding="utf-8"))
            startup_pose, return_pose, vision_state = self._parse_saved_state(payload)
            with self.state_lock:
                self._startup_pose = startup_pose
                self._return_pose = return_pose
                self._vision_status.update(
                    {
                        "desired_tracking_enabled": vision_state["desired_tracking_enabled"],
                        "last_applied_tracking_enabled": vision_state["last_applied_tracking_enabled"],
                        "pending_apply": vision_state["apply_pending"],
                    }
                )
        except Exception as exc:
            self.get_logger().warn(f"Failed to load saved state from {self.state_file}: {exc}")

    def _save_return_pose(self) -> None:
        payload = self._current_state_payload()

        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _apply_tracking_state(self, enabled: bool) -> None:
        parameters = [
            Parameter("tracking.enable", Parameter.Type.BOOL, enabled),
            Parameter("goal.enable", Parameter.Type.BOOL, enabled),
        ]
        future = self.vision_parameter_client.set_parameters(parameters)
        response = self._wait_for_future(
            future,
            5.0,
            "Timed out while waiting for lc_vision parameter update",
        )
        results = getattr(response, "results", response)
        failures = [item.reason for item in results if not item.successful]
        if failures:
            raise RuntimeError("; ".join(failures))

    def _set_desired_tracking_enabled(self, enabled: bool) -> Dict[str, Any]:
        if not self.vision_parameter_client.wait_for_services(timeout_sec=1.0):
            with self.state_lock:
                self._vision_status = {
                    "available": False,
                    "desired_tracking_enabled": enabled,
                    "effective_tracking_enabled": False,
                    "last_applied_tracking_enabled": self._vision_status["last_applied_tracking_enabled"],
                    "pending_apply": True,
                    "last_error": "lc_vision 尚未启动，已记录期望状态，待参数服务可用后自动应用",
                }
            self._save_return_pose()
            return {
                "available": True,
                "ok": True,
                "tracking_enabled": enabled,
                "pending_apply": True,
                "message": "lc_vision 尚未启动，已记录视觉追踪开关，稍后会自动应用",
            }

        self._apply_tracking_state(enabled)
        with self.state_lock:
            self._vision_status = {
                "available": True,
                "desired_tracking_enabled": enabled,
                "effective_tracking_enabled": enabled,
                "last_applied_tracking_enabled": enabled,
                "pending_apply": False,
                "last_error": None,
            }
        self._save_return_pose()
        return {
            "ok": True,
            "tracking_enabled": enabled,
            "pending_apply": False,
            "message": "视觉追踪已开启" if enabled else "视觉追踪已停止",
        }

    def _run_workspace_script(self, script_name: str, *args: str) -> Dict[str, Any]:
        script_path = self.workspace_dir / script_name
        if not script_path.is_file():
            raise RuntimeError(f"Missing script: {script_path}")

        completed = subprocess.run(
            ["bash", str(script_path), *args],
            cwd=self.workspace_dir,
            capture_output=True,
            text=True,
            timeout=self.command_timeout_sec,
            check=False,
        )
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        detail = "\n".join(part for part in [stdout, stderr] if part).strip()
        if completed.returncode != 0:
            raise RuntimeError(detail or f"{script_name} failed with exit code {completed.returncode}")

        return {
            "ok": True,
            "message": detail or f"{script_name} {' '.join(args)} executed successfully",
        }

    def _is_stack_running(self, session_name: str) -> bool:
        try:
            result = subprocess.run(
                ["tmux", "has-session", "-t", session_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"tmux is not available on this device: {exc}") from exc

        return result.returncode == 0

    def _lookup_robot_pose(self, target_frame: str) -> Optional[Dict[str, Any]]:
        try:
            transform = self.tf_buffer.lookup_transform(target_frame, self.robot_frame, Time())
        except TransformException:
            return None

        rotation = transform.transform.rotation
        return {
            "frame_id": target_frame,
            "x": float(transform.transform.translation.x),
            "y": float(transform.transform.translation.y),
            "yaw": float(yaw_from_quaternion(rotation)),
        }

    def _point_to_dict(self, point_msg: PointStamped) -> Dict[str, Any]:
        return {
            "frame_id": point_msg.header.frame_id,
            "x": float(point_msg.point.x),
            "y": float(point_msg.point.y),
        }

    def _pose_to_dict(self, pose_msg: PoseStamped) -> Dict[str, Any]:
        return {
            "frame_id": pose_msg.header.frame_id,
            "x": float(pose_msg.pose.position.x),
            "y": float(pose_msg.pose.position.y),
            "yaw": float(yaw_from_quaternion(pose_msg.pose.orientation)),
        }

    def _build_pose_stamped(self, pose_dict: Dict[str, Any]) -> PoseStamped:
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = pose_dict["frame_id"]
        pose.pose.position.x = float(pose_dict["x"])
        pose.pose.position.y = float(pose_dict["y"])
        pose.pose.position.z = 0.0
        orientation = quaternion_from_yaw(float(pose_dict["yaw"]))
        pose.pose.orientation.x = orientation["x"]
        pose.pose.orientation.y = orientation["y"]
        pose.pose.orientation.z = orientation["z"]
        pose.pose.orientation.w = orientation["w"]
        return pose

    def _wait_for_future(self, future, timeout_sec: float, timeout_message: str):
        event = threading.Event()
        result_holder: Dict[str, Any] = {}

        def _done(done_future):
            try:
                result_holder["result"] = done_future.result()
            except Exception as exc:  # pragma: no cover - defensive async boundary
                result_holder["error"] = exc
            finally:
                event.set()

        future.add_done_callback(_done)
        if not event.wait(timeout_sec):
            raise RuntimeError(timeout_message)

        if "error" in result_holder:
            raise RuntimeError(str(result_holder["error"]))

        return result_holder.get("result")

    def _ensure_return_pose_initialized(self, frame_id: str) -> Optional[Dict[str, Any]]:
        with self.state_lock:
            if self._startup_pose is not None and self._return_pose is not None:
                return dict(self._return_pose)

        current_pose = self._lookup_robot_pose(frame_id)
        if current_pose is None:
            return None

        with self.state_lock:
            if self._startup_pose is None:
                self._startup_pose = {
                    "frame_id": current_pose["frame_id"],
                    "x": float(current_pose["x"]),
                    "y": float(current_pose["y"]),
                    "yaw": float(current_pose["yaw"]),
                }
            if self._return_pose is None:
                self._return_pose = {
                    "frame_id": self._startup_pose["frame_id"],
                    "x": float(self._startup_pose["x"]),
                    "y": float(self._startup_pose["y"]),
                    "yaw": float(self._startup_pose["yaw"]),
                }
                initialized_pose = dict(self._return_pose)
            else:
                initialized_pose = dict(self._return_pose)

        self._save_return_pose()
        return initialized_pose

    def build_status_payload(self) -> Dict[str, Any]:
        with self.state_lock:
            latest_map = self._latest_map
            target_point = self._point_to_dict(self._latest_target_point) if self._latest_target_point else None
            goal_pose = self._pose_to_dict(self._latest_goal_pose) if self._latest_goal_pose else None
            vision_status = dict(self._vision_status)

        map_frame = self.global_frame
        if latest_map is not None and latest_map.header.frame_id:
            map_frame = latest_map.header.frame_id
        return_pose = self._ensure_return_pose_initialized(map_frame)

        return {
            "ok": True,
            "bind_host": self.bind_host,
            "port": self.port,
            "workspace_dir": str(self.workspace_dir),
            "stacks": {
                "slam": self._is_stack_running("slam"),
            },
            "map": {
                "available": latest_map is not None,
                "frame_id": map_frame,
            },
            "robot_pose": self._lookup_robot_pose(map_frame),
            "return_pose": return_pose,
            "vision": {
                **vision_status,
                "target_point": target_point,
                "goal_pose": goal_pose,
            },
        }

    def build_map_payload(self) -> Dict[str, Any]:
        with self.state_lock:
            map_msg = self._latest_map
            target_point = self._point_to_dict(self._latest_target_point) if self._latest_target_point else None
            goal_pose = self._pose_to_dict(self._latest_goal_pose) if self._latest_goal_pose else None

        if map_msg is None:
            return {
                "ok": True,
                "available": False,
                "message": "Map topic has not been received yet",
            }

        map_frame = map_msg.header.frame_id or self.global_frame
        return_pose = self._ensure_return_pose_initialized(map_frame)
        origin = map_msg.info.origin
        return {
            "ok": True,
            "available": True,
            "frame_id": map_frame,
            "resolution": float(map_msg.info.resolution),
            "width": int(map_msg.info.width),
            "height": int(map_msg.info.height),
            "origin": {
                "x": float(origin.position.x),
                "y": float(origin.position.y),
                "yaw": float(yaw_from_quaternion(origin.orientation)),
            },
            "data": list(map_msg.data),
            "robot_pose": self._lookup_robot_pose(map_frame),
            "return_pose": return_pose if return_pose and return_pose["frame_id"] == map_frame else None,
            "vision_target": target_point if target_point and target_point["frame_id"] == map_frame else None,
            "vision_goal": goal_pose if goal_pose and goal_pose["frame_id"] == map_frame else None,
        }

    def start_slam(self) -> Dict[str, Any]:
        result = self._run_workspace_script("slam.sh", "--detach")
        result["mode"] = "slam"
        return result

    def stop_slam(self) -> Dict[str, Any]:
        result = self._run_workspace_script("slam.sh", "--stop")
        result["mode"] = "slam"
        return result

    def set_tracking_enabled(self, enabled: bool) -> Dict[str, Any]:
        return self._set_desired_tracking_enabled(enabled)

    def set_return_pose_from_map_click(
        self, x: float, y: float, yaw: Optional[float] = None, frame_id: Optional[str] = None
    ) -> Dict[str, Any]:
        pose_frame = frame_id or self.global_frame
        if yaw is None:
            current_pose = self._lookup_robot_pose(pose_frame)
            yaw = current_pose["yaw"] if current_pose is not None else 0.0

        with self.state_lock:
            self._return_pose = {
                "frame_id": pose_frame,
                "x": float(x),
                "y": float(y),
                "yaw": float(yaw),
            }
        self._save_return_pose()

        return {
            "ok": True,
            "message": "返航点已更新",
            "return_pose": self._return_pose,
        }

    def set_return_pose_to_current(self) -> Dict[str, Any]:
        current_pose = self._lookup_robot_pose(self.global_frame)
        if current_pose is None:
            raise RuntimeError("Current robot pose is unavailable, cannot mark return point")
        return self.set_return_pose_from_map_click(
            current_pose["x"],
            current_pose["y"],
            current_pose["yaw"],
            current_pose["frame_id"],
        )

    def set_return_pose_to_startup(self) -> Dict[str, Any]:
        startup_pose = self._ensure_return_pose_initialized(self.global_frame)
        if startup_pose is None:
            raise RuntimeError("Startup pose is unavailable, please wait for robot pose")

        with self.state_lock:
            if self._startup_pose is None:
                raise RuntimeError("Startup pose is unavailable")
            self._return_pose = dict(self._startup_pose)
            return_pose = dict(self._return_pose)
        self._save_return_pose()

        return {
            "ok": True,
            "message": "返航点已恢复为启动位置",
            "return_pose": return_pose,
        }

    def return_home(self) -> Dict[str, Any]:
        return_pose = self._ensure_return_pose_initialized(self.global_frame)
        if return_pose is None:
            raise RuntimeError("Return point is unavailable, please wait for robot pose")

        if not self.navigate_client.wait_for_server(timeout_sec=1.5):
            raise RuntimeError("navigate_to_pose is unavailable, please start navigation support first")

        goal = NavigateToPose.Goal()
        goal.pose = self._build_pose_stamped(return_pose)
        future = self.navigate_client.send_goal_async(goal)
        goal_handle = self._wait_for_future(
            future,
            5.0,
            "Timed out while waiting for return-home goal response",
        )
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("Return-home goal was rejected by Nav2")

        return {
            "ok": True,
            "message": "返航目标已发送",
            "return_pose": return_pose,
        }


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WebControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
