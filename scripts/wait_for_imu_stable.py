#!/usr/bin/env python3

import argparse
import time

import rclpy
from sensor_msgs.msg import Imu


class ImuStartupMonitor:
    def __init__(self, stable_duration: float, max_gap: float):
        self.stable_duration = stable_duration
        self.max_gap = max_gap
        self.last_rx = None
        self.stable_start = None
        self.total_msgs = 0
        self.last_status_log = 0.0

    def callback(self, _msg: Imu):
        now = time.monotonic()
        self.total_msgs += 1
        if self.last_rx is None or (now - self.last_rx) > self.max_gap:
            self.stable_start = now
        self.last_rx = now

    def is_ready(self) -> bool:
        if self.last_rx is None or self.stable_start is None:
            return False
        now = time.monotonic()
        if (now - self.last_rx) > self.max_gap:
            self.stable_start = None
            return False
        return (now - self.stable_start) >= self.stable_duration

    def maybe_log(self, topic: str):
        now = time.monotonic()
        if now - self.last_status_log < 2.0:
            return
        self.last_status_log = now
        if self.last_rx is None:
            print(f"[localization] Waiting for IMU messages on {topic}...", flush=True)
            return

        gap = now - self.last_rx
        print(
            f"[localization] IMU seen ({self.total_msgs} msgs), latest gap {gap:.3f}s; "
            f"waiting for {self.stable_duration:.1f}s of continuous data...",
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Wait for a stable IMU stream before launching localization.")
    parser.add_argument("--topic", default="/base/imu0")
    parser.add_argument("--stable-duration", type=float, default=3.0)
    parser.add_argument("--max-gap", type=float, default=0.35)
    return parser.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    node = rclpy.create_node("imu_startup_monitor")
    monitor = ImuStartupMonitor(args.stable_duration, args.max_gap)
    node.create_subscription(Imu, args.topic, monitor.callback, 50)

    print(
        f"[localization] Waiting for stable IMU stream on {args.topic} "
        f"(continuous {args.stable_duration:.1f}s, max gap {args.max_gap:.2f}s)...",
        flush=True,
    )

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            monitor.maybe_log(args.topic)
            if monitor.is_ready():
                print("[localization] IMU stream is stable. Starting localization.", flush=True)
                break
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
