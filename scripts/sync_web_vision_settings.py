#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


SETTING_TO_PARAM = {
    "nav2_tracking_enabled": "tracking.nav_goal.enable",
    "stop_distance_m": "goal.standoff_distance_m",
    "lidar_protection_distance_m": "tracking.lidar.min_forward_protection_m",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision-yaml", required=True)
    parser.add_argument("--settings-file", required=True)
    return parser.parse_args()


def normalize_settings(payload):
    if not isinstance(payload, dict):
        raise ValueError("settings payload must be a JSON object")

    return {
        "nav2_tracking_enabled": bool(payload["nav2_tracking_enabled"]),
        "stop_distance_m": float(payload["stop_distance_m"]),
        "lidar_protection_distance_m": float(payload["lidar_protection_distance_m"]),
    }


def parse_yaml_values(vision_yaml_path):
    text = vision_yaml_path.read_text(encoding="utf-8")
    values = {}
    patterns = {
        "nav2_tracking_enabled": re.compile(r"^\s*tracking\.nav_goal\.enable:\s*(true|false)\s*$", re.MULTILINE),
        "stop_distance_m": re.compile(r"^\s*goal\.standoff_distance_m:\s*([-+]?\d+(?:\.\d+)?)\s*$", re.MULTILINE),
        "lidar_protection_distance_m": re.compile(
            r"^\s*tracking\.lidar\.min_forward_protection_m:\s*([-+]?\d+(?:\.\d+)?)\s*$",
            re.MULTILINE,
        ),
    }

    for key, pattern in patterns.items():
        match = pattern.search(text)
        if not match:
            raise ValueError(f"missing parameter in {vision_yaml_path}: {SETTING_TO_PARAM[key]}")
        if key == "nav2_tracking_enabled":
            values[key] = match.group(1) == "true"
        else:
            values[key] = float(match.group(1))

    return text, values


def apply_settings_to_yaml(text, settings):
    replacements = {
        "tracking.nav_goal.enable": "true" if settings["nav2_tracking_enabled"] else "false",
        "goal.standoff_distance_m": f"{settings['stop_distance_m']:.3f}",
        "tracking.lidar.min_forward_protection_m": f"{settings['lidar_protection_distance_m']:.3f}",
    }

    updated_text = text
    for param_name, value in replacements.items():
        pattern = re.compile(rf"^(\s*{re.escape(param_name)}:\s*).*$", re.MULTILINE)
        updated_text, count = pattern.subn(lambda match: f"{match.group(1)}{value}", updated_text, count=1)
        if count != 1:
            raise ValueError(f"failed to update parameter in vision.yaml: {param_name}")
    return updated_text


def main():
    args = parse_args()
    vision_yaml_path = Path(args.vision_yaml).expanduser().resolve()
    settings_file_path = Path(args.settings_file).expanduser().resolve()

    original_text, yaml_settings = parse_yaml_values(vision_yaml_path)

    if settings_file_path.is_file():
        file_settings = normalize_settings(json.loads(settings_file_path.read_text(encoding="utf-8")))
    else:
        file_settings = yaml_settings
        settings_file_path.parent.mkdir(parents=True, exist_ok=True)
        settings_file_path.write_text(
            json.dumps(file_settings, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    updated_text = apply_settings_to_yaml(original_text, file_settings)
    if updated_text != original_text:
        vision_yaml_path.write_text(updated_text, encoding="utf-8")

    if file_settings != yaml_settings or not settings_file_path.is_file():
        settings_file_path.parent.mkdir(parents=True, exist_ok=True)
        settings_file_path.write_text(
            json.dumps(file_settings, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
