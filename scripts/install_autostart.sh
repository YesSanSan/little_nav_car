#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SYSTEMD_DIR="/etc/systemd/system"
RUN_USER="${SUDO_USER:-${USER}}"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Please run as root: sudo ./scripts/install_autostart.sh" >&2
  exit 1
fi

install_unit() {
  local template_name="$1"
  local service_name="$2"
  local template_path="${WORKSPACE_DIR}/systemd/${template_name}"
  local service_path="${SYSTEMD_DIR}/${service_name}"

  sed \
    -e "s|@WORKSPACE_DIR@|${WORKSPACE_DIR}|g" \
    -e "s|@RUN_USER@|${RUN_USER}|g" \
    "${template_path}" > "${service_path}"
}

install_unit "little-nav-car-serial.service.template" "little-nav-car-serial.service"
install_unit "little-nav-car-web.service.template" "little-nav-car-web.service"

systemctl daemon-reload
systemctl enable --now little-nav-car-serial.service
systemctl enable --now little-nav-car-web.service

echo "Installed and started:"
echo "  little-nav-car-serial.service"
echo "  little-nav-car-web.service"
echo
echo "Check status with:"
echo "  systemctl status little-nav-car-serial.service"
echo "  systemctl status little-nav-car-web.service"
