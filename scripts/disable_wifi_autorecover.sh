#!/usr/bin/env bash
set -euo pipefail

SERVICE="/etc/systemd/system/wifi-watchdog.service"
TIMER="/etc/systemd/system/wifi-watchdog.timer"
POWERSAVE_OFF="/etc/NetworkManager/conf.d/wifi-powersave-off.conf"

log() {
  printf '%s\n' "$*"
}

need_root() {
  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    log "Please run as root: sudo $0"
    exit 1
  fi
}

disable_watchdog() {
  if systemctl list-unit-files | grep -q '^wifi-watchdog.timer'; then
    systemctl disable --now wifi-watchdog.timer >/dev/null 2>&1 || true
  fi
  if systemctl list-unit-files | grep -q '^wifi-watchdog.service'; then
    systemctl stop wifi-watchdog.service >/dev/null 2>&1 || true
  fi
  rm -f "$SERVICE" "$TIMER"
  systemctl daemon-reload >/dev/null 2>&1 || true
}

restore_powersave_default() {
  if [[ -f "$POWERSAVE_OFF" ]]; then
    rm -f "$POWERSAVE_OFF"
    systemctl restart NetworkManager >/dev/null 2>&1 || true
  fi
}

need_root

log "Disabling wifi-watchdog auto-recovery and auto-reboot..."
disable_watchdog

log "Removing Wi-Fi power-save override (if present)..."
restore_powersave_default

log "Done. Auto-recovery and auto-reboot are disabled."
