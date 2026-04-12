#!/usr/bin/env bash
set -euo pipefail

IFACE="${IFACE:-wlan0}"
CONNECTION="${CONNECTION:-fcyy6}"
LOOKBACK="${LOOKBACK:-3 min ago}"
LOG_TAG="${LOG_TAG:-wifi-watchdog}"
RECOVERY_WAIT="${RECOVERY_WAIT:-4}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-180}"
STATE_DIR="${STATE_DIR:-/run/wifi-watchdog}"
LAST_RECOVERY_FILE="$STATE_DIR/last_recovery"

log() {
  logger -t "$LOG_TAG" "$*"
  printf '%s: %s\n' "$LOG_TAG" "$*"
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

if ! have_cmd nmcli || ! have_cmd journalctl || ! have_cmd modprobe; then
  log "required commands are missing; aborting"
  exit 1
fi

recent_kernel_wifi_errors="$(
  journalctl -k --since "$LOOKBACK" --no-pager 2>/dev/null \
    | grep -Ei 'brcmf_sdio_bus_rxctl: resumed on timeout|brcmf_proto_bcdc_msg failed w/status -110|BRCMF_C_GET_ASSOCLIST failed, err=-110' \
    || true
)"

interface_present() {
  ip link show "$IFACE" >/dev/null 2>&1
}

wifi_blocked() {
  rfkill list "$IFACE" 2>/dev/null | grep -qi 'Soft blocked: yes\|Hard blocked: yes'
}

device_state() {
  nmcli -t -f GENERAL.STATE device show "$IFACE" 2>/dev/null | awk -F: '{print $2}'
}

mark_recovery() {
  mkdir -p "$STATE_DIR"
  date +%s >"$LAST_RECOVERY_FILE"
}

in_cooldown() {
  [[ -f "$LAST_RECOVERY_FILE" ]] || return 1
  local last now
  last="$(cat "$LAST_RECOVERY_FILE" 2>/dev/null || echo 0)"
  now="$(date +%s)"
  [[ $((now - last)) -lt "$COOLDOWN_SECONDS" ]]
}

recover_driver() {
  log "reloading brcmfmac stack on $IFACE"
  mark_recovery
  nmcli device disconnect "$IFACE" >/dev/null 2>&1 || true
  modprobe -r brcmfmac_wcc brcmfmac brcmutil >/dev/null 2>&1 || true
  sleep 2
  modprobe brcmfmac
  sleep "$RECOVERY_WAIT"
  if interface_present; then
    nmcli connection up "$CONNECTION" ifname "$IFACE" >/dev/null 2>&1 || true
  else
    log "interface $IFACE still missing after reload"
    return 1
  fi
}

reconnect_only() {
  log "reconnecting $IFACE via NetworkManager"
  nmcli device disconnect "$IFACE" >/dev/null 2>&1 || true
  sleep 2
  nmcli connection up "$CONNECTION" ifname "$IFACE" >/dev/null 2>&1 || true
}

if wifi_blocked; then
  log "interface $IFACE is rfkill blocked; skip recovery"
  exit 0
fi

if ! interface_present; then
  log "interface $IFACE not present; attempting driver recovery"
  recover_driver || {
    sleep "$RECOVERY_WAIT"
    if ! interface_present; then
      log "second recovery attempt for missing interface $IFACE"
      recover_driver || true
    fi
  }
  exit 0
fi

nm_state="$(device_state)"

if [[ -n "$recent_kernel_wifi_errors" ]]; then
  if [[ "$nm_state" == "100 (connected)" ]]; then
    log "recent brcmfmac errors found, but $IFACE is connected; skipping recovery"
    exit 0
  fi
  if in_cooldown; then
    log "recent brcmfmac errors found, but cooldown is active; skipping recovery"
    exit 0
  fi
  log "detected recent brcmfmac timeout signature"
  recover_driver
  exit 0
fi

if [[ "$nm_state" != "100 (connected)" ]]; then
  log "device state is '$nm_state'"
  if in_cooldown; then
    log "cooldown is active; skipping reconnect cycle"
    exit 0
  fi
  reconnect_only
  sleep 8
  nm_state="$(device_state)"
  if [[ "$nm_state" != "100 (connected)" ]]; then
    log "simple reconnect did not restore connectivity"
    recover_driver
  fi
  exit 0
fi

log "wifi healthy on $IFACE"
