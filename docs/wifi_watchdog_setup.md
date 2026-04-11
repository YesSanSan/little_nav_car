# Wi-Fi watchdog setup

This machine is currently using `wlan0` with the NetworkManager connection name `fcyy6`.
The watchdog should run every 15 seconds and keep trying recovery if `wlan0`
disappears after a failed `brcmfmac` reload.

## 1. Persistently disable Wi-Fi power saving

```bash
sudo mkdir -p /etc/NetworkManager/conf.d
printf "[connection]\nwifi.powersave = 2\n" | sudo tee /etc/NetworkManager/conf.d/wifi-powersave-off.conf
sudo systemctl restart NetworkManager
```

## 2. Install the watchdog

From the repository root:

```bash
sudo install -m 0755 scripts/wifi_watchdog.sh /usr/local/sbin/wifi_watchdog.sh
sudo tee /etc/systemd/system/wifi-watchdog.service >/dev/null <<'EOF'
[Unit]
Description=Recover Raspberry Pi Wi-Fi when brcmfmac stalls
After=NetworkManager.service
Wants=NetworkManager.service

[Service]
Type=oneshot
Environment=IFACE=wlan0
Environment=CONNECTION=fcyy6
ExecStart=/usr/local/sbin/wifi_watchdog.sh
EOF
sudo tee /etc/systemd/system/wifi-watchdog.timer >/dev/null <<'EOF'
[Unit]
Description=Run Wi-Fi watchdog every 15 seconds

[Timer]
OnBootSec=20s
OnUnitActiveSec=15s
AccuracySec=3s
Unit=wifi-watchdog.service

[Install]
WantedBy=timers.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now wifi-watchdog.timer
```

## 3. Verify

```bash
systemctl status wifi-watchdog.timer
systemctl status wifi-watchdog.service
journalctl -u wifi-watchdog.service -n 50 --no-pager
```

## 4. Manual recovery

If Wi-Fi stalls before the timer catches it:

```bash
sudo /usr/local/sbin/wifi_watchdog.sh
```
