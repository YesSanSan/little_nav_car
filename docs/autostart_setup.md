# Autostart setup

This repository keeps the serial node and `web.sh` out of the normal
`./nav.sh` and `./slam.sh` startup flow. They can be started separately at boot
with `systemd`.

## 1. Install autostart services

From the repository root:

```bash
sudo ./scripts/install_autostart.sh
```

This installs and enables:

- `little-nav-car-serial.service`
- `little-nav-car-web.service`

## 2. Verify

```bash
systemctl status little-nav-car-serial.service
systemctl status little-nav-car-web.service
```

The services call these repo scripts:

- `./auto_serial.sh --detach`
- `./web.sh --detach`

Their tmux sessions can be inspected with:

```bash
tmux attach -t serial
tmux attach -t web
```

## 3. Manage manually

```bash
sudo systemctl restart little-nav-car-serial.service
sudo systemctl restart little-nav-car-web.service
sudo systemctl disable --now little-nav-car-serial.service
sudo systemctl disable --now little-nav-car-web.service
```
