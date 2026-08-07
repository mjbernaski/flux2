#!/bin/bash
# Restore always-on autostart for the FLUX web server (port 2222).
#
# This re-installs and re-enables the flux-server systemd --user unit that
# was paused/disabled on 2026-08-02. After running this, the server starts
# now and will auto-start on every login/boot again (with linger enabled).
set -euo pipefail

cd "$(dirname "$0")"

systemctl --user enable --now ./flux-server.service
sudo loginctl enable-linger "$USER"

echo "flux-server.service re-enabled and started."
systemctl --user status flux-server.service --no-pager -l
