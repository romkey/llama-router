#!/bin/sh
# Apply wg0.conf from /etc/wireguard when llama-router updates the shared volume.
set -e
mkdir -p /etc/wireguard

apply_config() {
    if [ -f /etc/wireguard/wg0.conf ] && grep -q '^\[Interface\]' /etc/wireguard/wg0.conf 2>/dev/null; then
        wg-quick down wg0 2>/dev/null || true
        wg-quick up wg0 2>&1 || true
    else
        wg-quick down wg0 2>/dev/null || true
    fi
}

apply_config

while true; do
    # Watch the directory so creates/modifies/deletes of wg0.conf are picked up.
    inotifywait -qq -e modify -e create -e delete -e move /etc/wireguard 2>/dev/null || sleep 20
    sleep 1
    apply_config
done
