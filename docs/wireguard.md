# WireGuard with llama-router

This stack runs **two containers**: a small **WireGuard sidecar** (privileged, runs `wg-quick`) and **llama-router**, which shares the sidecar’s network namespace. You configure keys, addresses, and peers **only in the llama-router dashboard** (WireGuard tab). The app writes `wg0.conf` to a shared volume; the sidecar watches the file and reapplies the tunnel.

## Requirements

- **Linux host** (or VM) with Docker: WireGuard in containers needs `NET_ADMIN` and usually works best on Linux. Docker Desktop on macOS/Windows is often unreliable for this pattern.
- Open **UDP port 51820** (or your chosen listen port) on the host firewall toward the internet if remote peers connect in.

## Quick start

The `wireguard` service image is built in CI and published to **GHCR** as `ghcr.io/romkey/llama-router-wireguard` (tags `latest`, git SHA, and semver on releases). Forks should change the `image:` line in `docker-compose.wireguard.yml` or build locally from `docker/wireguard-sidecar`.

```bash
docker compose -f docker-compose.wireguard.yml pull
docker compose -f docker-compose.wireguard.yml up -d
```

Open the dashboard at `http://<host>/`, go to **WireGuard**:

1. Click **Generate new keypair** (or paste a private key you already trust).
2. Set **Tunnel address (CIDR)** (e.g. `10.8.0.1/24`) and **Listen port** (default `51820`).
3. Enable **Enable WireGuard** and save.
4. Add **Peers** with each remote machine’s **public key** and **AllowedIPs** (e.g. `10.8.0.2/32` for a single peer, or include LAN routes if you route subnets).
5. On **NAT’d** peers, set **Persistent keepalive** (e.g. `25`) on the side that initiates or behind NAT.

**Public endpoint** (dashboard field) is a hint for you when building the remote peer’s config; it is not written into `wg0.conf` automatically.

## Environment variables

| Variable | Description |
|----------|-------------|
| `LLAMA_ROUTER_WIREGUARD_CONFIG_PATH` | Absolute path where llama-router writes `wg0.conf` (must match the volume path inside the container, e.g. `/shared/wireguard/wg0.conf`). |

If this is unset, the dashboard still stores settings in SQLite, but **no file is written** and the sidecar has nothing to apply.

## Reaching remote backends

Once the tunnel is up, add providers as usual using **tunnel IPs** in URLs, e.g. `http://10.8.0.2:11434` for Ollama on a peer.

## OCI cache over the tunnel

If backends pull models through the router cache, set `LLAMA_ROUTER_CACHE_EXTERNAL_HOST` to an address **those backends can reach** on the WireGuard network (often this router’s tunnel IP, e.g. `10.8.0.1`).

## Applying config changes

The sidecar watches `/etc/wireguard` and reapplies `wg0.conf` after the dashboard saves. If something looks stuck, restart the sidecar:

```bash
docker compose -f docker-compose.wireguard.yml restart wireguard
```

## Security notes

- Interface **private keys** and optional **preshared keys** are stored in the **same SQLite database** as the rest of llama-router. Protect backups and filesystem permissions accordingly.
- **Key rotation** is not required on a schedule; rotate if a key may be compromised or a peer is removed permanently.

## Multiple peers

Add one dashboard row per peer. WireGuard supports many `[Peer]` sections; there is no separate limit in the app beyond practicality.
