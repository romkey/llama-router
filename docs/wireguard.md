# WireGuard

WireGuard integration is documented in the main [README](../README.md#wireguard-optional) (prerequisites, Docker `network_mode: host`, peering API, and dashboard connect flow).

For cache access over the tunnel, set `LLAMA_ROUTER_CACHE_EXTERNAL_HOST` to an address your remote Ollama backends can use to reach this router (often the tunnel IP of the llama-router host).
