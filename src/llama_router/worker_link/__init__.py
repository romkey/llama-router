"""Hub <-> worker links: member routers dial out to a hub over a WebSocket.

A *worker* is a llama-router (or any Ollama / llama.cpp server reachable from
it) that opens one outbound WebSocket to a *hub* and serves inference jobs the
hub sends down that socket. Nothing on the worker's network accepts inbound
connections, so it works from behind NAT, firewalls, or another VPN.

On the hub, each worker is an ordinary provider whose HTTP clients use a
custom ``httpx`` transport that tunnels requests over the worker's socket.
"""
