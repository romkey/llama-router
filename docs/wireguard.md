# Pairing Two llama-routers over WireGuard

## Prerequisites

Both machines need `wg-quick` installed and llama-router running. On Debian/Ubuntu:
```bash
apt install wireguard-tools
```

Decide on your tunnel IP addresses ahead of time. These are private IPs that only
exist inside the WireGuard tunnel — they do not need to match your LAN addresses.
A simple choice:

- Router A (the one you'll be working from): `10.8.0.1`
- Router B (the remote provider): `10.8.0.2`

There are two ways to pair routers depending on whether Router B's dashboard is
reachable over HTTPS:

- **Automatic** — Router A calls Router B's dashboard API directly. Requires
  Router B's dashboard to be reachable over HTTPS. Both routers get configured
  in one step.
- **Manual exchange** — each operator exports a JSON blob from their own
  dashboard and sends it to the other out-of-band (email, Slack, etc.). No
  network reachability between dashboards is required at setup time. Use this
  when Router B is behind a firewall, only reachable over HTTP, or its dashboard
  is not exposed to the internet (which it shouldn't be without login enabled).

---

## Option A: Automatic connect (Router B reachable over HTTPS)

### On Router B first — set it up to accept connections

**1. Open Router B's dashboard and go to the WireGuard tab.**

**2. Generate a keypair** if one doesn't exist yet. Click "Generate new keypair."
The public key appears in the read-only field below.

**3. Fill in the interface settings:**

- Tunnel address: `10.8.0.2/24`
- Listen port: `51820` (or whatever port is open in Router B's firewall)
- Public endpoint: `router-b.example.com:51820` — this is the address Router A
  will connect to. If Router B is on a dynamic IP or behind NAT, use a hostname
  or leave it blank and rely on Router A initiating the connection.
- Click **Save interface & write config**

**4. Enable inbound peering.** In the "Inbound Peering" card:

- Toggle "Accept peer requests from remote routers" on
- Click **Regenerate** or **Save**. The dashboard then shows the **full peering
  API key** in a read-only field (with a copy button). That value is what you
  put in the `X-Peering-Key` header / "Remote peering API key" on the other
  router — it is **not** the WireGuard `public_key` in the JSON below.
- The **masked prefix** under the password box is only the first characters of
  the key already stored on the server; until you click Save, what you type in
  the password field can differ from that prefix.
- After you reload the page, the full key is hidden again; click **Save** once
  more (leave the key field empty) to show the current stored key for copying.
- Copy the peering API key somewhere safe for the operator of Router A.

**5. Open Router B's firewall** to allow UDP on port 51820 inbound.

That is everything needed on Router B. Leave the dashboard open — you can watch
the peer table update when Router A connects.

---

### On Router A — initiate the connection

**6. Open Router A's dashboard and go to the WireGuard tab.**

**7. Generate a keypair** on Router A if one doesn't exist. Click "Generate new
keypair."

**8. Fill in Router A's interface settings:**

- Tunnel address: `10.8.0.1/24`
- Listen port: `51820`
- Public endpoint: `router-a.example.com:51820` — Router B needs this if you
  want the tunnel to be truly bidirectional and either side can initiate. If
  Router A is behind NAT and Router B is publicly reachable, you can leave
  Router A's endpoint blank and set a persistent keepalive instead.
- Click **Save interface & write config**

**9. Use the "Connect to Remote llama-router" form:**

> **Note:** The remote URL must use HTTPS. The peering API key is transmitted
> as an HTTP header, so a plaintext HTTP connection would expose it on the wire.
> If Router B's dashboard is only reachable over HTTP or is behind a firewall,
> use the Manual exchange workflow below instead.

Fill in:

| Field | Value |
|-------|-------|
| Remote router URL | `https://router-b.example.com` (Router B's dashboard URL, HTTPS required) |
| Remote peering API key | the key you copied from Router B in step 4 |
| Our tunnel IP | `10.8.0.1` |
| Their tunnel IP | `10.8.0.2` |
| Also add as provider | ✓ checked |

Click **Connect.**

---

### What happens automatically

Router A's dashboard calls Router B's peering API in the background and does the
following in sequence:

1. Fetches Router B's public key, endpoint, and API URLs
2. Registers Router A as a peer on Router B — Router B adds Router A to its
   WireGuard config and brings the peer live
3. Adds Router B as a peer on Router A — Router A adds Router B to its
   WireGuard config and brings the peer live
4. Both tunnels come up and perform a WireGuard handshake
5. Router B adds Router A as a provider (because `add_as_provider` was sent in
   the peer request)
6. Router A adds Router B as a provider using the tunnel IP
   `http://10.8.0.2:11434`

The connect form shows a status banner as each step completes. When it finishes
you should see something like:
```
✓ Connected to router-b
  WireGuard peer added on both sides
  Provider "router-b" added (http://10.8.0.2:11434)
  Router A added as provider on router-b
```

---

## Option B: Manual peer exchange (Router B behind a firewall or no HTTPS)

Use this when Router B's dashboard is not reachable over HTTPS from Router A —
for example when both routers are behind separate firewalls, or when the
dashboard is intentionally not exposed to the internet (the recommended
configuration when dashboard login is enabled).

Both operators work independently on their own dashboards. No network
connectivity between the two dashboards is required at setup time. The only
out-of-band communication needed is sharing two small JSON blobs.

### On both routers — set up the interface first

Each operator completes steps 1–3 from Option A on their own router:

- Generate a keypair if one doesn't exist
- Set tunnel address (`10.8.0.1/24` on Router A, `10.8.0.2/24` on Router B)
- Set listen port (`51820`)
- Set public endpoint if the router is publicly reachable over UDP
- Click **Save interface & write config**

### Exchange peer configs

**On Router A:**

1. Go to WireGuard tab → "Manual peer exchange" → click **Generate export**
2. A JSON blob appears in the read-only textarea. Copy it and send it to the
   Router B operator (email, Slack, a shared document — anything secure)

**On Router B:**

3. Go to WireGuard tab → "Manual peer exchange" → paste Router A's JSON into
   the "Import remote config" textarea
4. Set "Our tunnel IP" to `10.8.0.2` (should be pre-filled from your interface
   settings)
5. Check "Also add as provider" if you want Router A added as a provider
   automatically
6. Click **Import peer** — Router B adds Router A to its WireGuard config and
   writes the updated `wg0.conf`
7. Click **Generate export** on Router B, copy the JSON, and send it back to
   the Router A operator

**On Router A:**

8. Paste Router B's JSON into the "Import remote config" textarea
9. Set "Our tunnel IP" to `10.8.0.1`
10. Check "Also add as provider"
11. Click **Import peer**

Once both sides have imported each other's config, the WireGuard handshake
happens automatically. No further steps are needed — the tunnel comes up and the
provider URLs using the tunnel IPs (`http://10.8.0.2:11434` etc.) become
reachable.

---

## Verify it worked

On Router A:

- WireGuard tab → peer table shows Router B with a green handshake timestamp
  ("✓ 3s ago") and "router-b" in the Provider column
- Providers tab shows "router-b" with status idle and its models listed

On Router B:

- WireGuard tab → peer table shows Router A with a green handshake timestamp
- Providers tab shows "router-a" as a provider

Test routing actually works by sending a request to Router A for a model that
only Router B has:
```bash
curl http://router-a.example.com:11434/api/chat \
  -d '{"model": "llama3.2:latest", "messages": [{"role": "user", "content": "hello"}]}'
```

Router A should route the request transparently through the tunnel to Router B
and stream the response back.

---

## Ongoing operation

The WireGuard tab on each router shows live handshake status updated every 30
seconds. A green timestamp means packets are flowing. If it turns red, the
tunnel is down but Router A will still attempt to reach Router B at the
application layer — if the application health check also fails, Router B's
provider status will go offline and Router A will stop routing to it
automatically.

If Router B goes offline intentionally, nothing needs to be done on Router A —
the health check loop marks it offline and routes around it. When Router B comes
back, the health check detects it and marks it online again.

---

## Teardown

To disconnect the pairing, go to the WireGuard tab on either router and delete
the peer for the other router. You'll be asked whether to also remove the linked
provider. Choose yes to fully clean up, or no to keep the provider record but
sever the WireGuard link.