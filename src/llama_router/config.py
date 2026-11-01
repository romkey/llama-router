from pathlib import Path

from pydantic_settings import BaseSettings
from sqlalchemy.engine import make_url
from sqlalchemy.engine.url import URL


class Settings(BaseSettings):
    database_path: str = "llama_router.db"
    """Used when ``database_url`` is empty: SQLite file path for the default URL."""

    database_url: str = ""
    """Async SQLAlchemy URL, e.g. ``sqlite+aiosqlite:///...``, ``postgresql+asyncpg://...``."""
    dashboard_host: str = "0.0.0.0"  # noqa: S104  # containers bind all interfaces
    dashboard_port: int = 80
    api_host: str = "0.0.0.0"  # noqa: S104  # containers bind all interfaces
    api_port: int = 11434
    llamacpp_host: str = "0.0.0.0"  # noqa: S104  # containers bind all interfaces
    llamacpp_port: int = 8080
    # Also serve the Ollama and OpenAI APIs and the worker socket on the
    # dashboard port. Their own ports keep listening either way.
    api_on_dashboard_port: bool = False
    health_check_interval_seconds: int = 30
    # Leave successful GET /healthz and GET /health requests out of the access
    # log, so a container or load-balancer health check polling every few
    # seconds does not fill it. Failures are still logged.
    quiet_health_checks: bool = False
    # Proxies whose X-Forwarded-For header is believed, as IPs or CIDRs
    # separated by commas. Empty: the header is ignored and the TCP peer is
    # the client.
    trusted_proxies: str = ""
    # Request log rows older than this many days are deleted (0 keeps them all).
    request_log_retention_days: float = 30.0
    # A read timeout on a request that asked for more output tokens than this
    # is blamed on the request, not the provider, so it does not count
    # toward the provider's circuit breaker.
    circuit_timeout_max_tokens: int = 4096
    # Re-probe a provider's endpoints when the last probe is older than this
    # (0 turns periodic re-probing off; probes on add, edit and recovery stay).
    probe_max_age_hours: float = 24.0
    # Probe providers automatically (on add, edit, recovery and when stale).
    auto_probe_enabled: bool = True
    # Serve Ollama API requests (/api/chat, /api/generate, /api/embed) from
    # OpenAI-style providers too, translating both ways.
    ollama_api_translation: bool = True
    benchmark_prompt: str = "Write a short sentence about the weather."
    # Idle auto-benchmarking. Defaults for the dashboard settings of the same
    # name; the dashboard values win once saved.
    auto_benchmark_enabled: bool = True
    # A provider counts as idle after this long without a proxied request.
    auto_benchmark_idle_seconds: int = 300
    # Benchmarks older than this are run again.
    auto_benchmark_max_age_days: float = 7.0
    # How often the scheduler looks for work.
    auto_benchmark_check_seconds: int = 60
    # Providers benchmarked at once (each provider runs one at a time).
    auto_benchmark_concurrency: int = 1
    # Session affinity: keep each conversation on the provider that served it,
    # so the backend's prompt cache stays warm. An API key can override
    # the enabled flag.
    session_affinity_enabled: bool = True
    # A session forgets its provider after this long without a request.
    session_affinity_ttl_seconds: int = 1800
    # Most sessions remembered at once; the least recently used go first.
    session_affinity_max_sessions: int = 1000
    # Most sessions one API key (or one client address, without a key) may
    # hold, so one client cannot push everyone else's sessions out.
    session_affinity_max_sessions_per_key: int = 200
    # Longest a key's decision policy (tiered routing, gates) may take before
    # the request goes on without it.
    decision_policy_timeout_seconds: float = 3.0
    # Prometheus metrics at GET /metrics on the dashboard port. Scrapers send
    # ``Authorization: Bearer <metrics_token>``; without a token the endpoint
    # needs a dashboard sign-in like the rest of the dashboard.
    metrics_enabled: bool = True
    metrics_token: str = ""
    cache_enabled: bool = True
    cache_dir: str = "./model_cache"
    cache_host: str = "0.0.0.0"  # noqa: S104  # containers bind all interfaces
    cache_port: int = 9200
    cache_external_host: str = ""
    cache_manifest_ttl_hours: int = 240
    cache_max_concurrent_blobs: int = 4
    sentry_dsn: str = ""
    sentry_environment: str = "production"
    sentry_traces_sample_rate: float = 0.0
    sentry_profiles_sample_rate: float = 0.0
    sentry_send_default_pii: bool = False
    # Hub: accept outbound worker connections on the Ollama API port at /worker/ws.
    worker_hub_enabled: bool = True
    # Worker: dial out to this hub (https://, wss://, or bare host). Set together
    # with worker_token to pin the link; otherwise it is configured in the dashboard.
    worker_hub_url: str = ""
    worker_token: str = ""
    # Local upstreams the worker relays jobs to. Empty means this router's own
    # Ollama / llama.cpp API ports on 127.0.0.1; "none" disables that protocol.
    worker_ollama_url: str = ""
    worker_llamacpp_url: str = ""
    # Sent as API-KEY to the local upstream when it requires one.
    worker_upstream_api_key: str = ""
    worker_max_concurrency: int = 4
    # Allow the hub to pull and delete models on this worker.
    worker_allow_admin: bool = False
    # How the hub sees this worker's providers: "combined" (one provider) or
    # "individual" (each shared provider on its own). Empty: the dashboard's
    # choice, combined until set.
    worker_expose: str = ""
    # Where Hugging Face downloads are stored. Mount the same directory into a
    # llama.cpp server's --models-dir to share models without pushing them.
    hf_models_dir: str = "./hf_models"
    # If set, signs dashboard session cookies; otherwise persisted in app_settings.
    session_secret: str = ""
    # Set true behind HTTPS so session cookies are only sent over TLS.
    dashboard_cookie_secure: bool = False
    # OpenID Connect sign-in for the dashboard. An empty issuer turns it off.
    oidc_issuer: str = ""
    oidc_client_id: str = ""
    oidc_client_secret: str = ""
    oidc_scopes: str = "openid profile email groups"
    # Claim holding the user's groups; dotted paths reach nested claims
    # (Keycloak: realm_access.roles).
    oidc_groups_claim: str = "groups"
    # Members of this group are dashboard administrators.
    oidc_admin_group: str = ""
    # If set, only members of this group may sign in at all.
    oidc_allowed_group: str = ""
    # External base URL of the dashboard, e.g. https://llm.example.org. The
    # callback is {base}/login/oidc/callback. Empty uses the request's host.
    oidc_redirect_base_url: str = ""
    # Button label: "Sign in with {name}".
    oidc_display_name: str = "single sign-on"
    # Hide the password form and refuse password sign-in.
    oidc_only: bool = False
    # Also end the session at the identity provider on logout.
    oidc_logout_at_idp: bool = False
    # Permit an http:// issuer (local development only).
    oidc_allow_insecure: bool = False

    model_config = {"env_prefix": "LLAMA_ROUTER_"}

    def effective_database_url(self) -> str:
        """URL used by the app's async SQLAlchemy engine."""
        u = (self.database_url or "").strip()
        if u:
            return u
        path = Path(self.database_path).expanduser().resolve()
        return URL.create("sqlite+aiosqlite", database=str(path)).render_as_string(
            hide_password=False
        )

    def sync_database_url_for_alembic(self, async_url: str | None = None) -> str:
        """Synchronous driver URL for Alembic (no nested asyncio)."""
        u = make_url(async_url or self.effective_database_url())
        dn = u.drivername
        if dn == "sqlite+aiosqlite":
            u = u.set(drivername="sqlite")
        elif dn == "postgresql+asyncpg":
            u = u.set(drivername="postgresql+psycopg")
        elif dn in ("mysql+asyncmy", "mysql+aiomysql") or dn.startswith("mariadb+"):
            u = u.set(drivername="mysql+pymysql")
        # str(URL) masks the password as "***", which Alembic would then use
        # to log in.
        return u.render_as_string(hide_password=False)


settings = Settings()
