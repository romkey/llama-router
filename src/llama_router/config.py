from pathlib import Path

from pydantic_settings import BaseSettings
from sqlalchemy.engine import make_url
from sqlalchemy.engine.url import URL


class Settings(BaseSettings):
    database_path: str = "llama_router.db"
    """Used when ``database_url`` is empty: SQLite file path for the default async URL."""

    database_url: str = ""
    """Async SQLAlchemy URL, e.g. ``sqlite+aiosqlite:///...``, ``postgresql+asyncpg://...``."""
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 80
    api_host: str = "0.0.0.0"
    api_port: int = 11434
    llamacpp_host: str = "0.0.0.0"
    llamacpp_port: int = 8080
    health_check_interval_seconds: int = 30
    benchmark_prompt: str = "Write a short sentence about the weather."
    cache_enabled: bool = True
    cache_dir: str = "./model_cache"
    cache_host: str = "0.0.0.0"
    cache_port: int = 9200
    cache_external_host: str = ""
    cache_manifest_ttl_hours: int = 240
    cache_max_concurrent_blobs: int = 4
    sentry_dsn: str = ""
    sentry_environment: str = "production"
    sentry_traces_sample_rate: float = 0.0
    sentry_profiles_sample_rate: float = 0.0
    sentry_send_default_pii: bool = False
    # Host path for wg-quick config (interface name = basename without .conf).
    wireguard_config_path: str = "/etc/wireguard/wg0.conf"
    # If true, only write the file (no wg-quick/wg); for external tunnel management.
    wireguard_legacy_volume: bool = False
    # If true, apply WireGuard on process startup when wg-quick is available.
    wireguard_enabled: bool = False
    # If set, used to sign dashboard session cookies; otherwise persisted in app_settings.
    session_secret: str = ""
    # Set true behind HTTPS so session cookies are only sent over TLS.
    dashboard_cookie_secure: bool = False

    model_config = {"env_prefix": "LLAMA_ROUTER_"}

    def effective_database_url(self) -> str:
        """URL used by the app’s async SQLAlchemy engine."""
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
            return str(u.set(drivername="sqlite"))
        if dn == "postgresql+asyncpg":
            return str(u.set(drivername="postgresql+psycopg"))
        if dn in ("mysql+asyncmy", "mysql+aiomysql"):
            return str(u.set(drivername="mysql+pymysql"))
        if dn.startswith("mariadb+"):
            return str(u.set(drivername="mysql+pymysql"))
        return str(u)


settings = Settings()
