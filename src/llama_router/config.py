from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_path: str = "llama_router.db"
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

    model_config = {"env_prefix": "LLAMA_ROUTER_"}


settings = Settings()
