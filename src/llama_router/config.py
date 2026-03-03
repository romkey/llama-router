from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_path: str = "llama_router.db"
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 80
    api_host: str = "0.0.0.0"
    api_port: int = 11434
    health_check_interval_seconds: int = 30
    benchmark_prompt: str = "Write a short sentence about the weather."

    model_config = {"env_prefix": "LLAMA_ROUTER_"}


settings = Settings()
