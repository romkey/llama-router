from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("llama-router")
except PackageNotFoundError:
    __version__ = "dev"
