from importlib.metadata import version as _pkg_version, PackageNotFoundError

try:
    __version__ = _pkg_version("llama-router")
except PackageNotFoundError:
    __version__ = "dev"
