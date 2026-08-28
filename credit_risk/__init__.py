"""Credit risk modeling and underwriting decision support."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("credit-risk-dss")
except PackageNotFoundError:
    __version__ = "1.0.0"

__all__ = ["__version__"]
