"""Serving exports.

The bridge runtime should remain importable even when the optional HTTP serving
dependencies are not installed in a lightweight test environment.
"""

try:
    from .model_server import ModelServer, app
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight test envs
    ModelServer = None  # type: ignore[assignment]
    app = None  # type: ignore[assignment]

from .bridge_runtime import BridgeRuntime

__all__ = ["BridgeRuntime", "ModelServer", "app"]

__all__ = ["app", "ModelServer"]
