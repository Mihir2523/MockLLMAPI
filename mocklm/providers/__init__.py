"""
MockLLM Providers
~~~~~~~~~~~~~~~~~
Provider registry — maps provider names to their implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseProvider

# ── Registry ──────────────────────────────────────────────────────────────────
# Maps lowercase provider name → provider class.
# New providers are registered here.

_REGISTRY: dict[str, type[BaseProvider]] = {}


def register_provider(cls: type[BaseProvider]) -> type[BaseProvider]:
    """Decorator to register a provider class."""
    _REGISTRY[cls.name.lower()] = cls
    return cls


def get_provider(name: str) -> BaseProvider:
    """
    Get a provider instance by name.

    Args:
        name: Provider name (case-insensitive), e.g. "gemini", "chatgpt".

    Returns:
        An instance of the matching provider.

    Raises:
        ProviderNotFoundError: If the name doesn't match any registered provider.
    """
    from ..exceptions import ProviderNotFoundError

    key = name.lower().strip()
    if key not in _REGISTRY:
        raise ProviderNotFoundError(key, list(_REGISTRY.keys()))
    return _REGISTRY[key]()


def list_providers() -> list[str]:
    """Return a list of all registered provider names."""
    return list(_REGISTRY.keys())


# ── Import providers so they auto-register ────────────────────────────────────
from . import gemini  # noqa: E402, F401
