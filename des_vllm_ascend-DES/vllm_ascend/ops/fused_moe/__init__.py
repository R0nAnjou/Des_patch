"""Dynamic Expert Selection plugin for vLLM Ascend."""

from __future__ import annotations


def register() -> None:
    """vLLM general-plugin entry point."""
    from .patch import apply_patch

    apply_patch()


__all__ = ["register"]

