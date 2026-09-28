"""Install the DES hooks into an already installed vLLM Ascend package."""

from __future__ import annotations

import os
import threading
import warnings
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

TESTED_VLLM_ASCEND_VERSION = "0.23.0rc1"
_PATCH_MARKER = "__vllm_ascend_des_patch__"
_lock = threading.Lock()
_patched = False


def _installed_vllm_ascend_version() -> str:
    try:
        return version("vllm-ascend")
    except PackageNotFoundError:
        try:
            module = import_module("vllm_ascend")
        except ImportError as exc:
            raise RuntimeError(
                "vllm-ascend is not installed in this Python environment"
            ) from exc
        detected = getattr(module, "__version__", None)
        if detected is None:
            raise RuntimeError(
                "cannot determine the installed vllm-ascend version; "
                "install vllm-ascend 0.23.0rc1"
            )
        return str(detected)


def _check_compatibility() -> str:
    detected = _installed_vllm_ascend_version()
    if detected == TESTED_VLLM_ASCEND_VERSION:
        return detected

    message = (
        "vllm-ascend-des-patch was validated only with "
        f"vllm-ascend=={TESTED_VLLM_ASCEND_VERSION}, but found {detected}. "
        "Refusing to replace version-sensitive MoE methods."
    )
    if os.environ.get("VLLM_DES_ALLOW_UNSUPPORTED", "0") == "1":
        warnings.warn(message + " Continuing because VLLM_DES_ALLOW_UNSUPPORTED=1.", stacklevel=2)
        return detected
    raise RuntimeError(message + " Set VLLM_DES_ALLOW_UNSUPPORTED=1 only for development.")


def _replace(module_name: str, attribute: str, replacement: object) -> None:
    target_module = import_module(module_name)
    current = getattr(target_module, attribute)
    if getattr(current, _PATCH_MARKER, False):
        return
    setattr(replacement, _PATCH_MARKER, True)
    setattr(target_module, attribute, replacement)


def _replace_method(
    module_name: str,
    class_name: str,
    method_name: str,
    replacement: object,
) -> None:
    target_module = import_module(module_name)
    target_class = getattr(target_module, class_name)
    current = getattr(target_class, method_name)
    if getattr(current, _PATCH_MARKER, False):
        return
    setattr(replacement, _PATCH_MARKER, True)
    setattr(target_class, method_name, replacement)


def apply_patch() -> None:
    """Patch DES call sites once per process.

    vLLM invokes general plugins in the API process and in every worker. This
    function is intentionally idempotent because plugin loading can happen more
    than once in embedded/test environments.
    """
    global _patched

    with _lock:
        if _patched:
            return

        detected = _check_compatibility()

        generic_selector = import_module("vllm_ascend_des.experts_selector")
        generic_dispatcher = import_module("vllm_ascend_des.token_dispatcher")
        selector_310p = import_module("vllm_ascend_des._310p.experts_selector")
        dispatcher_310p = import_module("vllm_ascend_des._310p.token_dispatcher")

        _replace(
            "vllm_ascend.ops.fused_moe.experts_selector",
            "select_experts",
            generic_selector.select_experts,
        )
        _replace_method(
            "vllm_ascend.ops.fused_moe.token_dispatcher",
            "TokenDispatcherWithAllGather",
            "token_dispatch",
            generic_dispatcher.TokenDispatcherWithAllGather.token_dispatch,
        )
        _replace(
            "vllm_ascend._310p.fused_moe.experts_selector",
            "select_experts",
            selector_310p.select_experts,
        )
        _replace_method(
            "vllm_ascend._310p.fused_moe.token_dispatcher",
            "TokenDispatcherWithAllGather310",
            "token_dispatch",
            dispatcher_310p.TokenDispatcherWithAllGather310.token_dispatch,
        )

        _patched = True
        print(
            "[DES] vLLM Ascend patch plugin loaded "
            f"(vllm-ascend={detected}, target={TESTED_VLLM_ASCEND_VERSION})",
            flush=True,
        )


__all__ = ["TESTED_VLLM_ASCEND_VERSION", "apply_patch"]

