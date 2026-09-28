"""Install the DES hooks into an already installed vLLM Ascend package."""

from __future__ import annotations

import os
import threading
import warnings
from copy import copy
from functools import wraps
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from inspect import signature

TESTED_VLLM_ASCEND_VERSION = "0.23.0"
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
                "install vllm-ascend 0.23.0"
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
        warnings.warn(
            message + " Continuing because VLLM_DES_ALLOW_UNSUPPORTED=1.",
            stacklevel=2,
        )
        return detected
    raise RuntimeError(
        message + " Set VLLM_DES_ALLOW_UNSUPPORTED=1 only for development."
    )


def _argument_reader(function):
    positions = {name: index for index, name in enumerate(signature(function).parameters)}

    def read(args, kwargs, name, default=None):
        if name in kwargs:
            return kwargs[name]
        position = positions.get(name)
        if position is not None and position < len(args):
            return args[position]
        return default

    return read


def _copy_with_attribute(obj, name: str, value):
    clone = copy(obj)
    try:
        setattr(clone, name, value)
    except (AttributeError, TypeError):
        try:
            object.__setattr__(clone, name, value)
        except (AttributeError, TypeError) as exc:
            raise RuntimeError(
                f"cannot patch {type(obj).__name__}.{name}; unsupported "
                "vllm-ascend runtime metadata type"
            ) from exc
    return clone


def _install_selector_wrapper(module_name: str, des_module) -> None:
    target_module = import_module(module_name)
    original = target_module.select_experts
    if getattr(original, _PATCH_MARKER, False):
        return
    if hasattr(target_module, "DES") or hasattr(target_module, "apply_des_cutoff"):
        raise RuntimeError(
            f"{module_name} already contains an in-tree DES patch; restore the "
            "stock vllm-ascend files before installing this plugin"
        )

    read_argument = _argument_reader(original)

    @wraps(original)
    def select_experts(*args, **kwargs):
        topk_weights, topk_ids = original(*args, **kwargs)
        if not des_module.DES.enabled:
            return topk_weights, topk_ids
        if read_argument(args, kwargs, "scoring_func", "softmax") != "softmax":
            return topk_weights, topk_ids

        top_k = int(read_argument(args, kwargs, "top_k"))
        routed_weights = topk_weights[..., :top_k]
        routed_ids = topk_ids[..., :top_k]
        routed_weights, routed_ids = des_module.apply_des_cutoff(
            routed_weights,
            routed_ids,
            des_module.DES.threshold,
            des_module.DES.min_k,
        )

        # Generic 0.23.0 may append shared experts after the routed top-k.
        # DES applies only to routed experts; preserve the shared tail exactly.
        if topk_weights.shape[-1] > top_k:
            import torch

            routed_weights = torch.cat(
                (routed_weights, topk_weights[..., top_k:]), dim=-1
            )
            routed_ids = torch.cat((routed_ids, topk_ids[..., top_k:]), dim=-1)
        return routed_weights, routed_ids

    setattr(select_experts, _PATCH_MARKER, True)
    target_module.select_experts = select_experts


def _install_dispatcher_wrapper(
    module_name: str,
    class_name: str,
    des_module,
) -> None:
    target_module = import_module(module_name)
    target_class = getattr(target_module, class_name)
    original = target_class.token_dispatch
    if getattr(original, _PATCH_MARKER, False):
        return
    if hasattr(target_module, "DES") or hasattr(target_module, "log_skip_active"):
        raise RuntimeError(
            f"{module_name} already contains an in-tree DES patch; restore the "
            "stock vllm-ascend files before installing this plugin"
        )

    @wraps(original)
    def token_dispatch(self, token_dispatch_input):
        expert_map = token_dispatch_input.routing.expert_map
        des_skip = (
            des_module.DES.enabled
            and des_module.DES.skip
            and expert_map is None
            and self.top_k > 1
        )
        if not des_skip:
            return original(self, token_dispatch_input)

        import torch

        real_expert_count = int(self.num_experts_local)
        dropped = token_dispatch_input.topk_weights == 0
        patched_ids = torch.where(
            dropped,
            torch.full_like(token_dispatch_input.topk_ids, real_expert_count),
            token_dispatch_input.topk_ids,
        )
        patched_input = _copy_with_attribute(
            token_dispatch_input, "topk_ids", patched_ids
        )

        des_module.log_skip_active(
            class_name,
            top_k=self.top_k,
            num_experts_local=real_expert_count,
            hidden_dtype=token_dispatch_input.hidden_states.dtype,
        )

        # On the no-EP path, both 0.23.0 dispatchers derive expert_num and the
        # active range from num_experts_local. Temporarily expose one fake bucket
        # so zero-weight slots sort to the tail, then remove that bucket from the
        # grouped-matmul group list. Calls are synchronous within a worker.
        self.num_experts_local = real_expert_count + 1
        try:
            output = original(self, patched_input)
        finally:
            self.num_experts_local = real_expert_count

        real_group_list = output.group_list[:real_expert_count].contiguous()
        return _copy_with_attribute(output, "group_list", real_group_list)

    setattr(token_dispatch, _PATCH_MARKER, True)
    target_class.token_dispatch = token_dispatch


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

        des_module = import_module("vllm_ascend_des.des")

        _install_selector_wrapper(
            "vllm_ascend.ops.fused_moe.experts_selector",
            des_module,
        )
        _install_dispatcher_wrapper(
            "vllm_ascend.ops.fused_moe.token_dispatcher",
            "TokenDispatcherWithAllGather",
            des_module,
        )
        _install_selector_wrapper(
            "vllm_ascend._310p.fused_moe.experts_selector",
            des_module,
        )
        _install_dispatcher_wrapper(
            "vllm_ascend._310p.fused_moe.token_dispatcher",
            "TokenDispatcherWithAllGather310",
            des_module,
        )

        _patched = True
        print(
            "[DES] vLLM Ascend patch plugin loaded "
            f"(vllm-ascend={detected}, target={TESTED_VLLM_ASCEND_VERSION})",
            flush=True,
        )


__all__ = ["TESTED_VLLM_ASCEND_VERSION", "apply_patch"]
