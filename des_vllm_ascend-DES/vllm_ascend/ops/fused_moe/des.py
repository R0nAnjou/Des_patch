# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamic Expert Selection (DES) for vLLM-Ascend.

Ported from the validated GPU implementation
(vllm/model_executor/layers/fused_moe/des.py). Same algorithm, same env-var config,
so a single configuration controls DES on both GPU and NPU.

Instead of always using a fixed ``top_k`` experts per token, DES keeps only the
*valuable* experts: take the qwen3-default **renormalized** top-k weights (softmax
over the top-k, summing to 1), sort descending, take the running (cumulative) sum,
and drop experts once the cumulative score crosses a ``threshold``. The expert that
*crosses* the threshold is KEPT (min_k guaranteed). Surviving weights are kept
as-is (no re-normalization after the cut); dropped slots get weight 0.

Two modes, both FULL-GRAPH (aclgraph) safe:
  * Accuracy-only (default): dropped slots keep their VALID expert id and only the
    weight is zeroed, so the Ascend compute path is unchanged and a dropped expert
    contributes 0 at combine.
  * Compute-skip (VLLM_DES_SKIP=1): token_dispatcher routes the zero-weight slots
    to a fake expert bucket so ``npu_grouped_matmul`` skips those rows. This stays
    full-graph safe because the op's output height = x.shape[0] (= active_num,
    STATIC), not sum(group_list). The skipped tail rows are discarded at combine
    (dropped slots carry weight 0; the tail is finite, so 0*value=0). Scope:
    unquantized AllGather (no-EP) + softmax + top_k>1. See token_dispatcher.py.

All ops here are static-shape tensor ops (sort/cumsum/compare/where): output stays
``[num_tokens, top_k]`` so it composes in full graph mode (only values vary).

Config via env vars (read once at import; server restart to change):
  VLLM_DES_ENABLE     "1" to enable DES (default off)
  VLLM_DES_THRESHOLD  cumulative-mass cutoff over the renormalized top-k (default 1.0)
  VLLM_DES_MIN_K      always keep at least this many experts (default 1)
  VLLM_DES_SKIP       "1" to enable the compute-skip (no-EP unquantized softmax)
  VLLM_DES_BACKEND    cutoff implementation: "torch" (default, reference),
                      "triu", "fused", or "fused_sorted". The last skips the
                      sort and is only valid when the router emits descending
                      weights -- see des_cutoff_fused_sorted.

NOTE ON ENV VARS: AGENTS.md asks for env vars to live in ``vllm_ascend/envs.py``.
DES deliberately keeps its own here so the whole feature is one self-contained
file plus call-site hooks, which keeps the out-of-tree patch small. If DES is
ever upstreamed these should move to ``envs.py``.

--------------------------------------------------------------------------------
BACKEND SEAM (for a future fused kernel)
--------------------------------------------------------------------------------
The cutoff is expressed as ONE function behind a registry so a fused NPU kernel
can replace it without touching any call site. To add one:

    from vllm_ascend.ops.fused_moe.des import register_des_backend

    def my_fused_cutoff(topk_weights, topk_ids, threshold, min_k, fake_id):
        ...
        return new_weights, new_ids

    register_des_backend("fused", my_fused_cutoff)
    # then run with VLLM_DES_BACKEND=fused

Any backend MUST satisfy the contract in ``des_cutoff_contract.__doc__`` below.
``DES/scripts/bench_des_cutoff.py`` benchmarks registered backends against each
other and checks conformance against the reference, so a candidate kernel can be
validated and timed before it is wired into a model.
"""
import functools
import os
from collections.abc import Callable

import torch

# Signature every DES cutoff backend must implement. Kept as a named alias so the
# registry, the reference implementation and any future kernel cannot drift apart.
DESCutoffFn = Callable[
    [torch.Tensor, torch.Tensor, float, int, "int | None"],
    "tuple[torch.Tensor, torch.Tensor]",
]

DEFAULT_BACKEND_NAME = "torch"


class DESConfig:
    def __init__(self) -> None:
        self.enabled = os.environ.get("VLLM_DES_ENABLE", "0") == "1"
        self.threshold = float(os.environ.get("VLLM_DES_THRESHOLD", "1.0"))
        self.min_k = int(os.environ.get("VLLM_DES_MIN_K", "1"))
        # Compute-skip (Option A): route DES-dropped slots to a fake expert so the
        # grouped matmul skips those rows instead of just zeroing the weight.
        # FULL-GRAPH safe: npu_grouped_matmul output height = x.shape[0]
        # (= active_num, STATIC), NOT sum(group_list) (verified:
        # scripts/spike_npu_grouped_matmul_static_output.py Q1a). The skipped tail
        # rows are discarded at combine (dropped slots carry weight 0; tail is
        # finite). Wired in token_dispatcher.py (fake bucket via expert_num+1, then
        # group_list=expert_tokens[:E]).
        # Scope: unquantized AllGather (no-EP) + softmax + top_k>1. 310P measured
        # ~1.29x on the isolated grouped matmul (dense 2.38ms -> skip 1.84ms).
        self.skip = os.environ.get("VLLM_DES_SKIP", "0") == "1"
        self.backend = os.environ.get("VLLM_DES_BACKEND", DEFAULT_BACKEND_NAME)

    def __repr__(self) -> str:
        return (
            f"DESConfig(enabled={self.enabled}, threshold={self.threshold}, "
            f"min_k={self.min_k}, skip={self.skip}, backend={self.backend!r})"
        )


# Read once at import. Fixed for the lifetime of the process (and graph capture).
DES = DESConfig()

if DES.enabled:
    print(f"[DES] Dynamic Expert Selection ENABLED (vllm-ascend): {DES}", flush=True)


def des_cutoff_contract() -> None:
    """The contract every DES cutoff backend must satisfy.

    A fused kernel is a drop-in replacement only if all of the following hold.
    ``DES/scripts/bench_des_cutoff.py --check`` verifies them against the
    reference implementation.

    Inputs
      topk_weights  [num_tokens, top_k], floating point. The *renormalized*
                    top-k weights (each row sums to ~1). Not assumed sorted.
      topk_ids      [num_tokens, top_k], integer. Expert ids, same layout.
      threshold     keep experts while the cumulative sum *strictly before* them
                    is < threshold. The crossing expert is KEPT.
      min_k         the first ``min_k`` slots (by descending weight) are always
                    kept, whatever the threshold says.
      fake_id       None  -> dropped slots keep their real expert id.
                    int   -> dropped slots' ids are replaced by this value.

    Outputs
      (weights, ids), both [num_tokens, top_k], dtypes matching the inputs.

    Semantics that are easy to get wrong
      1. SLOTS ARE REORDERED. Both outputs are sorted by descending weight, so
         column j of the output is not column j of the input. Callers rely on
         this ordering; a kernel that keeps the original order is NOT conformant.
      2. NO RENORMALIZATION. Surviving weights keep their original values; they
         will not sum to 1 after a cut.
      3. Dropped weights are exactly 0, so they contribute nothing at combine.
      4. Shapes are STATIC. Output must stay [num_tokens, top_k] even though the
         number of kept experts varies per row -- this is what makes DES safe
         under aclgraph capture. A kernel must not compact rows.
      5. No host synchronisation. No ``.item()``, no ``.cpu()``, no data-dependent
         control flow: those stall the NPU pipeline and break graph capture.
    """


def des_cutoff_torch(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    threshold: float,
    min_k: int = 1,
    fake_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference DES cutoff, built from portable torch ops.

    This is the correctness oracle for any future fused kernel. See
    ``des_cutoff_contract`` for the semantics it defines.
    """
    # Sort descending so the cumulative sum is well-defined regardless of the
    # order the upstream selection returned.
    sorted_w, order = torch.sort(topk_weights, dim=-1, descending=True)
    sorted_ids = torch.gather(topk_ids, -1, order)

    k = sorted_w.shape[-1]
    inclusive = torch.cumsum(sorted_w, dim=-1)
    exclusive = inclusive - sorted_w  # cumulative sum *before* each expert

    keep = exclusive < threshold  # crossing expert kept; later experts dropped
    if min_k > 0:
        pos = torch.arange(k, device=topk_weights.device)
        keep = keep | (pos.unsqueeze(0) < min_k)

    new_w = torch.where(keep, sorted_w, torch.zeros_like(sorted_w))
    if fake_id is not None:
        sorted_ids = torch.where(
            keep, sorted_ids, torch.full_like(sorted_ids, fake_id)
        )
    return new_w, sorted_ids


@functools.lru_cache(maxsize=8)
def _prefix_sum_matrix(top_k: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Upper-triangular ones matrix, so ``w @ M`` is an inclusive prefix sum.

    Cached because it depends only on (top_k, dtype, device), all fixed for the
    lifetime of a served model.
    """
    return torch.triu(torch.ones(top_k, top_k, dtype=dtype, device=device))


def des_cutoff_triu(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    threshold: float,
    min_k: int = 1,
    fake_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same cutoff, with the prefix sum done as a triangular matmul.

    Identical maths to ``des_cutoff_torch``; only the cumulative sum differs.

    WHY: ``torch.cumsum`` along a length-``top_k`` axis hits a pathological slow
    path on 310P3 below ~16384 rows -- measured 7.27 ms at 8192 rows for what is
    arithmetically 7 additions per row, which was 89% of the whole cutoff. Above
    16384 rows it switches to a fast path and costs a flat 0.15 ms. Expressing
    the prefix sum as a matmul against a fixed upper-triangular ones matrix
    avoids the slow path entirely and is flat at ~0.15 ms everywhere:

        tokens    cumsum    triu     speedup
          1024    1.015    0.147       7.0x
          4096    3.690    0.150      25.6x
          8192    7.272    0.150      48.9x
         16384    0.156    0.152       1.03x   (no regression)

    ACCURACY: this is MORE accurate than ``torch.cumsum``, not less. Measured
    against an exact prefix sum, at every size tested:

        rows     cumsum err   triu err
        1024      1.18e-03    3.24e-04
        8192      1.25e-03    3.55e-04
       16384      1.47e-03    3.63e-04
       32768      1.47e-03    3.72e-04

    ``torch.cumsum`` on an fp16 tensor accumulates in fp16, while the matmul
    accumulates in fp32 internally, so this loses roughly 4x less precision. The
    handful of keep/drop decisions where the two disagree (~0.01% of slots, all
    within fp16 epsilon of the threshold) are ones where THIS function is the
    closer to correct.
    """
    sorted_w, order = torch.sort(topk_weights, dim=-1, descending=True)
    sorted_ids = torch.gather(topk_ids, -1, order)

    k = sorted_w.shape[-1]
    prefix = _prefix_sum_matrix(k, sorted_w.dtype, sorted_w.device)
    inclusive = sorted_w @ prefix
    exclusive = inclusive - sorted_w

    keep = exclusive < threshold
    if min_k > 0:
        pos = torch.arange(k, device=topk_weights.device)
        keep = keep | (pos.unsqueeze(0) < min_k)

    new_w = torch.where(keep, sorted_w, torch.zeros_like(sorted_w))
    if fake_id is not None:
        sorted_ids = torch.where(
            keep, sorted_ids, torch.full_like(sorted_ids, fake_id)
        )
    return new_w, sorted_ids


# Large enough to force the min_k comparison true, small enough that adding it to
# weights in [0, 1] stays far from fp16's 65504 limit.
_MIN_K_BIAS = 1024.0


@functools.lru_cache(maxsize=32)
def _keep_bias(top_k: int, min_k: int, threshold: float,
               dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """``threshold`` plus the min_k bias, precomputed.

    Folding both constants into one cached row turns (sub, cmp, or) into
    (add, cmp): ``exclusive < threshold`` is ``inclusive < threshold + w``, and
    biasing the first ``min_k`` entries makes their comparison unconditionally
    true. Both values are fixed for the life of a served model.
    """
    bias = torch.full((top_k,), float(threshold), dtype=dtype, device=device)
    if min_k > 0:
        bias[:min_k] += _MIN_K_BIAS
    return bias


def _des_cutoff_fused_impl(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    threshold: float,
    min_k: int,
    fake_id: int | None,
    assume_sorted: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused cutoff. 5 device ops when the input is already sorted, 7 otherwise.

    Standalone development version and full measurements:
    ``DES/kernels/des_fused_op.py`` and ``DES/results_des_fused_op.md``.
    """
    if assume_sorted:
        sorted_w, sorted_ids = topk_weights, topk_ids
    else:
        sorted_w, order = torch.sort(topk_weights, dim=-1, descending=True)
        sorted_ids = torch.gather(topk_ids, -1, order)

    k = sorted_w.shape[-1]
    inclusive = sorted_w @ _prefix_sum_matrix(k, sorted_w.dtype, sorted_w.device)
    rhs = sorted_w + _keep_bias(k, min_k, threshold, sorted_w.dtype, sorted_w.device)
    keep = inclusive < rhs

    new_w = sorted_w * keep.to(sorted_w.dtype)   # cheaper than where(): no zeros tensor
    if fake_id is None:
        return new_w, sorted_ids
    return new_w, torch.where(keep, sorted_ids, torch.full_like(sorted_ids, fake_id))


def des_cutoff_fused(topk_weights, topk_ids, threshold, min_k=1, fake_id=None):
    """Fused cutoff that still sorts. Safe on every routing path."""
    return _des_cutoff_fused_impl(topk_weights, topk_ids, threshold, min_k,
                                  fake_id, assume_sorted=False)


def des_cutoff_fused_sorted(topk_weights, topk_ids, threshold, min_k=1, fake_id=None):
    """Fused cutoff that SKIPS the sort. Only valid when weights arrive descending.

    DANGEROUS IF MISAPPLIED, which is why it is a separate backend name rather
    than a flag on the safe one: on unsorted input it silently selects the wrong
    experts. There is no crash and no warning, because the cutoff cannot verify
    sortedness without a host sync, which would break aclgraph capture.

    Valid: the non-grouped selector, which ends in ``topk_weights.topk(top_k,
    dim=-1)`` with no ``sorted=`` argument and therefore defaults to sorted=True
    (measured: 100% of rows descending at n=1/64/1024/4096).

    NOT valid: ``_select_expert_use_group_topk``, which passes ``sorted=False``.
    Raw ``torch.topk(sorted=False)`` on NPU genuinely returns unsorted output
    (measured: 0/2048 rows descending), so grouped-top-k models MUST use
    "fused" or "torch" instead.

    Verify with ``DES/scripts/verify_selector_sorted.py`` before enabling this
    for a model you have not tested.
    """
    return _des_cutoff_fused_impl(topk_weights, topk_ids, threshold, min_k,
                                  fake_id, assume_sorted=True)


# Mutable module state, which AGENTS.md asks us to avoid. It is justified here
# because a backend registry is exactly a late-binding table: a fused kernel
# living in another module (or shipped separately) must be able to register
# itself without any call site importing it. Kept private and small.
_DES_BACKEND_REGISTRY: dict[str, DESCutoffFn] = {
    DEFAULT_BACKEND_NAME: des_cutoff_torch,
    "triu": des_cutoff_triu,
    "fused": des_cutoff_fused,
    # skips the sort -- read des_cutoff_fused_sorted's docstring before using
    "fused_sorted": des_cutoff_fused_sorted,
}


def register_des_backend(name: str, fn: DESCutoffFn, *, replace: bool = False) -> None:
    """Register a DES cutoff implementation under ``name``.

    Refuses to shadow an existing name unless ``replace=True`` -- a silent
    override would make it impossible to tell which implementation produced a
    given benchmark number.
    """
    if name in _DES_BACKEND_REGISTRY and not replace:
        raise ValueError(
            f"DES backend {name!r} is already registered; pass replace=True to override"
        )
    _DES_BACKEND_REGISTRY[name] = fn


def available_des_backends() -> tuple[str, ...]:
    """Names of all registered DES cutoff backends."""
    return tuple(sorted(_DES_BACKEND_REGISTRY))


def _try_import_optional(name: str) -> None:
    """Import a module that self-registers a backend, if the name asks for it.

    Keeps optional backends (which may need the CANN toolkit) out of the import
    path unless actually selected.
    """
    if name == "tik":
        try:
            import vllm_ascend_des.des_tik  # noqa: F401
        except Exception as exc:
            raise ValueError(
                f"DES backend 'tik' requested but its module failed to import: "
                f"{type(exc).__name__}: {exc}") from None


def resolve_des_backend(name: str | None = None) -> DESCutoffFn:
    """Look up a backend by name, defaulting to the configured one.

    Fails loudly on an unknown name. Silently falling back to the reference would
    mean a run requested with VLLM_DES_BACKEND=fused could quietly measure the
    torch path instead, which is the kind of error that survives into a report.
    """
    key = DES.backend if name is None else name
    if key not in _DES_BACKEND_REGISTRY:
        _try_import_optional(key)
    try:
        return _DES_BACKEND_REGISTRY[key]
    except KeyError:
        raise ValueError(
            f"unknown DES backend {key!r}; available: {available_des_backends()}"
        ) from None


# Resolved once at import so the hot path does not pay a dict lookup or an env
# read per call. Re-resolved by set_des_backend() when a kernel registers late.
_active_des_cutoff: DESCutoffFn = resolve_des_backend()


def set_des_backend(name: str) -> None:
    """Switch the active backend (used by benchmarks and by late registration)."""
    global _active_des_cutoff
    _active_des_cutoff = resolve_des_backend(name)


def apply_des_cutoff(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    threshold: float,
    min_k: int = 1,
    fake_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the DES cutoff via the active backend.

    Stable entry point for all call sites (experts_selector, token_dispatcher).
    Swapping in a fused kernel means registering it and setting
    VLLM_DES_BACKEND -- no call site changes.
    """
    out = _active_des_cutoff(topk_weights, topk_ids, threshold, min_k, fake_id)
    if _STATS:
        _record_drop_rate(out[0])
    return out


# VLLM_DES_STATS=1 reports what fraction of expert slots the cutoff actually
# drops. This is the quantity that governs BOTH the speed gain and the accuracy
# cost, and it had never been measured -- every estimate came from a synthetic
# softmax that turned out to be far peakier than the real router. Off by default
# because it forces a device sync.
_STATS = os.environ.get("VLLM_DES_STATS", "0") == "1"
_STATS_EVERY = int(os.environ.get("VLLM_DES_STATS_EVERY", "480"))
# OPTIONAL extra filter: sample only this exact row count (0 = any). This used to
# be the primary guard against syncing inside capture; _unsafe_to_sync() now is,
# so this is only for narrowing a measurement to one batch shape.
_STATS_ROWS = int(os.environ.get("VLLM_DES_STATS_ROWS", "0"))
# Window resets every print; cumulative spans the whole run. Both are reported,
# because a single window is one sampled call and reads like an average.
_stats = {"calls": 0, "samples": 0,
          "win_kept": 0, "win_total": 0, "cum_kept": 0, "cum_total": 0}


def _unsafe_to_sync() -> bool:
    """True if a device->host sync here could kill the engine, or would lie.

    Fails CLOSED: any uncertainty means unsafe. A skipped sample costs one data
    point; a sync inside aclgraph capture costs the whole engine, which is not a
    hypothetical -- it died at startup with::

        Not allow to synchronize captured-stream, stream_id=14
        rtStreamSynchronizeWithTimeout failed, reason=stream is captured

    ``torch.npu.is_current_stream_capturing()`` is NOT sufficient and was the
    entire bug: it does not see vllm-ascend's AclmdlRICapture, so it returns
    False *during* capture and the sync proceeded anyway. The authoritative
    signal is ``forward_context.capturing``, which vllm-ascend sets around
    capture (compilation/acl_graph.py) and which its own attention backends
    consult for exactly this reason.

    ``in_profile_run`` is also treated as unsafe -- not because syncing there is
    illegal, but because those are _dummy_run synthetic activations whose
    routing is not the model's. Excluding them here is what makes the counters
    describe real traffic without any before/after snapshot differencing.
    """
    try:
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX
        if bool(_EXTRA_CTX.capturing) or bool(_EXTRA_CTX.in_profile_run):
            return True
    except Exception:  # noqa: BLE001
        # No forward context, or a runner that does not carry these flags.
        # Unknown state -> assume unsafe.
        return True
    try:
        return bool(torch.npu.is_current_stream_capturing())
    except Exception:  # noqa: BLE001
        return True


def _record_drop_rate(weights: torch.Tensor) -> None:
    s = _stats
    s["calls"] += 1
    # .item() syncs; that is why this is opt-in. Sampling every N calls keeps the
    # cost off the measured path when someone leaves it on by accident.
    if _STATS_ROWS and weights.shape[0] != _STATS_ROWS:
        return
    if s["calls"] % _STATS_EVERY:
        return
    if _unsafe_to_sync():
        return

    kept = int((weights != 0).sum().item())
    total = weights.numel()
    s["win_kept"] += kept
    s["win_total"] += total
    s["cum_kept"] += kept
    s["cum_total"] += total
    s["samples"] += 1
    if not s["win_total"]:
        return

    k = weights.shape[-1]
    win = s["win_kept"] / s["win_total"]
    cum = s["cum_kept"] / s["cum_total"]
    # Per-row spread matters as much as the mean: dummy profiling inputs give
    # every row the SAME kept count (zero variance), real text does not. If
    # min==max on a multi-row batch the sample is probably not real traffic.
    per_row = (weights != 0).sum(-1)
    lo, hi = int(per_row.min().item()), int(per_row.max().item())
    # Report BOTH, each labelled. The window alone is a single sampled call --
    # for a decode step that is one token, 8 slots, so it can only ever read
    # 12.5%, 25.0%, ... Quoting one line as "the drop rate" is wrong, and the
    # previous format invited exactly that by printing a window under the bare
    # label "dropped=". Use the cumulative column; it is the run's answer.
    print(f"[DES-STATS] calls={s['calls']} samples={s['samples']} | "
          f"window: rows={weights.shape[0]} slots={total} "
          f"mean_k={win * k:.3f}/{k} dropped={(1 - win) * 100:.2f}% "
          f"k_range={lo}-{hi} | "
          f"cumulative: slots={s['cum_total']} "
          f"mean_k={cum * k:.3f}/{k} dropped={(1 - cum) * 100:.2f}%",
          flush=True)
    s["win_kept"] = 0
    s["win_total"] = 0


_skip_logged = False


def log_skip_active(where: str, **info) -> None:
    """One-shot log the first time the compute-skip actually engages.

    The skip is silent by construction -- it only changes expert ids and the
    group_list -- so without this there is no way to tell from a server log
    whether it ran, or whether a guard (EP, top_k, quantization) silently
    excluded it. Fires once per process; negligible under graph capture.
    """
    global _skip_logged
    if not _skip_logged:
        _skip_logged = True
        detail = ", ".join(f"{k}={v}" for k, v in info.items())
        print(f"[DES] compute-skip ACTIVE in {where}: {detail}", flush=True)
