# SPDX-License-Identifier: Apache-2.0
"""TIK-kernel DES cutoff backend.

Registers backend name ``tik``. Falls back to ``fused_sorted`` for any shape the
kernel cannot serve, and says so once per process -- a silent fallback would make
a benchmark look like it measured the kernel when it measured torch ops.

SCOPE, and why it is narrow
---------------------------
The kernel compiles row count, threshold, min_k and fake_id into the binary, and
needs rows to be a multiple of 16 (one fp16 vector repeat = 128 lanes = 16 rows
of top_k=8). So:

  * DECODE is served. Captured decode uses batch 1 and 4, which pad to a single
    16-row kernel built once at first use.
  * PREFILL is not. At 10512 rows the per-core slice (10512/8 = 1314) is not
    repeat-aligned, so core splitting degrades to a single core, and the UB
    working set grows with the slice. Large batches therefore fall back.

That is the right coverage for the question this was built to answer: DES at
th=0.9 is +0.86% at prefill already and -0.90% at decode, so decode is where the
remaining gap is.

PER-CALL COST IS THE WHOLE GAME
------------------------------
The first version of this backend was 2.6% SLOWER end to end than the pure-torch
``fused_sorted``, despite the kernel being 5.7x faster in isolation. The kernel
was never the problem: the microbenchmark timed the LAUNCH with tensors already
flat, contiguous and fp16, while the backend had to manufacture those conditions
on every one of 48 calls per decode step -- an int32->fp16 id cast, zeros+cat
padding, two contiguous(), two empty_like allocations, then slice/reshape/cast
back. Roughly 8-10 device ops against fused_sorted's 5 in total.

This version removes that work:

  * WEIGHTS-ONLY KERNEL. The real call path never stamps fake ids (that happens
    later in token_dispatcher, off the weight==0 signal), so ids are not passed
    to the kernel at all -- no cast in, no cast out, no vec_sel inside.
  * PERSISTENT BUFFERS per shape. Input padding buffer and output buffer are
    allocated once and reused, so no zeros/cat/empty_like per call.
  * BUILT PER SHAPE. One kernel per padded row count, compiled on first sight.

What remains per call is a single narrow copy into the padded input buffer, the
launch, and a view of the output. When rows already equal the padded size even
the copy disappears.

BUFFER REUSE IS SAFE ONLY BECAUSE OF STREAM ORDERING: the returned tensor is a
view of a per-shape buffer that the next call overwrites. Every consumer runs on
the same stream between calls, so it is read before it is reused. It would NOT be
safe if a caller retained the result across layers.
"""
import os

import torch

from vllm_ascend_des.des import (des_cutoff_fused_sorted,
                                 register_des_backend)

# One fp16 vector repeat is 128 lanes = 16 rows of top_k=8.
ROWS_QUANTUM = 16
# One dynamic binary per repeat count, built on demand and cached. The ceiling
# is a policy choice, not a kernel limit: raise it and rows up to
# ROWS_QUANTUM*MAX_REPEATS are served, at the cost of one extra compile per new
# repeat count. Above the ceiling the backend falls back to fused_sorted.
MAX_REPEATS = max(1, int(os.environ.get("VLLM_DES_TIK_MAX_REPEATS", "2")))
HARD_MAX_ROWS = ROWS_QUANTUM * MAX_REPEATS          # 32 by default
# CLAMPED, not trusted. Raising this env var above HARD_MAX_ROWS used to admit
# rows the kernel never touches: it moves whole repeats in and out IN PLACE, so
# rows past its compiled width keep their original weights and DES is silently
# not applied to them. No crash, no wrong-looking log line -- and a correctness
# battery still passes, because dropping fewer experts rarely changes an answer.
MAX_TIK_ROWS = min(int(os.environ.get("VLLM_DES_TIK_MAX_ROWS", str(HARD_MAX_ROWS))),
                   HARD_MAX_ROWS)
# One binary for every row count 1..16 and every threshold, instead of one per
# (rows, threshold). Set VLLM_DES_TIK_DYNAMIC=0 to fall back to the per-shape
# build, which is kept so the two can be compared directly.
_DYNAMIC = os.environ.get("VLLM_DES_TIK_DYNAMIC", "1") == "1"

_cache: dict = {}
_logged: set = set()


def _log_once(key: str, msg: str) -> None:
    if key not in _logged:
        _logged.add(key)
        print(f"[DES-TIK] {msg}", flush=True)


def _build_dynamic(device, repeats: int = 1):
    """One binary for every row count up to 16*repeats, every threshold, min_k.

    Replaces the per-(rows, threshold, min_k) build. The lane-masked kernel is a
    single 128-lane repeat, and a vector op costs the same whether 8 lanes are
    enabled or 128 -- so always computing the full 16 rows is free, and the row
    count stops needing to be compiled in. Threshold and min_k arrive in a
    128-lane vector the host prepares (see _threshold_vec).

    This removes the whole stale-artifact hazard by construction: with nothing
    shape-, threshold- or min_k-specific in the binary, there is nothing to
    alias.
    """
    key = ("dyn", repeats)
    if key in _cache:
        return _cache[key]
    from vllm_ascend_des.tik.des_cutoff_tik import (  # noqa: E402
        SHIFTS,
        build_kernel_dynamic,
        offset_table,
    )
    from vllm_ascend_des.tik.tik_torch_wrapper import TikKernel  # noqa: E402
    import numpy as np

    # The repeat count MUST appear in the artifact name. It is the one thing
    # still baked into the binary, and loading a 1-repeat object for a 32-row
    # call would leave half the rows uncut -- the same class of silent wrong
    # answer as the stale-binary bug (results_tik_kernel_loader_bug.md).
    name = f"des_cutoff_dyn_r{repeats}"
    build_kernel_dynamic(kernel_name=name, repeats=repeats)
    meta_dir = os.path.join(os.getcwd(), "kernel_meta")
    obj, js = (os.path.join(meta_dir, name + ext) for ext in (".o", ".json"))
    if not (os.path.exists(obj) and os.path.exists(js)):
        raise RuntimeError(f"BuildCCE produced no artifact for {name} in {meta_dir}")
    kern = TikKernel(obj, js)
    offs = torch.from_numpy(
        np.concatenate([offset_table(s) for s in SHIFTS])).to(device).contiguous()
    _cache[key] = (kern, offs)
    _log_once(f"build-dyn{repeats}",
              f"compiled DYNAMIC kernel repeats={repeats} (any rows<="
              f"{ROWS_QUANTUM * repeats}, any threshold, any min_k) from {obj}")
    return _cache[key]


def _threshold_vec(threshold: float, min_k: int, device, repeats: int = 1):
    """The comparison vector, built once per (threshold, min_k, repeats).

    The kernel DMAs this straight into its comparison buffer. It deliberately
    does NOT read a scalar out of it: `set_as` on a UB location right after a
    DMA forces an MTE2->scalar pipeline barrier, stalling the vector pipe on
    every launch in every layer. Measured e2e, that stall cost ~0.3 ms/step --
    about as much as the entire cutoff. Preparing the full vector on the host
    removes the barrier, the vec_dup and the min_k vec_adds together.

    Cached, so the H2D copy happens once per configuration, never per launch.

    ``repeats`` must match the binary being launched: the kernel DMAs
    ``128*repeats`` lanes out of this buffer, so a vector built for one repeat
    would leave the second repeat's thresholds reading whatever follows it.
    """
    key = ("par", round(threshold, 6), min_k, str(device), repeats)
    if key not in _cache:
        from vllm_ascend_des.tik.des_cutoff_tik import (  # noqa: E402
            LANES,
            threshold_vector,
        )
        _cache[key] = torch.from_numpy(
            threshold_vector(threshold, min_k,
                             lanes=LANES * repeats)).to(device).contiguous()
    return _cache[key]


def _padding_is_ours(t: torch.Tensor, need_elems: int) -> bool:
    """Can we touch ``need_elems`` without stepping on another tensor?

    Two ways to be safe:
      * the storage genuinely has room (a >= 16-row tensor, or a view with slack)
      * the tensor starts its own allocation, in which case the over-touched
        bytes are allocator padding -- the NPU caching allocator has 512-byte
        granularity, so nothing else lives there.
    A view at a non-zero offset with no slack is the dangerous case: the padding
    would belong to a sibling, and we would corrupt it.

    NOTE the padding argument is only good for 512 bytes total. It covers the
    1-repeat kernel outright (256 bytes). For the 2-repeat kernel the caller
    must also have ensured rows > 16, so the tensor itself already fills more
    than half the block -- see the dispatch in des_cutoff_tik().
    """
    try:
        avail = (t.untyped_storage().size() // t.element_size()) - t.storage_offset()
    except Exception:  # noqa: BLE001
        return t.storage_offset() == 0
    return avail >= need_elems or t.storage_offset() == 0


def _build(rows: int, threshold: float, min_k: int, device, dtype):
    """Compile the lane-masked in-place kernel for this EXACT row count.

    Single repeat, only rows*8 lanes enabled, transfers rounded up to a whole
    32-byte block so the router output is read in place with no staging copy.
    No buffers: the kernel writes back over its own input.
    """
    key = (rows, round(threshold, 6), min_k)
    if key in _cache:
        return _cache[key]
    from vllm_ascend_des.tik.des_cutoff_tik import (  # noqa: E402
        SHIFTS,
        build_kernel_small,
        offset_table,
    )
    from vllm_ascend_des.tik.tik_torch_wrapper import TikKernel  # noqa: E402
    import numpy as np

    # THE BINARY MUST BE NAMED FOR ITS PARAMETERS, AND LOADED FROM WHERE IT WAS
    # BUILT. Both were wrong and the two bugs hid each other:
    #
    #   * BuildCCE emits to <cwd>/kernel_meta/<kernel_name>.{o,json}, but this
    #     loaded from a hardcoded /tmp/kernel_meta. Those are different files.
    #     Every "compiled kernel rows=N th=T" log line was therefore a lie: the
    #     binary actually launched was whatever stale artifact happened to sit
    #     in /tmp from an earlier manual run, at an unknown row count and
    #     threshold. A binary built for a different row count reads and writes
    #     the wrong number of lanes, which is what killed skip_tik at th=1.0.
    #   * kernel_name was the constant "des_cutoff_small", so rows and threshold
    #     did not appear in the path at all. Rebuilding for a new threshold
    #     silently reused the previous threshold's binary -- and /tmp persists
    #     across processes, so a th=0.9 server poisoned a later th=1.0 one.
    #
    # Naming the artifact for (rows, threshold, min_k) makes aliasing impossible
    # and makes a stale hit a miss instead of a wrong answer.
    name = f"des_cutoff_r{rows}_t{int(round(threshold * 10000))}_m{min_k}"
    build_kernel_small(rows, threshold, min_k, kernel_name=name)
    meta_dir = os.path.join(os.getcwd(), "kernel_meta")
    obj, js = (os.path.join(meta_dir, name + ext) for ext in (".o", ".json"))
    if not (os.path.exists(obj) and os.path.exists(js)):
        raise RuntimeError(
            f"BuildCCE produced no artifact for {name} in {meta_dir}; "
            f"refusing to launch a stale binary")
    kern = TikKernel(obj, js)
    offs = torch.from_numpy(
        np.concatenate([offset_table(s) for s in SHIFTS])).to(device).contiguous()
    _cache[key] = (kern, offs)
    # Log the artifact path, not just the parameters: the whole point of the bug
    # above was that the parameters said one thing and the loaded binary was
    # another, and the log gave no way to tell.
    _log_once(f"build{key}", f"compiled lane-masked in-place kernel rows={rows} "
                             f"th={threshold} min_k={min_k} from {obj}")
    return _cache[key]


def des_cutoff_tik(topk_weights, topk_ids, threshold, min_k=1, fake_id=None):
    rows, k = topk_weights.shape
    padded = rows          # exact-shape kernel; no padding

    # NOTE: fake_id is None on the real call path. The selector hook calls
    # apply_des_cutoff(w, ids, threshold, min_k) with no fake_id -- the
    # fake-expert stamping happens later in token_dispatcher.py, which reads the
    # weight==0 signal. An earlier version disqualified fake_id=None and so fell
    # back on EVERY call, silently measuring fused_sorted. Only the FALLBACK log
    # revealed it.
    unsupported = (k != 8 or padded > MAX_TIK_ROWS
                   or topk_weights.dtype != torch.float16
                   or not topk_weights.is_npu)
    if unsupported:
        _log_once("fallback",
                  f"FALLBACK to fused_sorted (rows={rows} top_k={k} "
                  f"fake_id={fake_id} dtype={topk_weights.dtype}); the kernel "
                  f"serves top_k=8, fp16, fake_id set, rows<={MAX_TIK_ROWS}")
        return des_cutoff_fused_sorted(topk_weights, topk_ids, threshold, min_k, fake_id)

    # This backend serves the no-stamp path only. fake_id is None on the real
    # call path; if a caller ever does want stamping, fused_sorted handles it.
    if fake_id is not None:
        _log_once("stamp", "FALLBACK: fake_id set, which this weights-only "
                           "kernel does not serve; using fused_sorted")
        return des_cutoff_fused_sorted(topk_weights, topk_ids, threshold, min_k, fake_id)

    if not topk_weights.is_contiguous():
        _log_once("noncontig", "FALLBACK: non-contiguous weights; using fused_sorted")
        return des_cutoff_fused_sorted(topk_weights, topk_ids, threshold, min_k, fake_id)

    # NARROWEST binary that covers this batch. This is not just an efficiency
    # choice: build_kernel_dynamic's padding-safety proof holds only for
    # ceil(rows/ROWS_QUANTUM). Rounding UP to a wider binary would touch past
    # the tensor's own allocation. Rows <= 16 therefore keep the 1-repeat kernel
    # and its original 256-byte footprint untouched.
    repeats = -(-rows // ROWS_QUANTUM)          # ceil

    # The dynamic kernel always touches whole repeats, so a tensor shorter than
    # that relies on the bytes past its end being allocator padding rather than
    # a sibling's data.
    if _DYNAMIC and not _padding_is_ours(topk_weights, ROWS_QUANTUM * repeats * 8):
        _log_once("nopad", "FALLBACK: weights are a view with no slack; the "
                           "dynamic kernel would touch a sibling's bytes")
        return des_cutoff_fused_sorted(topk_weights, topk_ids, threshold, min_k, fake_id)
    try:
        if _DYNAMIC:
            kern, offs = _build_dynamic(topk_weights.device, repeats)
        else:
            kern, offs = _build(padded, threshold, min_k,
                                topk_weights.device, topk_weights.dtype)
    except Exception as exc:  # compilation is the fragile part
        _log_once("buildfail", f"kernel build FAILED ({type(exc).__name__}: "
                               f"{str(exc)[:120]}); using fused_sorted")
        return des_cutoff_fused_sorted(topk_weights, topk_ids, threshold, min_k, fake_id)

    # Keyed by repeats: both binaries can serve in one run (decode 1-repeat,
    # a larger decode batch 2-repeat) and each should announce itself once.
    _log_once(f"active{repeats}",
              f"ACTIVE: rows={rows} top_k={k} repeats={repeats}, lane-masked "
              f"IN PLACE, 1 op, 0 allocations, 0 copies, "
              f"{'dynamic' if _DYNAMIC else 'per-shape'} kernel")

    # In place: input pointer == output pointer. Safe because the kernel completes
    # both data_moves into UB before any compute, so there is no read-after-write
    # hazard. Verified: 20 launches from fresh data, 0 wrong slots, at rows 1/2/4.
    ptr = topk_weights.data_ptr()
    if _DYNAMIC:
        par = _threshold_vec(threshold, min_k, topk_weights.device, repeats)
        kern.launch([ptr, offs.data_ptr(), par.data_ptr(), ptr])
    else:
        kern.launch([ptr, offs.data_ptr(), ptr])
    return topk_weights, topk_ids


try:
    register_des_backend("tik", des_cutoff_tik)
except Exception as exc:  # pragma: no cover
    print(f"[DES-TIK] could not register backend: {exc}", flush=True)
