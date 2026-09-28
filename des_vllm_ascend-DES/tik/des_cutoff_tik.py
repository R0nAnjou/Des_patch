#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""TIK prototype of the DES cutoff kernel (Ascend Python DSL).

Standalone. Run it directly: builds the kernel, then validates against a numpy
oracle with ``tikdb``.

WHY TIK: it is Ascend's closest analogue to Triton -- a Python DSL over tiles
with explicit GM<->UB movement. Verified working in this container (upstream
Triton is not installed; triton-ascend is not on the reachable pip index).
AscendC (C++) is the mainline route for anything upstreamed; TIK is the fast
path to a working prototype.

WHAT IT COMPUTES (the ``assume_sorted`` form the Qwen3 path needs)

    inclusive[j] = sum(w[0..j])                    prefix sum along top_k
    exclusive[j] = inclusive[j] - w[j]
    keep[j]      = exclusive[j] < threshold  or  j < min_k
    out_w[j]     = w[j] if keep[j] else 0
    out_id[j]    = id[j] if keep[j] else fake_id

No sort: on the non-grouped Qwen3 selector the weights already arrive descending
(validated end to end -- byte-identical generated text).

--------------------------------------------------------------------------------
THE CROSS-LANE PROBLEM, AND WHY vgather SOLVES IT
--------------------------------------------------------------------------------
Ascend vector instructions are strictly lane-to-lane: lane i of the destination
reads lane i of each source. A prefix scan needs lane i to read lane i-s. The
obvious encoding -- offsetting an operand tensor by s -- is illegal, because
operand addresses must be 32-byte aligned (16 fp16 elements) and s is 1, 2 or 4:

    Error: Address align error, dst[1] is not 32 Byte align

Padding rows to 16 lanes does not help: it aligns row *starts*, while the shift
is *within* a row. Making the tensor size even does not help either; for fp16 the
quantum is 16 elements, and size was never the constraint.

``vgather(mask, dst, src, src_offset, ...)`` gathers each destination lane from an
arbitrary source address given by an offset tensor. That turns the shift from an
*address* into *data*, so every operand stays base-aligned. Three constant offset
tables give a Hillis-Steele scan over top_k=8:

    dst lane i <- src lane i-s      when (i % 8) >= s
                  (anything)        otherwise, because the lane mask drops it

The lane masks are the same ones the aligned attempt already produced:

    s=1 -> 0xFE   s=2 -> 0xFC   s=4 -> 0xF0     (replicated per 64-lane half)

min_k rides in the threshold vector rather than costing a separate OR: the first
min_k lanes get a large bias so their comparison is unconditionally true.

DTYPE CONSTRAINT: ``vec_sel`` on 310P accepts fp16 only -- both int32 and int16
are rejected by the AiCore. Expert ids therefore travel as fp16, which is
lossless because fp16 represents every integer up to 2048 exactly and expert
counts here are 128-256. A production op would either keep this convention or
pay two ``vec_conv`` casts.

Note also that TIK defaults to Ascend310B; this file pins Ascend310P3 explicitly,
because the wrong target silently changes which instructions are legal.

SPEED IS NOT MEASURED YET. Two separate obstacles:

  * ``StartProfiling`` needs ``libtik_profiling.so``, which ships only inside
    per-chip simulator directories (``simulator/dav_*/lib``). Which ``dav_``
    code corresponds to Ascend310P3 is not something to guess -- profiling
    against the wrong simulator returns confident, meaningless cycle counts.
  * Even a correct cycle count would not compare directly with the torch
    backends' wall-clock milliseconds. A like-for-like number needs the kernel
    callable from PyTorch, i.e. packaged as a custom op (msopgen -> build ->
    install into opp), and then run through the existing backend seam and
    bench_des_cutoff.py like every other backend.

So the deliverable here is a CORRECT kernel, not a faster one. The speed claim
still has to be earned.

CAUTION: vgather does no bounds checking ("programmer should ensure all elements
in src"). A wrong offset table reads neighbouring memory silently rather than
faulting, so the tables are clamped into range and the result is checked against
the numpy oracle on every run.
"""
import argparse

import numpy as np

try:
    from tbe import tik
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"TIK unavailable ({exc}). Needs the CANN toolkit on PYTHONPATH.")

# TIK defaults to Ascend310B, which is NOT this hardware. Building for the wrong
# core silently changes which instructions are accepted (int32 vec_sel, for one),
# so pin it explicitly.
SOC = "Ascend310P3"
try:
    from tbe.common.platform import set_current_compile_soc_info
    set_current_compile_soc_info(SOC)
except Exception:  # older CANN layout
    try:
        from te.platform.cce_conf import te_set_version
        te_set_version(SOC)
    except Exception as exc:  # pragma: no cover
        print(f"WARNING: could not pin SoC to {SOC} ({exc}); build target may be wrong")

TOP_K = 8
LANES = 128                    # one fp16 vector repeat
ROWS_PER_REPEAT = LANES // TOP_K       # 16
SHIFTS = (1, 2, 4)
MIN_K_BIAS = 1024.0            # forces keep; far from fp16's 65504 ceiling
FP16_BYTES = 2


def lane_mask(valid_lanes_per_row: int) -> list:
    """[high64, low64] mask enabling the first ``valid_lanes_per_row`` of each row."""
    byte = sum(1 << i for i in range(valid_lanes_per_row))
    half = 0
    for rep in range(64 // TOP_K):
        half |= byte << (rep * TOP_K)
    return [half, half]


def scan_mask(shift: int) -> list:
    """Lanes that perform the shifted add: those with a valid source in-row.

    With vgather the destination is NOT offset -- lane i is absolute -- so the
    acting lanes are exactly i % TOP_K >= shift, matching offset_table().
    (The earlier aligned attempt offset the destination tensor instead, which
    shifted the lane indices and needed the complementary mask. Mixing the two
    conventions is what produced a wrong keep pattern.)
    """
    byte = sum(1 << i for i in range(shift, TOP_K))
    half = 0
    for rep in range(64 // TOP_K):
        half |= byte << (rep * TOP_K)
    return [half, half]


def offset_table(shift: int) -> np.ndarray:
    """Byte offsets so lane i reads lane i-shift, clamped inside the repeat.

    Lanes that would cross a row boundary are masked out of the add anyway; they
    are pointed at their own lane so the gather can never read out of bounds.
    """
    idx = np.arange(LANES)
    src = np.where((idx % TOP_K) >= shift, idx - shift, idx)
    return (src * FP16_BYTES).astype(np.int32)


def build_kernel(num_rows: int, threshold: float, min_k: int, fake_id: int,
                 kernel_name: str = "des_cutoff", cores: int = 8,
                 with_ids: bool = True):
    """Build the kernel, splitting rows across ``cores`` AI cores.

    with_ids=False builds a WEIGHTS-ONLY kernel: no id input, no id output, no
    id select. The real vLLM call path never stamps fake ids (that happens later
    in token_dispatcher, off the weight==0 signal), so carrying ids through costs
    an int32->fp16 cast on the way in, a cast back on the way out, and a vec_sel
    in the kernel -- all for a result nobody reads.

    Without this the build reports blockDim=1 and the kernel occupies a single
    core while the torch ops it is being compared against use the whole device --
    any speed comparison would be meaningless.
    """
    if num_rows % ROWS_PER_REPEAT:
        raise ValueError(f"num_rows must be a multiple of {ROWS_PER_REPEAT}")
    rows_per_core = num_rows // cores
    while cores > 1 and rows_per_core % ROWS_PER_REPEAT:
        cores //= 2                       # keep each core's slice repeat-aligned
        rows_per_core = num_rows // cores
    total = rows_per_core * TOP_K         # per-core element count
    repeats = total // LANES

    # disable_debug=False is required for tikdb.start_debug to run the kernel
    t = tik.Tik(disable_debug=False)
    w_gm = t.Tensor("float16", (num_rows * TOP_K,), name="w_gm", scope=tik.scope_gm)
    if with_ids:
        id_gm = t.Tensor("float16", (num_rows * TOP_K,), name="id_gm", scope=tik.scope_gm)
    off_gm = t.Tensor("int32", (len(SHIFTS) * LANES,), name="off_gm", scope=tik.scope_gm)
    ow_gm = t.Tensor("float16", (num_rows * TOP_K,), name="ow_gm", scope=tik.scope_gm)
    if with_ids:
        oid_gm = t.Tensor("float16", (num_rows * TOP_K,), name="oid_gm", scope=tik.scope_gm)

    w = t.Tensor("float16", (total,), name="w", scope=tik.scope_ubuf)
    pre = t.Tensor("float16", (total,), name="pre", scope=tik.scope_ubuf)
    tmp = t.Tensor("float16", (LANES,), name="tmp", scope=tik.scope_ubuf)
    thr = t.Tensor("float16", (total,), name="thr", scope=tik.scope_ubuf)
    if with_ids:
        ids = t.Tensor("float16", (total,), name="ids", scope=tik.scope_ubuf)
    sel = t.Tensor("uint16", (total // 16,), name="sel", scope=tik.scope_ubuf)
    off = t.Tensor("int32", (len(SHIFTS) * LANES,), name="off", scope=tik.scope_ubuf)

    core = t.Scalar("int32", name="core")
    with t.for_range(0, cores, block_num=cores) as blk:
        core.set_as(blk)
        base_elem = core * total
        t.data_move(w, w_gm[base_elem], 0, 1, total // 16, 0, 0)
        t.data_move(pre, w_gm[base_elem], 0, 1, total // 16, 0, 0)   # pre := w
        if with_ids:
            t.data_move(ids, id_gm[base_elem], 0, 1, total // 16, 0, 0)
        t.data_move(off, off_gm, 0, 1, len(SHIFTS) * LANES // 8, 0, 0)

        # threshold vector with the min_k bias folded in
        t.vec_dup(LANES, thr, threshold, repeats, 8)
        if min_k > 0:
            t.vec_adds(lane_mask(min_k), thr, thr, MIN_K_BIAS, repeats, 8, 8)

    # ---- masked Hillis-Steele scan, shift expressed via vgather ----
    # Repeats are unrolled at build time so base_addr stays a Python int; rows
    # never straddle a repeat (128 lanes = exactly 16 rows).
        for si, shift in enumerate(SHIFTS):
            m = scan_mask(shift)
            for r in range(repeats):
                base = r * LANES * FP16_BYTES
                t.vgather(LANES, tmp, pre, off[si * LANES], 1, 8, base_addr=base)
                t.vec_add(m, pre[r * LANES], pre[r * LANES], tmp, 1, 8, 8, 8)

    # exclusive = inclusive - w ; keep = exclusive < thr ; apply
        t.vec_sub(LANES, pre, pre, w, repeats, 8, 8, 8)
        t.vec_cmpv_lt(sel, pre, thr, repeats, 8, 8)
    # mode=1 -> src1 is a scalar, so no broadcast buffers are needed at all.
        t.vec_sel(LANES, 1, w, sel, w, 0.0, repeats, 8, 8, 8)
    # vec_sel on 310P supports fp16 ONLY (int32 and int16 both rejected), so
    # expert ids ride as fp16. Lossless: fp16 represents every integer up to 2048
    # exactly, and expert counts here are 128-256.
        if with_ids:
            t.vec_sel(LANES, 1, ids, sel, ids, float(fake_id), repeats, 8, 8, 8)

        t.data_move(ow_gm[base_elem], w, 0, 1, total // 16, 0, 0)
        if with_ids:
            t.data_move(oid_gm[base_elem], ids, 0, 1, total // 16, 0, 0)

    if with_ids:
        t.BuildCCE(kernel_name=kernel_name, inputs=[w_gm, id_gm, off_gm],
                   outputs=[ow_gm, oid_gm])
    else:
        t.BuildCCE(kernel_name=kernel_name, inputs=[w_gm, off_gm], outputs=[ow_gm])
    return t


def active_mask(n_lanes: int) -> list:
    """[high64, low64] enabling exactly the first ``n_lanes`` lanes."""
    if n_lanes <= 64:
        return [0, (1 << n_lanes) - 1]
    return [(1 << (n_lanes - 64)) - 1, (1 << 64) - 1]


def build_kernel_small(num_rows: int, threshold: float, min_k: int,
                       kernel_name: str = "des_cutoff_small"):
    """Single-repeat, lane-masked kernel for num_rows <= 16. Weights only.

    WHY THIS EXISTS. The padded kernel needs >= 16 rows (one 128-lane fp16
    repeat), but captured decode supplies 1 or 4. Padding rows up to 16 cost a
    ``copy_`` into a staging buffer on every call -- and at ~3.13 us per device
    op per layer x 48 layers, an extra op is ~0.5% of a decode step. That copy is
    what stopped the earlier kernel from being a true single-op cutoff.

    Here the row count is baked in and only ``num_rows * TOP_K`` LANES are
    enabled, so the router's output is read in place. One repeat, no padding, no
    staging copy.
    """
    if num_rows > ROWS_PER_REPEAT:
        raise ValueError(f"num_rows must be <= {ROWS_PER_REPEAT}")
    n = num_rows * TOP_K
    act = active_mask(n)
    n_bytes = n * FP16_BYTES
    # data_move works in whole 32-byte blocks. One row (8 fp16 = 16 bytes) is half
    # a block, and data_move_pad is rejected for fp16 on this core. Rather than
    # stage a copy, move a whole block and let the lane mask confine the COMPUTE
    # to the live lanes: the extra bytes touched are allocator padding.
    #
    # SAFE because the NPU caching allocator has 512-byte granularity -- measured:
    # 256 live 16-byte tensors were all 32-byte aligned with a minimum pointer gap
    # of 512 bytes, and 64 live topk outputs likewise. No two tensors share a
    # 32-byte block, so the over-touched bytes belong to nothing else.
    # If that ever changes, this must revert to a staging copy.
    move_bytes = ((n_bytes + 31) // 32) * 32

    t = tik.Tik(disable_debug=False)
    n_gm = move_bytes // FP16_BYTES          # block-rounded, see note above
    w_gm = t.Tensor("float16", (n_gm,), name="w_gm", scope=tik.scope_gm)
    off_gm = t.Tensor("int32", (len(SHIFTS) * LANES,), name="off_gm", scope=tik.scope_gm)
    ow_gm = t.Tensor("float16", (n_gm,), name="ow_gm", scope=tik.scope_gm)

    # UB buffers are a full repeat wide so gathers can never step outside them,
    # even though only n lanes are enabled.
    w = t.Tensor("float16", (LANES,), name="w", scope=tik.scope_ubuf)
    pre = t.Tensor("float16", (LANES,), name="pre", scope=tik.scope_ubuf)
    tmp = t.Tensor("float16", (LANES,), name="tmp", scope=tik.scope_ubuf)
    thr = t.Tensor("float16", (LANES,), name="thr", scope=tik.scope_ubuf)
    sel = t.Tensor("uint16", (LANES // 16,), name="sel", scope=tik.scope_ubuf)
    off = t.Tensor("int32", (len(SHIFTS) * LANES,), name="off", scope=tik.scope_ubuf)

    # n*2 bytes is not always a whole 32-byte block (1 row = 16 bytes), so the
    # block-granular data_move cannot express it; data_move_pad is byte-granular.
    t.data_move(w, w_gm, 0, 1, move_bytes // 32, 0, 0)
    t.data_move(pre, w_gm, 0, 1, move_bytes // 32, 0, 0)
    t.data_move(off, off_gm, 0, 1, len(SHIFTS) * LANES // 8, 0, 0)

    t.vec_dup(act, thr, threshold, 1, 8)
    if min_k > 0:
        bias = 0
        for r in range(num_rows):
            for i in range(min_k):
                bias |= 1 << (r * TOP_K + i)
        mk = [bias >> 64, bias & ((1 << 64) - 1)]
        t.vec_adds(mk, thr, thr, MIN_K_BIAS, 1, 8, 8)

    for si, shift in enumerate(SHIFTS):
        m_full = scan_mask(shift)
        m = [m_full[0] & act[0], m_full[1] & act[1]]     # stay inside live lanes
        t.vgather(act, tmp, pre, off[si * LANES], 1, 8, base_addr=0)
        t.vec_add(m, pre, pre, tmp, 1, 8, 8, 8)

    t.vec_sub(act, pre, pre, w, 1, 8, 8, 8)
    t.vec_cmpv_lt(sel, pre, thr, 1, 8, 8)
    t.vec_sel(act, 1, w, sel, w, 0.0, 1, 8, 8, 8)

    t.data_move(ow_gm, w, 0, 1, move_bytes // 32, 0, 0)

    t.BuildCCE(kernel_name=kernel_name, inputs=[w_gm, off_gm], outputs=[ow_gm])
    return t


def threshold_vector(threshold: float, min_k: int, lanes: int = LANES) -> np.ndarray:
    """The comparison vector the dynamic kernel expects as ``par_gm``.

    ``lanes`` must match the kernel's repeat count: ``LANES`` for the 1-repeat
    binary (rows <= 16), ``2 * LANES`` for the 2-repeat one (rows <= 32). Passing
    a short vector would leave the second repeat's thresholds undefined, which
    the kernel cannot detect.

    Every lane holds ``threshold``. The first ``min_k`` lanes of each 8-lane row
    additionally carry MIN_K_BIAS, which pushes them past any reachable prefix
    sum so they are always kept. Doing this on the host is what lets the kernel
    drop its vec_dup and vec_adds -- and what makes min_k a runtime parameter
    rather than a compile-time constant.

    Cheap to build and meant to be cached per (threshold, min_k) by the caller;
    it must not be rebuilt per launch.
    """
    v = np.full(lanes, threshold, dtype=np.float16)
    if min_k > 0:
        lane = np.arange(lanes) % TOP_K
        v[lane < min_k] += np.float16(MIN_K_BIAS)
    return v


def build_kernel_dynamic(min_k: int = 1, kernel_name: str = "des_cutoff_dyn",
                         repeats: int = 1):
    """ONE binary for every row count up to 16*repeats, every threshold, min_k.

    ``min_k`` is accepted only for call-site compatibility and is NOT used: it
    now travels in the parameter vector (see threshold_vector). Passing a
    different value here produces an identical binary.

    WHY THIS IS FREE
    The lane-masked kernel is already a single 128-lane fp16 repeat. A vector op
    processes all 128 lanes in that repeat whether 8 of them are enabled or 128 --
    the mask disables lanes, it does not shorten the work. So always computing the
    full 16 rows costs exactly what computing 1 row costs, and the row count stops
    needing to be a compile-time constant.

    HOW THE PARAMETERS GET IN, AND WHY NOT THE OBVIOUS WAY
    The first version read the threshold as a scalar (`set_as(par[0])`) and
    broadcast it with vec_dup. That cost far more than it looked: a scalar read
    out of UB immediately after a DMA forces an MTE2->scalar pipeline barrier, so
    the vector pipe stalls waiting for the transfer on every launch, in every one
    of the 48 layers. Measured e2e, it cut the decode gain from ~+4.4% to +3.3%
    -- roughly doubling the cutoff's device cost.

    So the host prepares the comparison vector instead: all 128 lanes already
    hold the threshold, with the min_k bias already added to the lanes that need
    it. The kernel just DMAs it into place. No scalar read, no barrier, no
    vec_dup, no vec_adds -- strictly FEWER ops than the per-shape kernel, and
    min_k stops being a compile-time constant as a side effect.

    WHAT IS STILL BAKED IN
    Only the repeat count: this binary serves rows 1..``16*repeats``. Everything
    else -- row count, threshold, min_k -- is runtime.

    WHY NOT ONE BINARY WITH A RUNTIME REPEAT COUNT
    The vector ops would accept a Scalar repeat_times, but the scan would not:
    the vgather needs a compile-time ``base_addr`` per repeat, which is why
    build_kernel_padded unrolls its repeats too. Unrolling to a maximum and
    masking the unused repeats would work arithmetically but would make every
    launch touch the full span, breaking the SAFETY argument below for small
    batches. So the caller compiles one binary per repeat count -- cached, so
    each is built once -- and dispatches on the actual row count.

    SAFETY: reading and writing whole repeats when the caller's tensor holds
    fewer rows touches allocator padding. That is sound because the NPU caching
    allocator has 512-byte granularity (measured by verify_alloc_granularity.py:
    256 live 16-byte tensors, all 32-byte aligned, minimum pointer gap 512
    bytes), so the bytes past a tensor's end belong to that tensor's own
    allocation.

    The guarantee holds for ANY repeat count PROVIDED the caller picks the
    narrowest binary that fits, i.e. ``repeats = ceil(rows / 16)``. Then:

        touched   = ceil(rows/16) * 256 bytes
        allocated = ceil(rows*16 / 512) * 512 = ceil(rows/32) * 512 bytes

    and since ``ceil(rows/16) <= 2*ceil(rows/32)``, touched <= allocated for
    every row count. Over-touch is always under one repeat (< 256 bytes).
    Choosing a WIDER binary than the row count needs breaks this -- that is the
    rule the backend's dispatch must not relax.

    The BACKEND must still refuse tensors that are views into a larger storage,
    where the padding would belong to a sibling -- see des_tik.py.

    Garbage in the unused rows cannot contaminate the live ones: the scan is
    lane-masked within each 8-lane group, so rows are independent.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    n = ROWS_PER_REPEAT * TOP_K * repeats    # always whole repeats
    # Per-repeat lane mask: every lane of one repeat. The repeat COUNT is passed
    # separately to each vector op; the mask is not widened.
    act = active_mask(LANES)
    move_bytes = n * FP16_BYTES              # 256 or 512, whole blocks either way

    t = tik.Tik(disable_debug=False)
    n_gm = move_bytes // FP16_BYTES
    w_gm = t.Tensor("float16", (n_gm,), name="w_gm", scope=tik.scope_gm)
    off_gm = t.Tensor("int32", (len(SHIFTS) * LANES,), name="off_gm", scope=tik.scope_gm)
    # 16 fp16 = 32 bytes = one block, the smallest legal data_move. Only [0] is
    # used today; the rest is room for future runtime parameters.
    # The comparison vector, arriving ready to use: every lane already holds the
    # threshold, and the lanes covered by min_k already carry the keep bias. See
    # threshold_vector(), which is what the host must pass here.
    par_gm = t.Tensor("float16", (LANES * repeats,), name="par_gm", scope=tik.scope_gm)
    ow_gm = t.Tensor("float16", (n_gm,), name="ow_gm", scope=tik.scope_gm)

    w = t.Tensor("float16", (LANES * repeats,), name="w", scope=tik.scope_ubuf)
    pre = t.Tensor("float16", (LANES * repeats,), name="pre", scope=tik.scope_ubuf)
    # One repeat wide: it is the gather destination, written per repeat.
    tmp = t.Tensor("float16", (LANES,), name="tmp", scope=tik.scope_ubuf)
    thr = t.Tensor("float16", (LANES * repeats,), name="thr", scope=tik.scope_ubuf)
    sel = t.Tensor("uint16", (LANES * repeats // 16,), name="sel", scope=tik.scope_ubuf)
    off = t.Tensor("int32", (len(SHIFTS) * LANES,), name="off", scope=tik.scope_ubuf)

    t.data_move(w, w_gm, 0, 1, move_bytes // 32, 0, 0)
    t.data_move(pre, w_gm, 0, 1, move_bytes // 32, 0, 0)
    t.data_move(off, off_gm, 0, 1, len(SHIFTS) * LANES // 8, 0, 0)
    t.data_move(thr, par_gm, 0, 1, LANES * repeats * FP16_BYTES // 32, 0, 0)

    # Masked Hillis-Steele scan. The gather is unrolled per repeat so base_addr
    # stays a Python int, exactly as build_kernel_padded does it; rows never
    # straddle a repeat (128 lanes = exactly 16 rows), so the scan stays inside
    # each 8-lane row and the repeats are independent.
    for si, shift in enumerate(SHIFTS):
        m_full = scan_mask(shift)
        m = [m_full[0] & act[0], m_full[1] & act[1]]
        for r in range(repeats):
            base = r * LANES * FP16_BYTES
            t.vgather(act, tmp, pre, off[si * LANES], 1, 8, base_addr=base)
            t.vec_add(m, pre[r * LANES], pre[r * LANES], tmp, 1, 8, 8, 8)

    t.vec_sub(act, pre, pre, w, repeats, 8, 8, 8)
    t.vec_cmpv_lt(sel, pre, thr, repeats, 8, 8)
    t.vec_sel(act, 1, w, sel, w, 0.0, repeats, 8, 8, 8)

    t.data_move(ow_gm, w, 0, 1, move_bytes // 32, 0, 0)

    t.BuildCCE(kernel_name=kernel_name, inputs=[w_gm, off_gm, par_gm], outputs=[ow_gm])
    return t


def reference(w, ids, threshold, min_k, fake_id):
    inc = np.cumsum(w.astype(np.float32), axis=-1)
    exc = inc - w.astype(np.float32)
    keep = exc < threshold
    keep[:, :min_k] = True
    return (w * keep).astype(np.float16), np.where(keep, ids, fake_id).astype(np.float16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--min-k", type=int, default=1)
    ap.add_argument("--fake-id", type=int, default=128)
    ap.add_argument("--profile", action="store_true",
                    help="run StartProfiling to get on-device cycle counts")
    args = ap.parse_args()

    print("=" * 72)
    print(f"TIK DES cutoff (vgather scan): rows={args.rows} top_k={TOP_K} "
          f"th={args.threshold} min_k={args.min_k}")
    print("=" * 72)
    for s in SHIFTS:
        tbl = offset_table(s)
        print(f"  shift {s}: mask 0x{scan_mask(s)[0]:016X}  "
              f"offsets[0:10]={(tbl[:10] // FP16_BYTES).tolist()}")

    t = build_kernel(args.rows, args.threshold, args.min_k, args.fake_id)
    print(f"BuildCCE OK (blockDim should now be > 1)")

    rng = np.random.default_rng(0)
    logits = rng.standard_normal((args.rows, TOP_K)).astype(np.float32) * 1.8
    e = np.exp(logits - logits.max(-1, keepdims=True))
    w = (e / e.sum(-1, keepdims=True)).astype(np.float16)
    w = -np.sort(-w, axis=-1)                       # descending, as the router emits
    ids = rng.integers(0, 128, (args.rows, TOP_K)).astype(np.float16)
    offs = np.concatenate([offset_table(s) for s in SHIFTS])

    feed = {"w_gm": w.reshape(-1), "id_gm": ids.reshape(-1), "off_gm": offs}
    out = t.tikdb.start_debug(feed_dict=feed, interactive=False)
    got_w = np.array(out[0]).reshape(args.rows, TOP_K)
    got_id = np.array(out[1]).reshape(args.rows, TOP_K)

    ref_w, ref_id = reference(w, ids, args.threshold, args.min_k, args.fake_id)
    diff = (got_w != ref_w) | (got_id != ref_id)
    n_diff = int(diff.sum())

    # The oracle accumulates in fp32, the kernel in fp16. Differences are only
    # meaningful if they are NOT borderline: classify each by how far that slot's
    # exclusive sum sat from the threshold. Inside fp16 epsilon the keep/drop
    # call is arbitrary and neither answer is more correct.
    exc = np.cumsum(w.astype(np.float32), -1) - w.astype(np.float32)
    dist = np.abs(exc - args.threshold)
    worst = float(dist[diff].max()) if n_diff else 0.0
    borderline = worst <= 5e-3

    print(f"\ncorrectness vs numpy oracle:")
    print(f"  slots differing:  {n_diff}/{w.size} ({100.0*n_diff/w.size:.3f}%)")
    print(f"  kept slots: kernel {int((got_w != 0).sum())}  oracle {int((ref_w != 0).sum())}")
    if n_diff:
        print(f"  worst differing slot sat {worst:.2e} from the threshold "
              f"({'borderline -- fp16 rounding' if borderline else 'NOT borderline'})")
        r0 = int(np.argmax(diff.any(axis=-1)))
        e0 = exc[r0]
        print(f"  example row {r0}: exclusive sums {[round(float(x), 5) for x in e0]}")
        print(f"    kernel {got_w[r0].tolist()}")
        print(f"    oracle {ref_w[r0].tolist()}")
    if args.profile:
        print("\n--- StartProfiling ---")
        try:
            t.StartProfiling(feed_dict=feed, simulatorlog_path="/tmp/tik_prof")
            print("  profiling completed; see /tmp/tik_prof")
        except Exception as exc:
            print(f"  profiling failed: {type(exc).__name__}: {str(exc)[:200]}")

    ok = (n_diff == 0) or borderline
    print("\nVERDICT:", "kernel matches the oracle" if n_diff == 0 else
          ("kernel correct; differences are fp16-borderline only" if ok else
           "MISMATCH -- real disagreement"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
