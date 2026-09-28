#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Call the TIK DES cutoff kernel on torch NPU tensors, inside torch's stream.

This is what makes the kernel testable under aclgraph capture: the launch is
issued on the *current torch stream*, so it is recorded like any other device op
rather than running beside the graph.

MECHANISM (ctypes over libruntime.so, following CANN's own rts_api.py):

    rtDevBinaryRegister(&rtDevBinary_t{magic, version, data, length}, &handle)
    rtFunctionRegister(handle, name, name, name, func_mode)   -> stub pointer
    rtKernelLaunch(stub, blockDim, args, argsSize, smDesc, stream)

``args`` is a flat array of uint64 device pointers, one per kernel parameter, in
the order BuildCCE declared them (inputs then outputs). torch tensors are already
on device, so their ``data_ptr()`` values go straight in -- no host copies, which
also keeps the timing comparable with the torch backends.

Why not the higher-level routes:
  * msopgen custom-op packaging is only needed to make an op callable from a
    graph *definition*; for calling and measuring a kernel it is unnecessary.
  * CANN's single-op runner (msopst) has the right API but is unimportable in
    this build (circular import in msopst.runtime); the msaicerr copy imports but
    is the AI-core-error variant with a different API.
  * The acl python bindings expose the same primitives but no signatures.
    Probing them by trial segfaulted the process, which is not an acceptable way
    to explore native APIs on a shared NPU.

ASCII only.
"""
import ctypes
import os
import sys

CANN = os.environ.get("CANN_HOME", "/usr/local/Ascend/cann-9.1.0-beta.1")
sys.path.insert(0, CANN + "/tools/msaicerr")

import torch  # noqa: E402
import torch_npu  # noqa: E402


class rtDevBinary_t(ctypes.Structure):
    _fields_ = [("magic", ctypes.c_uint32),
                ("version", ctypes.c_uint32),
                ("data", ctypes.c_char_p),
                ("length", ctypes.c_uint64)]


def _find_runtime() -> str:
    for p in (os.path.join(CANN, "runtime/lib64/libruntime.so"),
              os.path.join(CANN, "aarch64-linux/lib64/libruntime.so"),
              "/usr/local/Ascend/latest/runtime/lib64/libruntime.so"):
        if os.path.exists(p):
            return p
    for root in (CANN, "/usr/local/Ascend"):
        for dirpath, _, names in os.walk(root):
            if "libruntime.so" in names:
                return os.path.join(dirpath, "libruntime.so")
    raise RuntimeError("libruntime.so not found")


def _magic() -> int:
    from ms_interface.single_op_test_frame.runtime import rts_info
    return rts_info.MAGIC_MAP["RT_DEV_BINARY_MAGIC_ELF"]


def _torch_stream_ptr() -> int:
    """Raw aclrtStream for torch's current NPU stream.

    Launching on torch's stream is the whole point: it orders the kernel against
    surrounding torch ops and lets aclgraph capture record it.
    """
    st = torch.npu.current_stream()
    for attr in ("npu_stream", "_as_parameter_", "cuda_stream", "stream"):
        v = getattr(st, attr, None)
        if isinstance(v, int) and v:
            return v
    raise RuntimeError(f"cannot get raw stream pointer from {type(st)}: "
                       f"{[a for a in dir(st) if not a.startswith('__')][:12]}")


class TikKernel:
    """A BuildCCE-produced kernel, registered once and launchable many times."""

    _rts = None

    def __init__(self, bin_path: str, json_path: str):
        import json
        if TikKernel._rts is None:
            TikKernel._rts = ctypes.CDLL(_find_runtime())
        self.rts = TikKernel._rts
        meta = json.load(open(json_path))
        self.block_dim = int(meta["blockDim"])
        self.kernel_name = meta["kernelName"]

        with open(bin_path, "rb") as fh:
            blob = fh.read()
        # keep references alive for the process lifetime: the runtime holds these
        # pointers, and letting Python free them corrupts the device binary
        self._blob = ctypes.c_char_p(blob)
        self._binary = rtDevBinary_t(magic=_magic(), version=0,
                                     data=self._blob, length=len(blob))
        self._handle = ctypes.c_void_p()
        self.rts.rtDevBinaryRegister.restype = ctypes.c_uint64
        err = self.rts.rtDevBinaryRegister(
            ctypes.c_void_p(ctypes.addressof(self._binary)),
            ctypes.c_void_p(ctypes.addressof(self._handle)))
        if err:
            raise RuntimeError(f"rtDevBinaryRegister failed: 0x{err:x}")

        self._name_p = ctypes.c_char_p(self.kernel_name.encode())
        self.rts.rtFunctionRegister.restype = ctypes.c_uint64
        err = self.rts.rtFunctionRegister(self._handle, self._name_p, self._name_p,
                                          self._name_p, ctypes.c_uint32(0))
        if err:
            raise RuntimeError(f"rtFunctionRegister failed: 0x{err:x}")
        self.rts.rtKernelLaunch.restype = ctypes.c_uint64

    def launch(self, ptrs, block_dim=None, stream=None):
        """Launch with a list of device pointers (ints), on torch's stream."""
        n = len(ptrs)
        arr = (ctypes.c_uint64 * n)(*[int(p) for p in ptrs])
        err = self.rts.rtKernelLaunch(
            self._name_p,
            ctypes.c_uint32(self.block_dim if block_dim is None else block_dim),
            ctypes.c_void_p(ctypes.addressof(arr)),
            ctypes.c_uint32(n * 8),
            ctypes.c_void_p(0),
            ctypes.c_uint64(_torch_stream_ptr() if stream is None else stream))
        if err:
            raise RuntimeError(f"rtKernelLaunch failed: 0x{err:x}")


_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = TikKernel("/tmp/kernel_meta/des_cutoff.o",
                            "/tmp/kernel_meta/des_cutoff.json")
    return _KERNEL


def des_cutoff_tik_into(weights: torch.Tensor, ids: torch.Tensor,
                        offsets: torch.Tensor, out_w: torch.Tensor,
                        out_id: torch.Tensor) -> None:
    """Write into caller-provided buffers -- the form graph capture needs.

    Capture records the launch with whatever addresses it sees, so the output
    tensors must be stable across replays. Allocating fresh outputs inside the
    captured region would bake in addresses from the capture pool and quietly
    return stale data on replay.
    """
    _kernel().launch([weights.data_ptr(), ids.data_ptr(), offsets.data_ptr(),
                      out_w.data_ptr(), out_id.data_ptr()])


def des_cutoff_tik(weights: torch.Tensor, ids: torch.Tensor,
                   offsets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """DES cutoff via the TIK kernel. Inputs must be contiguous NPU tensors.

    weights/ids: [rows*top_k] float16 (ids carried as fp16 -- vec_sel on 310P
    accepts fp16 only). offsets: [3*128] int32 gather tables.
    """
    out_w = torch.empty_like(weights)
    out_id = torch.empty_like(ids)
    _kernel().launch([weights.data_ptr(), ids.data_ptr(), offsets.data_ptr(),
                      out_w.data_ptr(), out_id.data_ptr()])
    return out_w, out_id
