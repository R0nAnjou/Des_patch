# Installing DES into a fresh vllm-ascend container

Dynamic Expert Selection (DES) on Ascend 310P3, vllm-ascend 0.23.0rc1.

Every command below was executed end to end in a container created from the
stock image; see "Verification" for what was checked and "Known pitfalls" for
the three things that actually go wrong.

## What DES is, in one paragraph

Per token, the router's renormalized top-k weights are sorted descending and a
cumulative-mass cutoff is applied: an expert is kept while the cumulative sum
*before* it is `< threshold`, so the expert that crosses the threshold is kept
and everything after it is dropped. Surviving weights are NOT renormalized, and
`min_k` experts are always kept. With `VLLM_DES_SKIP=1` the dropped slots are
routed to a fake expert id so the grouped matmul skips them; with `SKIP=0` their
weight is merely zeroed (same numerics, no compute saving).

## What gets installed

Six files, all inside `vllm_ascend`. The Ascend path is self-contained -- it does
NOT import the GPU-side `vllm/model_executor/layers/fused_moe/des.py`, so a pure
Ascend install does not need the vLLM-side patch at all. All the files are in `vllm_ascend` directory in this repository.

| file | kind |
|---|---|
| `vllm_ascend/ops/fused_moe/des.py` | new -- config, backends, cutoff |
| `vllm_ascend/ops/fused_moe/des_tik.py` | new -- TIK kernel backend |
| `vllm_ascend/ops/fused_moe/experts_selector.py` | hook (additive) |
| `vllm_ascend/ops/fused_moe/token_dispatcher.py` | hook |
| `vllm_ascend/_310p/fused_moe/experts_selector.py` | hook (additive) |
| `vllm_ascend/_310p/fused_moe/token_dispatcher.py` | hook |

Plus two helper modules for the `tik` backend, which live OUTSIDE the package
(default `/root`, override with `VLLM_DES_TIK_DIR`). All the files can be found in `tik` directory in the repository:

| file | purpose |
|---|---|
| `des_cutoff_tik.py` | builds the TIK kernel via BuildCCE |
| `tik_torch_wrapper.py` | ctypes launcher (rtDevBinaryRegister/rtKernelLaunch) |

### Why copying whole files is safe here

The DES delta against the image's upstream is purely additive except for two
lines in each `token_dispatcher.py` (`expert_num` and `active_expert_range`,
which DES has to widen by one for the fake expert). Verified by diffing every
hook file against the stock image and listing every removed line:

```
vllm_ascend/ops/fused_moe/experts_selector.py    (no lines removed)
vllm_ascend/ops/fused_moe/token_dispatcher.py    -expert_num=global_num_experts
                                                 -active_expert_range=[first, last]
vllm_ascend/_310p/fused_moe/experts_selector.py  (no lines removed)
vllm_ascend/_310p/fused_moe/token_dispatcher.py  -expert_num=self.num_experts_local
                                                 -active_expert_range=[first, last]
```

If you are installing onto a DIFFERENT vllm-ascend build than
`v0.23.0rc1-310p`, re-run that diff before copying -- wholesale copy would
silently revert any upstream change in those four files.

## 1. Create the container

```bash
docker run -itd --name des --network host --privileged --shm-size=32g \
  --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci2 \
  --device /dev/davinci3 --device /dev/davinci4 --device /dev/davinci5 \
  --device /dev/davinci6 --device /dev/davinci7 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/driver \
  -v /usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/common \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /etc/vnpu.cfg:/etc/vnpu.cfg \
  -v /home:/home-host -v /root:/root-host \
  quay.io/ascend/vllm-ascend:v0.23.0rc1-310p bash
```

`--privileged` is REQUIRED. Passing only the two `/dev/davinci*` devices you
intend to use, without `--privileged`, yields `torch.npu.device_count() == 0`.
Expose all devices here and restrict per run with `ASCEND_RT_VISIBLE_DEVICES`.

Check it worked:

```bash
docker exec des python3 -c "import torch, torch_npu; print(torch.npu.device_count())"
# expect: 8
```

The stock image already contains vLLM 0.23.0 and vllm-ascend 0.23.0rc1 as
editable source checkouts under `/vllm-workspace`, so editing files there takes
effect with no reinstall:

```bash
docker exec des python3 -c "import vllm_ascend, os; print(os.path.dirname(vllm_ascend.__file__))"
# expect: /vllm-workspace/vllm-ascend/vllm_ascend
```

## 2. Copy the DES files in

From the repo root on your workstation:

```bash
tar czf des_src.tgz -C vllm-ascend \
  vllm_ascend/ops/fused_moe/des.py \
  vllm_ascend/ops/fused_moe/des_tik.py \
  vllm_ascend/ops/fused_moe/experts_selector.py \
  vllm_ascend/ops/fused_moe/token_dispatcher.py \
  vllm_ascend/_310p/fused_moe/experts_selector.py \
  vllm_ascend/_310p/fused_moe/token_dispatcher.py
tar czf des_kernels.tgz -C DES/kernels des_cutoff_tik.py tik_torch_wrapper.py

scp -P <port> des_src.tgz des_kernels.tgz root@<server>:/tmp/
```

On the server:

```bash
docker cp /tmp/des_src.tgz des:/tmp/
docker cp /tmp/des_kernels.tgz des:/tmp/
docker exec des bash -c '
  mkdir -p /tmp/des_src && tar xzf /tmp/des_src.tgz -C /tmp/des_src
  A=/vllm-workspace/vllm-ascend
  cp /tmp/des_src/vllm_ascend/ops/fused_moe/des.py               $A/vllm_ascend/ops/fused_moe/
  cp /tmp/des_src/vllm_ascend/ops/fused_moe/des_tik.py           $A/vllm_ascend/ops/fused_moe/
  cp /tmp/des_src/vllm_ascend/ops/fused_moe/experts_selector.py  $A/vllm_ascend/ops/fused_moe/
  cp /tmp/des_src/vllm_ascend/ops/fused_moe/token_dispatcher.py  $A/vllm_ascend/ops/fused_moe/
  cp /tmp/des_src/vllm_ascend/_310p/fused_moe/experts_selector.py $A/vllm_ascend/_310p/fused_moe/
  cp /tmp/des_src/vllm_ascend/_310p/fused_moe/token_dispatcher.py $A/vllm_ascend/_310p/fused_moe/
  tar xzf /tmp/des_kernels.tgz -C /root
'
```

Import check (DES off is the correct default):

```bash
docker exec des python3 -c \
  "import vllm_ascend.ops.fused_moe.des as d; print(d.DES)"
# expect: DESConfig(enabled=False, threshold=1.0, min_k=1, skip=False, backend='torch')
```

## 3. Configuration

All read once at import; restart the server to change any of them.

| variable | default | meaning |
|---|---|---|
| `VLLM_DES_ENABLE` | `0` | `1` turns DES on |
| `VLLM_DES_THRESHOLD` | `1.0` | cumulative-mass cutoff; `1.0` drops nothing |
| `VLLM_DES_MIN_K` | `1` | always keep at least this many experts |
| `VLLM_DES_SKIP` | `0` | `1` = skip dropped experts' compute (the speed win) |
| `VLLM_DES_BACKEND` | `torch` | `torch`, `triu`, `fused`, `fused_sorted`, `tik` |
| `VLLM_DES_TIK_DIR` | `/root` | where the two TIK helper modules live |
| `VLLM_DES_TIK_DYNAMIC` | `1` | one kernel binary for all shapes/thresholds |
| `VLLM_DES_STATS` | `0` | log the measured drop rate (see pitfalls) |

Backends are all numerically identical; `torch` is the reference oracle and
`tik` is the fastest. `tik` serves at most 16 rows and falls back to
`fused_sorted` above that -- see "The 16-row ceiling".

## 4. Run

Fastest version uses TIK based kernel:

```bash
docker exec -d des bash -c 'cd /root && \
  ASCEND_RT_VISIBLE_DEVICES=6,7 \
  VLLM_DES_ENABLE=1 VLLM_DES_THRESHOLD=0.9 VLLM_DES_MIN_K=1 \
  VLLM_DES_SKIP=1 VLLM_DES_BACKEND=tik \
  vllm serve /home-host/models/Qwen3-30B-A3B \
    --tensor-parallel-size 2 --dtype float16 \
    --host 127.0.0.1 --port 3099 --served-model-name M \
    --max-model-len 4096 --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.85 --no-enable-prefix-caching \
    --compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[1,4]}" \
  > /root/serve.log 2>&1 &'
```

`--dtype float16` is required: the 310P has no bf16.

vLLM prints `Unknown vLLM environment variable detected: VLLM_DES_*` for each of
these. That warning is expected and harmless -- vLLM validates its own env
namespace and DES reads these directly from `os.environ`.

## 5. Verification

Do not assume DES is active because the server started. Check all three:

```bash
# 1. DES is on, with the settings you asked for
docker exec des grep -m1 "Dynamic Expert Selection ENABLED" /root/serve.log
# [DES] Dynamic Expert Selection ENABLED (vllm-ascend):
#   DESConfig(enabled=True, threshold=0.9, min_k=1, skip=True, backend='tik')

# 2. the compute-skip path is engaged (only with VLLM_DES_SKIP=1)
docker exec des grep -m1 "compute-skip ACTIVE" /root/serve.log
# [DES] compute-skip ACTIVE in TokenDispatcherWithAllGather310: top_k=8, ...

# 3. the TIK kernel is really running, not silently falling back
docker exec des grep -E "DES-TIK" /root/serve.log | head -4
# [DES-TIK] compiled DYNAMIC kernel (any rows<=16, any threshold, any min_k)
#           from /root/kernel_meta/des_cutoff_dyn.o
# [DES-TIK] ACTIVE: rows=4 top_k=8, lane-masked IN PLACE, 1 op, ...
```

Then confirm the model actually answers:

```bash
docker exec des bash -c 'curl -s http://127.0.0.1:3099/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"M\",\"prompt\":\"2+2=\",\"max_tokens\":16,\"temperature\":0}"'
```

### Proving DES changes the output

The strongest check, and the one this project learned the hard way: send the
same greedy prompt to a DES-enabled server and a `VLLM_DES_ENABLE=0` server and
compare the text. If they are identical, DES is not firing. A stale kernel
binary once produced output bit-identical across thresholds while every log line
said DES was active -- logs alone are not proof.

Observed when this document was validated (`temperature=0, seed=0`, prompt
`"Question: What is 17 times 4? Answer:"`):

| config | output | sha256[:16] |
|---|---|---|
| `ENABLE=1 THRESHOLD=0.9 SKIP=1 BACKEND=tik` | ` 17 times 4 is 68. 17 * 4 = 68. The answer is 68.` | `0e71a12515061e10` |
| `ENABLE=0` | ` 17 times 4 is 68. So the answer is 68.` | `2aa8a3358851199a` |

Different text, same correct answer -- which is exactly what a working install
should look like: DES perturbs the computation without breaking it.

## Known pitfalls

**`--privileged` is mandatory.** Without it, `torch.npu.device_count()` returns
0 even with the `/dev/davinci*` devices passed through.

**`VLLM_DES_STATS=1` will kill the engine under aclgraph.** It calls `.item()`,
a device-to-host sync, which is illegal during graph capture
(`LocalScalarDenseNpu` inside `AclmdlRICapture`). Use `VLLM_DES_STATS_ROWS` to
restrict sampling to a row count that only occurs outside capture, or run with
`cudagraph_mode=NONE`. `torch.npu.is_current_stream_capturing()` does NOT
detect vllm-ascend's capture and is not a usable guard.

**Restarting on the same chips needs a real wait, not `sleep 8`.** NPU memory is
not reclaimed the instant the process dies. Killing a server and immediately
starting another on the same chips fails with `Free memory on device (9.52/43.24
GiB) on startup is less than desired GPU memory utilization`. Poll `npu-smi
info` until the chips are back at their ~1.5 GiB baseline before relaunching.

**Only two decode graphs can be captured on this 310P3.** Any
`cudagraph_capture_sizes` list with three or more entries dies with
`capture_end 507903` / `Resource_Busy` on the third graph, regardless of the
sizes and with DES disabled. It is not a memory problem: lowering
`gpu-memory-utilization` makes it worse (below ~0.85 there is no KV cache at
all), and freeing 0.65 GiB by shrinking `max-num-batched-tokens` bought no third
graph.

**The 16-row ceiling.** The TIK kernel serves at most 16 rows, and rows == batch
size at decode, so above 16 concurrent sequences it falls back to
`fused_sorted`. This costs less than it sounds: the kernel's advantage is a
roughly constant ~1.1 ms/step, so it is worth ~4% at batch 1-4 and is already
inside the noise by batch 16. DES itself keeps gaining at larger batch.

## Uninstall

```bash
docker exec des bash -c '
  cd /vllm-workspace/vllm-ascend
  rm -f vllm_ascend/ops/fused_moe/des.py vllm_ascend/ops/fused_moe/des_tik.py
  git checkout -- vllm_ascend/ops/fused_moe/experts_selector.py \
                  vllm_ascend/ops/fused_moe/token_dispatcher.py \
                  vllm_ascend/_310p/fused_moe/experts_selector.py \
                  vllm_ascend/_310p/fused_moe/token_dispatcher.py
'
```

Or simply leave the files in place and set `VLLM_DES_ENABLE=0`, which restores
the upstream code path exactly.
