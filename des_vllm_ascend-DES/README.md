# vLLM Ascend DES patch

This repository packages Dynamic Expert Selection (DES) as an out-of-tree
vLLM plugin. It targets **vLLM Ascend 0.23.0 on Ascend 310P3**.

The plugin uses vLLM's `vllm.general_plugins` entry point. It is loaded in the
API process and every engine/worker process, then wraps the four DES call sites
in memory. Each wrapper delegates to the installed 0.23.0 implementation, so
upstream behavior is retained. It does not copy files into or modify the
installed `vllm_ascend` source tree.

## Install

From this directory, in the same Python environment as vLLM Ascend:

```bash
python -m pip install -e .
python -m vllm_ascend_des
```

The second command verifies the installed distribution, vLLM plugin entry
point, and target vLLM Ascend version. Restart every running vLLM process after
installing or changing DES settings.

DES remains off by default. A typical 310P launch is:

```bash
VLLM_DES_ENABLE=1 \
VLLM_DES_THRESHOLD=0.9 \
VLLM_DES_MIN_K=1 \
VLLM_DES_SKIP=1 \
VLLM_DES_BACKEND=tik \
vllm serve /path/to/model --dtype float16
```

The patch is loaded automatically. If `VLLM_PLUGINS` is set, it must include
`vllm_ascend_des`, otherwise vLLM's plugin allowlist will exclude this patch:

```bash
export VLLM_PLUGINS=vllm_ascend_des
```

See [INSTALL_ASCEND.md](INSTALL_ASCEND.md) for configuration, runtime checks,
hardware notes, and known pitfalls.

## Compatibility policy

The plugin refuses to load on a version other than vLLM Ascend 0.23.0. For
development only, `VLLM_DES_ALLOW_UNSUPPORTED=1` changes that error into a
warning.

Uninstall with:

```bash
python -m pip uninstall vllm-ascend-des-patch
```
