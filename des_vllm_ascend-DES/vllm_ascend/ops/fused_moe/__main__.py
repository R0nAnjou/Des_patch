"""Installation diagnostics for ``python -m vllm_ascend_des``."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, entry_points, version

from .patch import TESTED_VLLM_ASCEND_VERSION, _installed_vllm_ascend_version


def main() -> None:
    try:
        patch_version = version("vllm-ascend-des-patch")
    except PackageNotFoundError:
        patch_version = "source tree (not installed)"

    ascend_version = _installed_vllm_ascend_version()
    plugin = next(
        (
            ep
            for ep in entry_points(group="vllm.general_plugins")
            if ep.name == "vllm_ascend_des"
        ),
        None,
    )

    print(f"DES patch package: {patch_version}")
    print(f"vllm-ascend: {ascend_version} (expected {TESTED_VLLM_ASCEND_VERSION})")
    print(f"vLLM plugin entry point: {'OK -> ' + plugin.value if plugin else 'MISSING'}")

    if plugin is None or ascend_version != TESTED_VLLM_ASCEND_VERSION:
        raise SystemExit(1)
    print("Installation check: OK")


if __name__ == "__main__":
    main()
