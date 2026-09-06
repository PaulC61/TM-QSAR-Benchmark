#!/usr/bin/env python
"""Patch known portability issues in vendored native submodules (dev/*).

The upstream `cair/tmu` C extension hardcodes the x86-only `-mrdrnd` compiler
flag (used nowhere in its actual source - it only calls `xorshift128p_seed`/
`pcg32_seed`), which fails to build with clang on Apple Silicon (osx-arm64)
and on some non-x86_64 Linux hosts (e.g. linux-aarch64). Since dev/tmu is a
pinned git submodule we don't control upstream, this script strips that flag
from the submodule's *local checkout* so `pixi install` can build it on any
of the platforms declared in pixi.toml. It's idempotent and safe to re-run
(e.g. after `git submodule update`), and is wired up as the pixi
`fix-submodules` task, which must run once after submodule init and before
`pixi install`.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TMU_PYPROJECT = REPO_ROOT / "dev" / "tmu" / "pyproject.toml"


def patch_tmu_flags() -> None:
    if not TMU_PYPROJECT.exists():
        print(
            f"[patch_native_deps] {TMU_PYPROJECT} not found - "
            "did you run `git submodule update --init --recursive`?"
        )
        return

    text = TMU_PYPROJECT.read_text()
    if '"-mrdrnd"' not in text:
        print("[patch_native_deps] tmu pyproject.toml already patched, skipping.")
        return

    patched = re.sub(r',?\s*"-mrdrnd"', "", text, count=1)
    TMU_PYPROJECT.write_text(patched)
    print(
        "[patch_native_deps] Removed unused/x86-only '-mrdrnd' compiler flag "
        f"from {TMU_PYPROJECT} for cross-platform builds."
    )


def main() -> None:
    patch_tmu_flags()


if __name__ == "__main__":
    main()
