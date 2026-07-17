# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Regression tests for the gfx90a (CDNA2) safety guards.

These lock in the Tier 0 bring-up invariants (see the dev/gfx90a work):
- the CDNA-generation classifier (`is_cdna3`/`is_cdna4`),
- FP8 fail-fast in `default_f8_type` for archs without native FP8,
- the 64KB LDS capacity entry for gfx90a.

They are host-only (no GPU), so they run on any CI arch and catch a silent
revert of a guard that would let gfx90a emit an unsupported instruction.
"""

import pytest

# ── T0.1: CDNA-generation classifier ────────────────────────────────────────


@pytest.mark.l0_backend_agnostic
def test_cdna_classifiers():
    from flydsl.runtime.device import is_cdna3, is_cdna4, is_rdna_arch

    # gfx942 = CDNA3, gfx950/gfx95* = CDNA4, gfx90a = CDNA2 (neither).
    assert is_cdna3("gfx942")
    assert not is_cdna3("gfx90a")
    assert not is_cdna3("gfx950")

    assert is_cdna4("gfx950")
    assert is_cdna4("gfx95x")
    assert not is_cdna4("gfx90a")
    assert not is_cdna4("gfx942")

    # CDNA2 must be classified as neither CDNA3 nor CDNA4, and as CDNA (not RDNA),
    # so the gfx950-assuming `else` branches fall back to the CDNA3-safe path.
    assert not is_cdna3("gfx90a") and not is_cdna4("gfx90a")
    assert not is_rdna_arch("gfx90a")


# ── T0.2: FP8 fail-fast in default_f8_type ──────────────────────────────────


def _patch_arch(monkeypatch, arch):
    monkeypatch.setattr("flydsl.expr.typing.get_rocm_arch", lambda: arch)


@pytest.mark.l1a_compile_no_target_dialect
@pytest.mark.parametrize("arch", ["gfx90a", "gfx908", "gfx1100", "gfx1101"])
def test_default_f8_type_rejects_archs_without_fp8(monkeypatch, arch):
    from flydsl.expr.typing import default_f8_type

    _patch_arch(monkeypatch, arch)
    with pytest.raises(RuntimeError, match="no native FP8"):
        default_f8_type()


@pytest.mark.l1a_compile_no_target_dialect
@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx1201"])
def test_default_f8_type_allows_fp8_archs(monkeypatch, arch):
    from flydsl._mlir import ir
    from flydsl.expr.typing import default_f8_type

    _patch_arch(monkeypatch, arch)
    # Must NOT raise the "no native FP8" RuntimeError; builds a real fp8 type.
    with ir.Context():
        t = default_f8_type()
    assert t is not None


# ── T0.3: LDS capacity enforcement ──────────────────────────────────────────


@pytest.mark.l0_backend_agnostic
def test_smem_capacity_gfx90a():
    from flydsl.utils.smem_allocator import SMEM_CAPACITY_MAP, check_smem_capacity

    # gfx90a (CDNA2 / MI250) has 64KB LDS per CU.
    assert SMEM_CAPACITY_MAP.get("gfx90a") == 65536

    check_smem_capacity(60000, "gfx90a")  # within limit → no raise
    with pytest.raises(RuntimeError, match="Shared Memory Overflow"):
        check_smem_capacity(70000, "gfx90a")
