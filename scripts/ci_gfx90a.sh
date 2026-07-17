#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Local gfx90a (CDNA2 / MI250) validation "CI job".
#
# gfx90a is NOT in the official support matrix (CDNA3+); this is the dev/gfx90a
# bring-up gate. The remote repo has no gfx90a CI runner, so run this directly on
# an MI250 box. It validates the arch-supported subset (examples 01-05 +
# tests/kernels + tests/unit + tests/system) and asserts pass/skip only — zero
# core dumps, zero unexpected failures.
#
# Usage:
#   bash scripts/ci_gfx90a.sh                 # auto-select emptiest GPU, run gate
#   HIP_VISIBLE_DEVICES=3 bash scripts/ci_gfx90a.sh
#
# Exit 0 = gate green; non-zero = a regression (crash / unexpected failure).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# ── Environment ────────────────────────────────────────────────────────────
# Repo root on PYTHONPATH so examples can import kernels.* / tests.* directly
# (pytest adds rootdir automatically; a bare `python3 examples/x.py` does not).
# flydsl itself still resolves via the editable install (python/flydsl).
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${REPO_ROOT}/build-fly/python_packages/flydsl/_mlir/_mlir_libs:${LD_LIBRARY_PATH:-}"
export TMPDIR="${TMPDIR:-/remote/vast0/loctran/tmp}"
export FLYDSL_RUN_QUANT=1
# Softmax test_all defaults to a ~1GB shape. This box's HIP hipMemGetInfo
# mis-reports free VRAM (~1GB free on an idle 64GB GCD; ROCm 7.2.2 runtime vs
# torch 2.10+rocm7.0), so large allocations OOM regardless of arch. Use a modest
# sweep that still exercises the softmax kernel end-to-end on gfx90a.
export ROCDSL_SOFTMAX_SHAPES="${ROCDSL_SOFTMAX_SHAPES:-1024,8192,bf16;64,2000,f32;32,128,f16;128,1024,f32}"

# ── GPU selection (rocm-smi is authoritative; torch mem_get_info is broken) ──
if [[ -z "${HIP_VISIBLE_DEVICES:-}" ]] && command -v rocm-smi &>/dev/null; then
    _best="$(rocm-smi --showmeminfo vram 2>/dev/null | grep 'VRAM Total Used Memory' \
        | sed -E 's/^GPU\[([0-9]+)\].*:[[:space:]]*([0-9]+)[[:space:]]*$/\1 \2/' \
        | sort -k2 -n | head -1 | awk '{print $1}')"
    if [[ -n "${_best}" ]]; then
        export HIP_VISIBLE_DEVICES="${_best}"
    fi
fi
echo "[ci_gfx90a] HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-<unset>}"

ARCH="$(python3 -c 'from flydsl.runtime.device import get_rocm_arch; print(get_rocm_arch())' 2>/dev/null | tail -1)"
echo "[ci_gfx90a] detected arch: ${ARCH}"
if [[ "${ARCH}" != gfx90a* ]]; then
    echo "[ci_gfx90a] WARNING: this gate is intended for gfx90a; detected '${ARCH}'."
fi

LOG_DIR="${REPO_ROOT}/build-fly/ci_gfx90a"
mkdir -p "${LOG_DIR}"
JUNIT="${LOG_DIR}/kernels.junit.xml"
KLOG="${LOG_DIR}/kernels.log"

fail=0

# ── 1. Examples (all run on CDNA per tests/arch_compat.py) ───────────────────
echo ""
echo "==================== examples ===================="
example_status=()
for ex in 01-vectorAdd 02-tiledCopy 03-tiledMma 04-preshuffle_gemm 05-gather_scatter; do
    if timeout 300 python3 "examples/${ex}.py" >"${LOG_DIR}/${ex}.log" 2>&1; then
        if grep -qiE "correct: True|PASS|correct" "${LOG_DIR}/${ex}.log"; then
            echo "  PASS  ${ex}"
            example_status+=("PASS ${ex}")
        else
            echo "  FAIL  ${ex} (ran but no success marker)"
            example_status+=("FAIL ${ex}")
            fail=1
        fi
    else
        echo "  FAIL  ${ex} (exit != 0 / timeout / crash)"
        example_status+=("FAIL ${ex}")
        fail=1
    fi
done

# ── 2. Test suites (supported subset: no large/bench/multi-gpu) ──────────────
# kernels = device tier; unit/system = host + compile + device tiers.
echo ""
echo "==================== tests/kernels + tests/unit + tests/system ===================="
timeout 2400 python3 -m pytest tests/kernels/ tests/unit/ tests/system/ \
    -m "not large_shape and not benchmark and not multi_gpu" \
    -q -p no:cacheprovider --junit-xml="${JUNIT}" 2>&1 | tee "${KLOG}" | grep -ivE "amdgpu.ids" | tail -20
pytest_rc="${PIPESTATUS[0]}"

# ── 3. Guardrails: no core dumps, no collection errors ───────────────────────
crashes="$(grep -ciE "dumped core|Fatal Python error|Aborted|core dumped" "${KLOG}" 2>/dev/null)"
crashes="${crashes:-0}"
if [[ "${crashes}" -gt 0 ]]; then
    echo "[ci_gfx90a] ERROR: ${crashes} core-dump/abort marker(s) — a kernel emitted an unsupported instruction."
    fail=1
fi
# pytest_rc: 0=all pass/skip, 1=failures, 5=no tests collected, 2=collection error
if [[ "${pytest_rc}" -ne 0 ]]; then
    echo "[ci_gfx90a] ERROR: pytest exit code ${pytest_rc} (expected 0 = pass/skip only)."
    fail=1
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "==================== gfx90a gate summary ===================="
printf '%s\n' "${example_status[@]}"
grep -E "(passed|failed|error)" "${KLOG}" | grep -E "warning|deselected|passed|failed|error in" | tail -1
echo "  junit: ${JUNIT}"
if [[ "${fail}" -eq 0 ]]; then
    echo "[ci_gfx90a] RESULT: PASS ✅ (pass/skip only, zero crashes)"
else
    echo "[ci_gfx90a] RESULT: FAIL ❌"
fi
exit "${fail}"
