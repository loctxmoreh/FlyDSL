# gfx90a (CDNA2 / MI250) Test Triage

Snapshot of the full test triage on gfx90a, validating the `dev/gfx90a` bring-up. gfx90a is
**experimental**, not in the official support matrix (CDNA3+).

**Reproduce:** `source .venv/bin/activate && bash scripts/ci_gfx90a.sh`
(runs on an MI250 — the remote repo has no gfx90a CI runner). The gate asserts **pass/skip only**:
zero core dumps, pytest exit 0.

**Last validated:** 2026-07-17, ROCm 7.2.2 / torch 2.10+rocm7.0, MI250 `gfx90a`.

## Summary

| Suite | Passed | Skipped | Failed | Crashes |
|---|---:|---:|---:|---:|
| examples 01–05 | 5 | 0 | 0 | 0 |
| `tests/kernels` (¬large/¬bench/¬multi_gpu) | 225 | 2310 | 0 | 0 |
| `tests/unit` + `tests/system` (¬large/¬bench/¬multi_gpu) | 816 | 6 | 0 | 0 |
| **total** | **1046** | **2316** | **0** | **0** |

(pytest-collected total 1041 pass / 2316 skip + 5 examples.)

## Why things skip (categorized — none are silent miscompiles)

Every skip/fail-fast names gfx90a and the missing feature. Main categories:

- **FP8 GEMM / MoE / attention** — no FP8 MFMA on gfx90a (CDNA3+); kernels raise, tests skip.
- **FP4 / MX-scaled / CDNA4 transpose** (`ds_read_tr*`, scaled MFMA) — CDNA4-only (gfx950).
- **int8 GEMM** — the K=16/8 i8 MFMA atoms now exist and int8 GEMM works via the tiled-MMA API
  (`tests/kernels/test_i8_mma_gfx90a.py`); the production `preshuffle_gemm` int8 path still
  fail-fasts pending a K=16 wiring — see [`../TODO.md`](../TODO.md).
- **Split-K HGEMM** (`test_hgemm_splitk`) — `sc0`/`sc1` system-scope cache modifiers, gfx942/gfx950
  only — planned, see [`../TODO.md`](../TODO.md).
- **bf16-output MoE** — no packed bf16 global atomic on gfx90a.
- **gfx1250 / RDNA-specific tests** — different arch families (self-skip).
- **`multi_gpu`, `large_shape`, `benchmark`** — deselected by the gate's marker filter.

## Known environment caveat (not an arch boundary)

This box's HIP `hipMemGetInfo` mis-reports free VRAM (~1 GB free on an idle 64 GB GCD; ROCm 7.2.2
runtime vs torch 2.10+rocm7.0), so allocations >~1 GB OOM regardless of arch. The gate works around
it two ways: GPU selection uses `rocm-smi` (not torch), and softmax runs a modest sweep via
`ROCDSL_SOFTMAX_SHAPES` so its default ~1 GB shape doesn't trip the quirk. The softmax **kernel**
itself compiles and runs correctly on gfx90a.

## Open follow-ups

See [`../TODO.md`](../TODO.md): (1) int8 enablement (K=16/8 i8 MFMA atoms + retune), (2) split-K
HGEMM port (`sc0`/`sc1` → `glc`/`slc`, cross-CU coherence). Each would move a ⛔ row above to ✅ and
extend this gate's coverage.
