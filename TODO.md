# gfx90a (CDNA2 / MI250) — Follow-up TODO

Living tracker for deferred gfx90a bring-up work. The bring-up (safety guards, supported-path
enablement, docs, and a local CI gate) is implemented on `dev/gfx90a`; validated test status is in
[`docs/gfx90a_triage.md`](docs/gfx90a_triage.md). These two items are the porting efforts explicitly
deferred out of that work.

---

## 1. Enable int8 GEMM on gfx90a — DONE ✅

**The K=16/K=8 i8 MFMA atoms** (commit `18d1568`): `mfma_i32_16x16x16i8` / `mfma_i32_32x32x8i8` added
to the CDNA3 lowering (`getMfmaABType` packs 4 i8 → i32 for these; ThrVal/acc/verify were already
K-parametric; `MFMA(m,n,k,Int8,Int32)` builds them, no binding change). FileCheck
`tests/mlir/Conversion/mma_atom_i8_cdna2.mlir` + numeric `tests/kernels/test_i8_mma_gfx90a.py`.

**Production preshuffle wiring** (commit `421e63f`): the int8 path now branches on arch — gfx942/gfx950
keep K=32 MFMA (`k_perm=(8,4,2)/(1,16,8)`), gfx90a uses K=16 (`k_perm=(4,4,4)/(1,16,4)`), i.e. the same
64-wide preshuffle-K tile run as 4 MFMA substeps of 16 (KPerThread=4) instead of 2 of 32. Derived from
the shared `(KPerThread, 4, substeps)/(1, KPerThread*substeps, KPerThread)` structure; `k_coord=(None,
ki)` unchanged (the tiled_mma expands substeps). Guard relaxed to fp8-only; int8 un-skipped in the test.
Verified on MI250: `test_preshuffle_gemm` int8 matches torch across small + large shapes, eager + graph
(f16/bf16 unregressed); CI gate green (1048 passed). fp8 stays out permanently — no FP8 hardware.

---

## 2. Port the split-K HGEMM family to gfx90a

**Status:** not started. `test_hgemm_splitk` stays gfx942/gfx950-only.

**Why deferred:** `hgemm_splitk` / `splitk_hgemm` / `small_m_hgemm` emit `sc0`/`sc1`
**system-scope cache modifiers** (gfx942+ assembler syntax) in (a) the async `buffer_load_lds`
and (b) the split-K epilogue inline-asm stores. Confirmed on gfx90a via
`error: invalid operand for instruction ... buffer_load_dword ... sc0 lds`. gfx90a uses
`glc`/`slc` and has **no `sc1` (system-scope) equivalent** — and the split-K cross-CU
accumulation coherence relies on that scope, so a naive swap risks data races.

**Surface of change:**
- `kernels/gemm/hgemm_splitk.py` (epilogue `global_store_dwordx4/dword ... sc0 sc1` at ~312/325;
  async `buffer_load_lds_inline` at ~406), `splitk_hgemm.py` (~314/334/493/532),
  `small_m_hgemm.py` (~662/795/988) — make the cache-modifier asm arch-aware (`glc`/`slc`
  on gfx90a) and route async through a sync copy where `buffer_load_lds` isn't valid.
- Verify the split-K semaphore / accumulation coherence still holds under gfx90a scopes
  (the risky part — needs a correctness harness, not just compile success).
- `tests/kernels/test_hgemm_splitk.py:101` — add gfx90a once it passes.

**Effort:** high + correctness risk (cross-CU coherence).

**Acceptance:** `test_hgemm_splitk` un-gated for gfx90a, f16/bf16 cases match reference, split-K
accumulation is race-free under stress.
