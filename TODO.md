# gfx90a (CDNA2 / MI250) — Follow-up TODO

Living tracker for deferred gfx90a bring-up work. The bring-up (safety guards, supported-path
enablement, docs, and a local CI gate) is implemented on `dev/gfx90a`; validated test status is in
[`docs/gfx90a_triage.md`](docs/gfx90a_triage.md). These two items are the porting efforts explicitly
deferred out of that work.

---

## 1. Enable int8 GEMM on gfx90a

**Status:** atom DONE ✅; production preshuffle wiring remaining.

**Done — the K=16/K=8 i8 MFMA atoms** (commit `18d1568`): `mfma_i32_16x16x16i8` (K=16) and
`mfma_i32_32x32x8i8` (K=8) added to the CDNA3 lowering (`getMfmaABType` now packs 4 i8 → i32 for
these; ThrVal/acc/verify were already K-parametric). No Python-binding change needed —
`MFMA(m,n,k,Int8,Int32)` builds them directly. Verified by FileCheck
(`tests/mlir/Conversion/mma_atom_i8_cdna2.mlir`) and a numeric single-tile GEMM on MI250
(`tests/kernels/test_i8_mma_gfx90a.py` — both K=16 and K=8 match the int32 reference). So int8 GEMM
via the tiled-MMA API works on gfx90a today. (fp8 stays out permanently — no FP8 hardware.)

**Remaining — wire the K=16 atom into the production `preshuffle_gemm` kernel.** A drop-in atom swap
does NOT work: the int8 path's `k_perm=(8,4,2)/(1,16,8)`, `tile_K_perm=64`, and the
`k_coord=(None, ki)` fragment indexing (`kernels/gemm/preshuffle_gemm.py` ~L175, L468-471, L696-698)
are all coupled to K=32. A K=16 atom changes the fragment K-decomposition — an experiment hit a
compile-time `profile mismatch` (`int_tuple<(*,*,(*,0))>` vs the K=16 fragment `(4,2,(2,2,8))`). The
work is: derive a K=16-correct `k_perm` + matching host `shuffle_weight` layout + `k_coord` indexing,
verified numerically against torch. Then relax the `is_8bit`→`is_fp8` guard
(`preshuffle_gemm.py` ~L159) and the test skip (`test_preshuffle_gemm.py` ~L165) for int8.

**Effort:** medium (focused layout work on one kernel; the atom + guards are done).

**Acceptance:** `test_preshuffle_gemm` int8 cases match the torch reference on gfx90a and pass.

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
