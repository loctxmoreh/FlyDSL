# gfx90a (CDNA2 / MI250) — Follow-up TODO

Living tracker for deferred gfx90a bring-up work. The bring-up (safety guards, supported-path
enablement, docs, and a local CI gate) is implemented on `dev/gfx90a`; validated test status is in
[`docs/gfx90a_triage.md`](docs/gfx90a_triage.md). These two items are the porting efforts explicitly
deferred out of that work.

---

## 1. Enable int8 GEMM on gfx90a

**Status:** not started. Currently fail-fasts (Tier 0 guard) + test-skips.

**Why deferred:** the CDNA3 C++ lowering only emits i8 MFMA at K=32 / K=16
(`mfma_i32_16x16x32_i8`, `mfma_i32_32x32x16_i8` — both gfx942+). gfx90a's valid i8 MFMA
(`mfma_i32_16x16x16_i8`, `mfma_i32_32x32x8_i8`, K=16/8) are **not in the dispatch at all**.
(fp8 is out of scope permanently — gfx90a has no FP8 hardware.)

**Surface of change:**
- `lib/Dialect/FlyROCDL/CDNA3/MmaAtom.cpp:210-211` — add K=16/8 i8 dispatch entries.
- `python/flydsl/expr/rocdl/__init__.py` — add the `mfma_i32_16x16x16_i8` / `32x32x8_i8`
  bindings if the ROCDL op wrappers are missing.
- `kernels/gemm/preshuffle_gemm.py` — add a K=16 int8 path for gfx90a (tile-K micro-step,
  MMA operand + LDS register layouts differ from K=32); relax the `is_8bit` guard to allow
  int8 on gfx90a once the K=16 path exists (keep fp8 rejected).
- `tests/kernels/test_preshuffle_gemm.py:165` — allow int8 on gfx90a in the skip predicate.

**Effort:** medium-high (new atom op via `/add-target-atom-op` + kernel retune + numeric verify).

**Acceptance:** an int8 preshuffle GEMM case matches the torch reference on gfx90a; no core dump;
test collects and passes.

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
