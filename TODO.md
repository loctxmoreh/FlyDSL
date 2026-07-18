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

## 2. Port the split-K HGEMM to gfx90a — DONE ✅ (commit `cc10da2`)

`hgemm_splitk` emitted `sc0`/`sc1` system-scope cache modifiers (gfx942+ asm syntax) in the async
`buffer_load_lds` and the split-K epilogue (zero-C store, signal store/load), which gfx90a's
assembler rejects. Made the modifiers arch-aware: CDNA3+ keeps `sc0 sc1`; gfx90a uses `glc`
(device/L2 scope) for the coherence-critical stores/loads and drops the modifier on the input DMA.
Rationale (documented inline at the def site): one MI250 GCD shares a single L2, so `glc` gives the
split-K cross-CU coherence `sc0 sc1` provides on CDNA3; escalate to `glc slc` if a race ever appears.
f16 split-K works (pk_add_f16); bf16 split-K (SPLIT_K>1) fail-fasts (no pk_add_bf16 on gfx90a, same
as MoE) while bf16 SPLIT_K=1 (plain store) and all f16 work. Un-gated the test; bf16 SPLIT_K>1 skips.
Verified on MI250: `test_hgemm_splitk` = 18 passed / 10 skipped / 0 failed; f16 SPLIT_K>1 re-run 3x
for coherence — stable; CI gate green (1066 passed). Scope: only `hgemm_splitk.py` (the sole
test-covered module); `splitk_hgemm.py`/`small_m_hgemm.py` are untested siblings, left as-is.

---

**All gfx90a bring-up follow-ups are now complete.** Remaining out-of-scope-by-hardware on gfx90a:
FP8/FP4/MX (no hardware), CDNA4-only ops, and bf16 packed atomics (bf16 split-K / bf16-output MoE) —
all fail-fast cleanly. See [`docs/gfx90a_triage.md`](docs/gfx90a_triage.md).
