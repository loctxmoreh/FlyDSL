# Plan: gfx90a (CDNA2 / MI250) Compatibility

**Status:** proposal / bring-up plan. gfx90a is **not** in the official support matrix
(CDNA3 `gfx942` and later). This document scopes what it takes to make gfx90a a safe,
partially-supported target on the `dev/gfx90a` branch.

> **This document is frozen.** It is a fixed scope/plan, not a living tracker. Do not add
> progress markers, status columns, checkboxes, or completion notes here. Track execution
> state elsewhere (issue tracker / task list / PRs). The only edits expected are a deliberate
> re-scope, which should be treated as a new revision of the plan.

## Background: where the boundary actually sits

The boundary is narrower than "CDNA3+" suggests. Mapped two ways — a static catalog of every
arch gate, plus empirical runs on a real MI250 — and they agree:

- No source file mentions `gfx90a`. It is classified today by `is_rdna_arch('gfx90a') == False`
  and `get_warp_size == 64`, i.e. treated as **CDNA / wave64**. Both are already correct.
- The C++ MFMA / BufferCopy lowering (`lib/Dialect/FlyROCDL/CDNA3/`) has **no arch guard** — it
  emits whatever atom the Python layer requests. So gfx90a safety is decided **entirely in Python**.

**Verified working on the MI250, unmodified:**

| Check | Result |
|---|---|
| `examples/01..05` (incl. 04 = production f16 preshuffle GEMM) | pass |
| `tests/kernels/test_vec_add.py` | 3 passed |
| `tests/kernels/test_rmsnorm.py` | 14 passed, 1 skipped |
| `tests/kernels/test_preshuffle_gemm.py` **fp16 + bf16** cases | 8 passed |

bf16 works because it lowers to `mfma_f32_16x16x16bf16_1k`, which gfx90a has.

**Genuinely unsupported on gfx90a — today these CRASH rather than fail-fast:**

| Feature | Availability | Emitter |
|---|---|---|
| FP8 MFMA (`mfma_*_fp8/bf8`) | gfx942+ | `CDNA3/MmaAtom.cpp:201-208` |
| K=32/16 f16/bf16 MFMA (`mfma_f32_16x16x32_*`, `32x32x16_*`) | gfx942+ | `CDNA3/MmaAtom.cpp:188-199` |
| i8 K=32 (`mfma_i32_16x16x32_i8`, `32x32x16_i8`) | gfx942+ | `CDNA3/MmaAtom.cpp:210-211` |
| 128b `buffer_load_dwordx4_lds` | gfx950+ | `CDNA3/CopyAtom.cpp:324` |
| MX/scaled MFMA, `ds_read_tr*`, fp4 | gfx950 (CDNA4) | `CDNA4/*` |
| Packed bf16 atomics (`buffer_`/`global_atomic_pk_add_bf16`) | gfx942+ (gfx90a has **neither**) | Python-side decision |

Empirically, `test_preshuffle_gemm[…-fp8]` **core-dumps** (`Fatal Python error: Aborted`) — the
kernel compiles and dispatches an fp8 MFMA the hardware cannot execute → GPU fault.

**Highest-risk pattern — silent gfx950-assuming `else`:** several kernels branch
`if arch == "gfx942": <K16 path> else: <gfx950 path>`. gfx90a matches neither prefix and falls
into the gfx950 `else`, assuming K=32 MFMA / 128b LDS DMA it lacks, with no fail-fast.

Environment caveat (not a compat boundary): `test_softmax` OOMs even on an idle GPU here (harness
memory behaviour); the venv needed `pytest` + `pandas` installed (not seeded by `uv venv`).

---

## Tier 0 — Make gfx90a safe and honest

Goal: gfx90a either runs correctly or fails fast with a clear message. **No silent miscompiles, no
GPU core dumps.** This tier does not add new capability; it closes the crash/miscompile holes.

### T0.1 — CDNA-generation classifier + fix gfx950-assuming `else` branches

**Surface of change:**
- `python/flydsl/runtime/device.py`: add a small, single-source-of-truth helper alongside
  `is_rdna_arch`, e.g. `is_cdna3(arch)` / `is_cdna4(arch)` (or `cdna_gen(arch) -> int`). Prefix
  rules: `gfx90a → CDNA2`, `gfx94* → CDNA3`, `gfx95* → CDNA4`. Keep it prefix-based to match the
  existing style.
- Replace the binary `if arch == "gfx942" … else <gfx950>` branches so the gfx950 body only runs
  for actual CDNA4, and gfx90a takes the gfx942-safe body (K=16 MFMA, 4B LDS DMA) or raises:
  - `kernels/gemm/hgemm_splitk.py:131`
  - `kernels/gemm/splitk_hgemm.py:129`
  - `kernels/gemm/small_m_hgemm.py:395` (also the `raise` at `:392` only catches gfx942)
  - `kernels/gemm/blockscale_preshuffle_gemm.py:91` (`a_async_load_bytes = 4 if _is_gfx942 else 16`)
  - `kernels/attention/flash_attn_generic.py:165` (`_has_lds_load_b128 = not startswith("gfx942")`)
- Prefer the arch-helper module over inline `gfx*` string compares (per CLAUDE.md conventions).

**Success criteria:**
- Each listed kernel, when built for gfx90a, either takes a K=16 / 4B-LDS path that compiles and
  runs, or raises a clear `NotImplementedError`/`ValueError` naming gfx90a — never emits a K=32
  MFMA or 128b `buffer_load_lds`.
- Grep shows no remaining `else`-assumes-gfx950 branch reachable by gfx90a.
- A unit test (compile-only, no GPU) asserts the new classifier: `is_cdna3('gfx942') and not
  is_cdna3('gfx90a')`, etc.

### T0.2 — FP8 fail-fast in `default_f8_type`

**Surface of change:**
- `python/flydsl/expr/typing.py:161-186` (`default_f8_type`): today only `gfx11*` raises; everything
  that is not `gfx95*`/`gfx12*` (including gfx90a) silently falls through to `Float8E4M3FNUZ`. Add
  gfx90a (and any non-fp8 CDNA) to the reject path so requesting an fp8 type on gfx90a raises with a
  message like the existing "FP8 instructions are available on gfx94*, gfx95*, and gfx12*".

**Success criteria:**
- `default_f8_type('gfx90a')` raises `RuntimeError` (not a returned FNUZ type).
- `tests/kernels/test_preshuffle_gemm.py::...[...-fp8]` on gfx90a **skips or raises a Python
  exception before dispatch** — no `Fatal Python error: Aborted`, no core dump.
- A compile-tier unit test covers the raise.

### T0.3 — LDS capacity enforcement

**Surface of change:**
- `python/flydsl/utils/smem_allocator.py:239-270`: add `gfx90a: 65536` to `SMEM_CAPACITY_MAP` so
  `check_smem_capacity` enforces the 64KB limit instead of silently skipping unknown arch.
- Sanity-check any kernel `_lds_limit` maps that `.get(arch, 0)` (e.g.
  `kernels/moe/mixed_moe_gemm_2stage/gemm1.py:227`) — gfx90a → 0 disables an optimization but does
  not miscompile; leave unless it blocks a Tier-1 kernel.

**Success criteria:**
- Allocating > 64KB LDS on gfx90a raises the capacity error (add a targeted unit test).
- gfx90a-sized (<=64KB) allocations still pass.

### T0.4 — Guard fp8 / fp4 / MX / CDNA4 kernels + tests to skip cleanly

**Surface of change:**
- Ensure the CDNA4-only / fp8 kernels and their tests refuse gfx90a with a skip or clear raise
  rather than compiling to an invalid instruction. Anchor points:
  `kernels/gemm/mxfp4_preshuffle.py`, `fp4_gemm_4wave.py`, `fp8_gemm_*wave.py`,
  `conv3d_implicit_8wave_fp8.py`, and the CDNA4 flash-attn kernels (which already raise for
  non-gfx950).
- Where a test parametrizes dtypes over one kernel (e.g. `test_preshuffle_gemm.py`'s
  fp8/int8/fp16/bf16 matrix), add a gfx90a-aware skip for the fp8 (and K=32 i8) params so the suite
  does not abort the whole process on one crashing case.

**Success criteria:**
- Running the full `tests/kernels/` collection on gfx90a produces **pass/skip only — zero core
  dumps / process aborts**.
- Every skip carries a reason string that names gfx90a and the missing feature.

---

## Tier 1 — Enable the supported path on gfx90a

Goal: turn on and validate the f32 / f16 / bf16 / i8 kernels that gfx90a can actually run. The core
MFMA path already works (proven); this tier is mostly un-gating + one i8 correction.

### T1.1 — Add gfx90a to test arch allowlists

**Surface of change:**
- `tests/kernels/test_hgemm_splitk.py:101` (`if ARCH not in ["gfx950","gfx942"]: skip`) → include
  gfx90a.
- Audit sibling GEMM tests for the same allowlist pattern and extend where the kernel is a
  supported-path kernel.
- `tests/arch_compat.py` / `tests/kernels/conftest.py`: `is_cdna = "gfx9" in arch` already makes
  gfx90a run the CDNA tests; ensure CDNA-only tests that are actually fp8/CDNA4 are excluded via
  the T0.4 skips rather than via `is_cdna`.

**Success criteria:**
- The supported-path GEMM tests (f16/bf16/f32, non-async where async is gfx942+) **collect and
  pass** on gfx90a instead of skipping wholesale.
- No test that requires a gfx942+/gfx950 feature is un-gated onto gfx90a.

### T1.2 — i8 GEMM K selection

**Surface of change:**
- Trace the int8 preshuffle path (`kernels/gemm/preshuffle_gemm.py` and the `test_mfma_a8`
  parametrization). If it selects `mfma_i32_16x16x32_i8` / `32x32x16_i8` (gfx942+), add a gfx90a
  fallback to the K=16/8 forms (`mfma_i32_16x16x16_i8`, `32x32x8_i8`) via the T0.1 classifier.

**Success criteria:**
- An int8 GEMM case runs correctly on gfx90a (numerically matches the torch reference), OR — if the
  kernel's tiling assumes K=32 and a K=16 retune is out of scope — it fails fast naming gfx90a.
- Decision (support vs fail-fast) recorded here.

### T1.3 — bf16 packed-atomic comments + optional fallback

**Surface of change:**
- Correct the misleading comments in `kernels/moe/moe_gemm_2stage/gemm2.py:210-213`,
  `gemm1.py:1261`, `kernels/moe/moe_blockscale_2stage/gemm2.py:140-142`: gfx90a has **neither**
  `buffer_atomic_pk_add_bf16` nor `global_atomic_pk_add_bf16`.
- bf16-output MoE already fail-fasts via `supports_bf16_global_atomics` (gfx90a excluded) — that is
  correct. Only if bf16 MoE output on gfx90a is required, add a non-packed (element-wise) atomic
  fallback; otherwise leave the fail-fast.

**Success criteria:**
- Comments accurately describe gfx90a.
- bf16-output MoE on gfx90a raises the existing clear error (default), or — if the fallback is built
  — produces correct bf16 output.

---

## Tier 2 — Validation, CI, and docs

Goal: lock in the bring-up and make gfx90a's status discoverable.

### T2.1 — Full-suite triage on gfx90a

**Surface of change:**
- Run `tests/kernels`, `tests/unit`, `tests/system` (minus `large_shape`, fp8/CDNA4) on the MI250;
  capture a pass/skip/fail table. File follow-up tasks for any residual failure.
- Resolve or document the `test_softmax` OOM (pin `HIP_VISIBLE_DEVICES` to a free GCD; confirm it is
  a harness memory issue, not arch).

**Success criteria:**
- A committed results table (in a separate file, not this frozen doc) showing every gfx90a test
  as pass / skip(reason) / known-fail(ticket). Zero unexplained aborts.

### T2.2 — Optional gfx90a CI tier

**Surface of change:**
- If gfx90a hardware is to stay in the loop, add a label-gated / manual-dispatch job (mirroring the
  `multi_gpu` gating pattern) that runs the supported-path subset on an MI250 runner.

**Success criteria:**
- The job runs green on the supported subset and is not part of the default `run_tests.sh` flow
  (gfx90a is not in the official matrix).

### T2.3 — Docs

**Surface of change:**
- `CLAUDE.md` GPU Architecture Support table and `docs/prebuilt_kernels_guide.md`: add a gfx90a row
  marked **experimental / partial**, listing supported (f32/f16/bf16/i8 GEMM, norm, softmax,
  elementwise, tiled copy/MMA) vs unsupported (fp8, fp4/MX, CDNA4 transpose/scale MFMA, packed bf16
  atomics, 128b LDS DMA, K=32 MFMA).

**Success criteria:**
- A reader can tell from the docs exactly which kernels/dtypes are expected to work on gfx90a and
  which fail fast.

---

## Out of scope

Bringing FP8 / FP4 / MX / CDNA4-transpose kernels to gfx90a — the hardware lacks the instructions;
the correct behaviour is fail-fast (Tier 0), not emulation. gfx90a remains outside the official
support matrix; this plan targets a safe, partial bring-up on `dev/gfx90a`.
