#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Numeric regression for the gfx90a (CDNA2) i8 MFMA atoms.

Exercises MFMA(16,16,16, i8) (K=16) and MFMA(32,32,8, i8) (K=8) — the i8 MFMA
instructions gfx90a has (the K=32/16 i8 ops are gfx942+). A single-tile tiled-MMA
GEMM validates the atoms' ThrVal layouts + operand packing end-to-end against an
int32 reference. Runs on any CDNA (gfx9xx); the K=16/K=8 i8 ops exist on
gfx908/gfx90a/gfx942/gfx950.
"""

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.runtime.device import get_rocm_arch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available.", allow_module_level=True)

_ARCH = str(get_rocm_arch())
if not _ARCH.startswith("gfx9"):
    # i8 MFMA is a CDNA (gfx9xx) instruction; RDNA/gfx1250 use WMMA.
    pytest.skip(f"i8 MFMA requires CDNA (gfx9xx), got {_ARCH}", allow_module_level=True)


@pytest.mark.parametrize("mnk", [(16, 16, 16), (32, 32, 8)], ids=["16x16x16", "32x32x8"])
def test_i8_mfma_single_tile(mnk):
    block_m, block_n, block_k = mnk

    @flyc.kernel
    def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        A = fx.rocdl.make_buffer_tensor(A)
        B = fx.rocdl.make_buffer_tensor(B)
        C = fx.rocdl.make_buffer_tensor(C)

        bA = fx.slice(fx.zipped_divide(A, (block_m, block_k)), (None, 0))
        bB = fx.slice(fx.zipped_divide(B, (block_n, block_k)), (None, 0))
        bC = fx.slice(fx.zipped_divide(C, (block_m, block_n)), (None, 0))

        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(block_m, block_n, block_k, fx.Int8, fx.Int32))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
        thr_mma = tiled_mma.thr_slice(tid)

        copy_ab = fx.make_copy_atom(fx.rocdl.BufferCopy8b(), fx.Int8)
        copy_c = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        tca = fx.make_tiled_copy_A(copy_ab, tiled_mma).get_slice(tid)
        tcb = fx.make_tiled_copy_B(copy_ab, tiled_mma).get_slice(tid)
        tcc = fx.make_tiled_copy_C(copy_c, tiled_mma).get_slice(tid)

        frag_A = thr_mma.make_fragment_A(bA)
        frag_B = thr_mma.make_fragment_B(bB)
        frag_C = thr_mma.make_fragment_C(bC)

        fx.copy(copy_ab, tca.partition_S(bA), tca.retile(frag_A), pred=None)
        fx.copy(copy_ab, tcb.partition_S(bB), tcb.retile(frag_B), pred=None)
        frag_C.fill(0)
        fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)
        fx.copy(copy_c, tcc.retile(frag_C), tcc.partition_S(bC), pred=None)

    @flyc.jit
    def run(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        gemm_kernel(A, B, C).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    M, N, K = block_m, block_n, block_k
    A = torch.randint(-8, 8, (M, K), dtype=torch.int8).cuda()
    B = torch.randint(-8, 8, (N, K), dtype=torch.int8).cuda()
    C = torch.zeros(M, N, dtype=torch.int32).cuda()
    run(A, B, C, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    # int matmul is unsupported on ROCm torch; compute the reference on CPU.
    expected = (A.cpu().to(torch.int32) @ B.cpu().to(torch.int32).T).cuda()
    assert torch.equal(C, expected), f"i8 {M}x{N}x{K} mismatch: max |diff|={(C - expected).abs().max().item()}"
