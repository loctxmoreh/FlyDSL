// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-rocdl=chip=gfx90a -split-input-file 2>&1 | FileCheck %s

// Emit-time arch guard: with chip=gfx90a (CDNA2), the shared CDNA3 MFMA type must
// reject instructions gfx90a lacks (FP8 MFMA, wide-K MFMA) with a clear diagnostic,
// and accept the CDNA2-valid subset (K=16/K=8 i8).

// gfx90a has no FP8 MFMA (gfx942+ only).
// CHECK: FP8 MFMA is not available on gfx90a (CDNA2)
func.func @fp8_rejected_on_gfx90a(%a: vector<8xi8>, %b: vector<8xi8>, %c: vector<4xf32>) -> vector<4xf32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x32, (f8E4M3FNUZ, f8E4M3FNUZ) -> f32>>
  %res = fly.mma_atom_call_ssa(%atom, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x32, (f8E4M3FNUZ, f8E4M3FNUZ) -> f32>>, vector<8xi8>, vector<8xi8>, vector<4xf32>) -> vector<4xf32>
  return %res : vector<4xf32>
}

// -----

// gfx90a i8 MFMA tops out at K=16 (16x16); 16x16x32 is gfx942+.
// CHECK: MFMA 16x16x32 is not available on gfx90a (CDNA2)
func.func @widek_i8_rejected_on_gfx90a(%a: vector<8xi8>, %b: vector<8xi8>, %c: vector<4xi32>) -> vector<4xi32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x32, (i8, i8) -> i32>>
  %res = fly.mma_atom_call_ssa(%atom, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x32, (i8, i8) -> i32>>, vector<8xi8>, vector<8xi8>, vector<4xi32>) -> vector<4xi32>
  return %res : vector<4xi32>
}

// -----

// The CDNA2-valid K=16 i8 MFMA is accepted and lowers normally on gfx90a.
// CHECK: rocdl.mfma.i32.16x16x16i8
func.func @i8_k16_ok_on_gfx90a(%a: vector<4xi8>, %b: vector<4xi8>, %c: vector<4xi32>) -> vector<4xi32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (i8, i8) -> i32>>
  %res = fly.mma_atom_call_ssa(%atom, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (i8, i8) -> i32>>, vector<4xi8>, vector<4xi8>, vector<4xi32>) -> vector<4xi32>
  return %res : vector<4xi32>
}
