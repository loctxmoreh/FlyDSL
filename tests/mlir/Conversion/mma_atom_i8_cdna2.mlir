// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-rewrite-func-signature --fly-canonicalize --fly-layout-lowering --convert-fly-to-rocdl | FileCheck %s

// gfx90a (CDNA2) i8 MFMA atoms: narrower K than CDNA3 (4 packed i8 -> i32 operands).
//   mfma_i32_16x16x16i8 (K=16) and mfma_i32_32x32x8i8 (K=8).

// CHECK-LABEL: @test_mma_i8_16x16x16
// CHECK-SAME: (%[[A:.*]]: vector<4xi8>, %[[B:.*]]: vector<4xi8>, %[[C:.*]]: vector<4xi32>)
func.func @test_mma_i8_16x16x16(
    %a: vector<4xi8>,
    %b: vector<4xi8>,
    %c: vector<4xi32>) -> vector<4xi32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (i8, i8) -> i32>>
  // CHECK: %[[A_CAST:.*]] = llvm.bitcast %[[A]] : vector<4xi8> to i32
  // CHECK: %[[B_CAST:.*]] = llvm.bitcast %[[B]] : vector<4xi8> to i32
  // CHECK: %[[RES:.*]] = rocdl.mfma.i32.16x16x16i8 %[[A_CAST]], %[[B_CAST]], %[[C]]
  %res = fly.mma_atom_call_ssa(%atom, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (i8, i8) -> i32>>, vector<4xi8>, vector<4xi8>, vector<4xi32>) -> vector<4xi32>
  return %res : vector<4xi32>
}

// CHECK-LABEL: @test_mma_i8_32x32x8
// CHECK-SAME: (%[[A:.*]]: vector<4xi8>, %[[B:.*]]: vector<4xi8>, %[[C:.*]]: vector<16xi32>)
func.func @test_mma_i8_32x32x8(
    %a: vector<4xi8>,
    %b: vector<4xi8>,
    %c: vector<16xi32>) -> vector<16xi32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.cdna3.mfma<32x32x8, (i8, i8) -> i32>>
  // CHECK: %[[A_CAST:.*]] = llvm.bitcast %[[A]] : vector<4xi8> to i32
  // CHECK: %[[B_CAST:.*]] = llvm.bitcast %[[B]] : vector<4xi8> to i32
  // CHECK: %[[RES:.*]] = rocdl.mfma.i32.32x32x8i8 %[[A_CAST]], %[[B_CAST]], %[[C]]
  %res = fly.mma_atom_call_ssa(%atom, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.cdna3.mfma<32x32x8, (i8, i8) -> i32>>, vector<4xi8>, vector<4xi8>, vector<16xi32>) -> vector<16xi32>
  return %res : vector<16xi32>
}
