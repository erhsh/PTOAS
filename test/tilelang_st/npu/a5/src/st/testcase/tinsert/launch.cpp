// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <stdint.h>

#ifndef AICORE
#define AICORE [aicore]
#endif

// Case 0: f32 8x8 -> 16x16 idx(0,0)
extern "C" __global__ AICORE void TINSERT_f32_8x8_to_16x16_idx0_0(__gm__ float *src, __gm__ float *dst_init, __gm__ float *out);

void LaunchTINSERT_f32_8x8_to_16x16_idx0_0(float *src, float *dst_init, float *out, void *stream) {
    TINSERT_f32_8x8_to_16x16_idx0_0<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst_init, (__gm__ float *)out);
}

// Case 1: f32 8x8 -> 16x16 idx(4,8)
extern "C" __global__ AICORE void TINSERT_f32_8x8_to_16x16_idx4_8(__gm__ float *src, __gm__ float *dst_init, __gm__ float *out);

void LaunchTINSERT_f32_8x8_to_16x16_idx4_8(float *src, float *dst_init, float *out, void *stream) {
    TINSERT_f32_8x8_to_16x16_idx4_8<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst_init, (__gm__ float *)out);
}

// Case 2: f16 16x16 -> 32x32 idx(8,16)
extern "C" __global__ AICORE void TINSERT_f16_16x16_to_32x32_idx8_16(__gm__ uint16_t *src, __gm__ uint16_t *dst_init, __gm__ uint16_t *out);

void LaunchTINSERT_f16_16x16_to_32x32_idx8_16(uint16_t *src, uint16_t *dst_init, uint16_t *out, void *stream) {
    TINSERT_f16_16x16_to_32x32_idx8_16<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint16_t *)dst_init, (__gm__ uint16_t *)out);
}

// Case 3: f32 16x16 v_shape(16,10) -> 32x16 idx(4,0) aligned_vector path
extern "C" __global__ AICORE void TINSERT_f32_16x16_v10_to_32x16_idx4_0_aligned(__gm__ float *src, __gm__ float *dst_init, __gm__ float *out);

void LaunchTINSERT_f32_16x16_v10_to_32x16_idx4_0_aligned(float *src, float *dst_init, float *out, void *stream) {
    TINSERT_f32_16x16_v10_to_32x16_idx4_0_aligned<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst_init, (__gm__ float *)out);
}

// Case 4: i32 8x8 -> 16x16 idx(0,0)
extern "C" __global__ AICORE void TINSERT_i32_8x8_to_16x16_idx0_0(__gm__ int32_t *src, __gm__ int32_t *dst_init, __gm__ int32_t *out);

void LaunchTINSERT_i32_8x8_to_16x16_idx0_0(int32_t *src, int32_t *dst_init, int32_t *out, void *stream) {
    TINSERT_i32_8x8_to_16x16_idx0_0<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int32_t *)dst_init, (__gm__ int32_t *)out);
}

// Case 7: f32 16x16 v_shape(16,6) -> 32x16 idx(0,0) aligned_vector path
extern "C" __global__ AICORE void TINSERT_f32_16x16_v6_to_32x16_idx0_0_aligned(__gm__ float *src, __gm__ float *dst_init, __gm__ float *out);

void LaunchTINSERT_f32_16x16_v6_to_32x16_idx0_0_aligned(float *src, float *dst_init, float *out, void *stream) {
    TINSERT_f32_16x16_v6_to_32x16_idx0_0_aligned<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst_init, (__gm__ float *)out);
}

// Case 8: f32 64x64 -> 128x128 idx(0,0) large
extern "C" __global__ AICORE void TINSERT_f32_64x64_to_128x128_idx0_0_large(__gm__ float *src, __gm__ float *dst_init, __gm__ float *out);

void LaunchTINSERT_f32_64x64_to_128x128_idx0_0_large(float *src, float *dst_init, float *out, void *stream) {
    TINSERT_f32_64x64_to_128x128_idx0_0_large<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst_init, (__gm__ float *)out);
}

// Case 9: f32 64x64 -> 128x128 idx(32,32) large
extern "C" __global__ AICORE void TINSERT_f32_64x64_to_128x128_idx32_32_large(__gm__ float *src, __gm__ float *dst_init, __gm__ float *out);

void LaunchTINSERT_f32_64x64_to_128x128_idx32_32_large(float *src, float *dst_init, float *out, void *stream) {
    TINSERT_f32_64x64_to_128x128_idx32_32_large<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst_init, (__gm__ float *)out);
}

// Case 10: f16 32x32 -> 64x64 idx(16,16) large
extern "C" __global__ AICORE void TINSERT_f16_32x32_to_64x64_idx16_16_large(__gm__ uint16_t *src, __gm__ uint16_t *dst_init, __gm__ uint16_t *out);

void LaunchTINSERT_f16_32x32_to_64x64_idx16_16_large(uint16_t *src, uint16_t *dst_init, uint16_t *out, void *stream) {
    TINSERT_f16_32x32_to_64x64_idx16_16_large<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint16_t *)dst_init, (__gm__ uint16_t *)out);
}

// Case 11: f32 8x8 -> 16x16 idx(8,8) edge
extern "C" __global__ AICORE void TINSERT_f32_8x8_to_16x16_idx8_8_edge(__gm__ float *src, __gm__ float *dst_init, __gm__ float *out);

void LaunchTINSERT_f32_8x8_to_16x16_idx8_8_edge(float *src, float *dst_init, float *out, void *stream) {
    TINSERT_f32_8x8_to_16x16_idx8_8_edge<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst_init, (__gm__ float *)out);
}

// Case 13: f32 32x8 -> 64x16 idx(16,0) row shape
extern "C" __global__ AICORE void TINSERT_f32_32x8_to_64x16_idx16_0_row_shape(__gm__ float *src, __gm__ float *dst_init, __gm__ float *out);

void LaunchTINSERT_f32_32x8_to_64x16_idx16_0_row_shape(float *src, float *dst_init, float *out, void *stream) {
    TINSERT_f32_32x8_to_64x16_idx16_0_row_shape<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst_init, (__gm__ float *)out);
}

// Case 16: f32 8x32 -> 16x64 idx(0,16) col shape
extern "C" __global__ AICORE void TINSERT_f32_8x32_to_16x64_idx0_16_col_shape(__gm__ float *src, __gm__ float *dst_init, __gm__ float *out);

void LaunchTINSERT_f32_8x32_to_16x64_idx0_16_col_shape(float *src, float *dst_init, float *out, void *stream) {
    TINSERT_f32_8x32_to_16x64_idx0_16_col_shape<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst_init, (__gm__ float *)out);
}

// Case: i8 32x32 -> 64x64 idx(0,32) DMA path
extern "C" __global__ AICORE void TINSERT_i8_32x32_to_64x64_idx0_32(__gm__ int8_t *src, __gm__ int8_t *dst_init, __gm__ int8_t *out);

void LaunchTINSERT_i8_32x32_to_64x64_idx0_32(int8_t *src, int8_t *dst_init, int8_t *out, void *stream) {
    TINSERT_i8_32x32_to_64x64_idx0_32<<<1, nullptr, stream>>>((__gm__ int8_t *)src, (__gm__ int8_t *)dst_init, (__gm__ int8_t *)out);
}

// Case: i8 16x32 v_shape(16,20) -> 32x32 idx(0,0) aligned_vector path
extern "C" __global__ AICORE void TINSERT_i8_16x32_v20_to_32x32_idx0_0_aligned(__gm__ int8_t *src, __gm__ int8_t *dst_init, __gm__ int8_t *out);

void LaunchTINSERT_i8_16x32_v20_to_32x32_idx0_0_aligned(int8_t *src, int8_t *dst_init, int8_t *out, void *stream) {
    TINSERT_i8_16x32_v20_to_32x32_idx0_0_aligned<<<1, nullptr, stream>>>((__gm__ int8_t *)src, (__gm__ int8_t *)dst_init, (__gm__ int8_t *)out);
}

// Case: f16 16x16 v_shape(16,12) -> 32x32 idx(2,0) aligned_vector path
extern "C" __global__ AICORE void TINSERT_f16_16x16_v12_to_32x32_idx2_0_aligned(__gm__ uint16_t *src, __gm__ uint16_t *dst_init, __gm__ uint16_t *out);

void LaunchTINSERT_f16_16x16_v12_to_32x32_idx2_0_aligned(uint16_t *src, uint16_t *dst_init, uint16_t *out, void *stream) {
    TINSERT_f16_16x16_v12_to_32x32_idx2_0_aligned<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint16_t *)dst_init, (__gm__ uint16_t *)out);
}

// Case: i32 16x16 v_shape(16,10) -> 32x16 idx(4,0) aligned_vector path
extern "C" __global__ AICORE void TINSERT_i32_16x16_v10_to_32x16_idx4_0_aligned(__gm__ int32_t *src, __gm__ int32_t *dst_init, __gm__ int32_t *out);

void LaunchTINSERT_i32_16x16_v10_to_32x16_idx4_0_aligned(int32_t *src, int32_t *dst_init, int32_t *out, void *stream) {
    TINSERT_i32_16x16_v10_to_32x16_idx4_0_aligned<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int32_t *)dst_init, (__gm__ int32_t *)out);
}

// Case: f32 16x24 v_shape(16,16) -> 32x24 idx(4,0) DMA stride-gap path
extern "C" __global__ AICORE void TINSERT_f32_16x24_v16_to_32x24_idx4_0_gap_dma(__gm__ float *src, __gm__ float *dst_init, __gm__ float *out);

void LaunchTINSERT_f32_16x24_v16_to_32x24_idx4_0_gap_dma(float *src, float *dst_init, float *out, void *stream) {
    TINSERT_f32_16x24_v16_to_32x24_idx4_0_gap_dma<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst_init, (__gm__ float *)out);
}

// Case: f32 8x8 -> 16x16 idx(0,3) unaligned_vector path
extern "C" __global__ AICORE void TINSERT_f32_8x8_to_16x16_idx0_3_unaligned(__gm__ float *src, __gm__ float *dst_init, __gm__ float *out);

void LaunchTINSERT_f32_8x8_to_16x16_idx0_3_unaligned(float *src, float *dst_init, float *out, void *stream) {
    TINSERT_f32_8x8_to_16x16_idx0_3_unaligned<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst_init, (__gm__ float *)out);
}

// Case: f16 16x16 -> 32x32 idx(0,5) unaligned_vector path
extern "C" __global__ AICORE void TINSERT_f16_16x16_to_32x32_idx0_5_unaligned(__gm__ uint16_t *src, __gm__ uint16_t *dst_init, __gm__ uint16_t *out);

void LaunchTINSERT_f16_16x16_to_32x32_idx0_5_unaligned(uint16_t *src, uint16_t *dst_init, uint16_t *out, void *stream) {
    TINSERT_f16_16x16_to_32x32_idx0_5_unaligned<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint16_t *)dst_init, (__gm__ uint16_t *)out);
}