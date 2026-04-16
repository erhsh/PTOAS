// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You can not use the file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <stdint.h>

#ifndef AICORE
#define AICORE [aicore]
#endif

// Case 0: f32 16x64 full (src0/src1/dst all same valid region)
extern "C" __global__ AICORE void TPARTMUL_f32_16x64_full(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTPARTMUL_f32_16x64_full(float *a, float *b, float *c, void *stream) {
    TPARTMUL_f32_16x64_full<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}

// Case 1: f32 16x64 src1 row less (src1 valid region 8x64, dst 16x64)
extern "C" __global__ AICORE void TPARTMUL_f32_16x64_src1_row_less(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTPARTMUL_f32_16x64_src1_row_less(float *a, float *b, float *c, void *stream) {
    TPARTMUL_f32_16x64_src1_row_less<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}

// Case 2: f32 16x64 src1 col less (src1 valid region 16x32, dst 16x64)
extern "C" __global__ AICORE void TPARTMUL_f32_16x64_src1_col_less(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTPARTMUL_f32_16x64_src1_col_less(float *a, float *b, float *c, void *stream) {
    TPARTMUL_f32_16x64_src1_col_less<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}

// Case 3: f32 32x32 src1 row less (src1 valid region 16x32, dst 32x32)
extern "C" __global__ AICORE void TPARTMUL_f32_32x32_src1_row_less(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTPARTMUL_f32_32x32_src1_row_less(float *a, float *b, float *c, void *stream) {
    TPARTMUL_f32_32x32_src1_row_less<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}