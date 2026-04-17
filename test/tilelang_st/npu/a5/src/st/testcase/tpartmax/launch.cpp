// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You can not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <stdint.h>

#ifndef AICORE
#define AICORE [aicore]
#endif

// Case 0: f32 64x64 full
extern "C" __global__ AICORE void TPARTMAX_f32_64x64_full(__gm__ void *a, __gm__ void *b, __gm__ void *c);

void LaunchTPARTMAX_f32_64x64_full(void *a, void *b, void *c, void *stream) {
    TPARTMAX_f32_64x64_full<<<1, nullptr, stream>>>((__gm__ void *)a, (__gm__ void *)b, (__gm__ void *)c);
}

// Case 1: f32 64x64 src0 row less
extern "C" __global__ AICORE void TPARTMAX_f32_64x64_src0_row_less(__gm__ void *a, __gm__ void *b, __gm__ void *c);

void LaunchTPARTMAX_f32_64x64_src0_row_less(void *a, void *b, void *c, void *stream) {
    TPARTMAX_f32_64x64_src0_row_less<<<1, nullptr, stream>>>((__gm__ void *)a, (__gm__ void *)b, (__gm__ void *)c);
}

// Case 2: f32 64x64 src0 col less
extern "C" __global__ AICORE void TPARTMAX_f32_64x64_src0_col_less(__gm__ void *a, __gm__ void *b, __gm__ void *c);

void LaunchTPARTMAX_f32_64x64_src0_col_less(void *a, void *b, void *c, void *stream) {
    TPARTMAX_f32_64x64_src0_col_less<<<1, nullptr, stream>>>((__gm__ void *)a, (__gm__ void *)b, (__gm__ void *)c);
}

// Case 3: f32 64x64 src1 row less
extern "C" __global__ AICORE void TPARTMAX_f32_64x64_src1_row_less(__gm__ void *a, __gm__ void *b, __gm__ void *c);

void LaunchTPARTMAX_f32_64x64_src1_row_less(void *a, void *b, void *c, void *stream) {
    TPARTMAX_f32_64x64_src1_row_less<<<1, nullptr, stream>>>((__gm__ void *)a, (__gm__ void *)b, (__gm__ void *)c);
}

// Case 4: f32 64x64 src1 col less
extern "C" __global__ AICORE void TPARTMAX_f32_64x64_src1_col_less(__gm__ void *a, __gm__ void *b, __gm__ void *c);

void LaunchTPARTMAX_f32_64x64_src1_col_less(void *a, void *b, void *c, void *stream) {
    TPARTMAX_f32_64x64_src1_col_less<<<1, nullptr, stream>>>((__gm__ void *)a, (__gm__ void *)b, (__gm__ void *)c);
}

// Case 5: f16 8x48 src0 col less
extern "C" __global__ AICORE void TPARTMAX_f16_8x48_src0_col_less(__gm__ void *a, __gm__ void *b, __gm__ void *c);

void LaunchTPARTMAX_f16_8x48_src0_col_less(void *a, void *b, void *c, void *stream) {
    TPARTMAX_f16_8x48_src0_col_less<<<1, nullptr, stream>>>((__gm__ void *)a, (__gm__ void *)b, (__gm__ void *)c);
}

// Case 6: f16 8x768 src0 col less
extern "C" __global__ AICORE void TPARTMAX_f16_8x768_src0_col_less(__gm__ void *a, __gm__ void *b, __gm__ void *c);

void LaunchTPARTMAX_f16_8x768_src0_col_less(void *a, void *b, void *c, void *stream) {
    TPARTMAX_f16_8x768_src0_col_less<<<1, nullptr, stream>>>((__gm__ void *)a, (__gm__ void *)b, (__gm__ void *)c);
}

// Case 7: i16 8x48 src1 col less
extern "C" __global__ AICORE void TPARTMAX_i16_8x48_src1_col_less(__gm__ void *a, __gm__ void *b, __gm__ void *c);

void LaunchTPARTMAX_i16_8x48_src1_col_less(void *a, void *b, void *c, void *stream) {
    TPARTMAX_i16_8x48_src1_col_less<<<1, nullptr, stream>>>((__gm__ void *)a, (__gm__ void *)b, (__gm__ void *)c);
}

// Case 8: i32 64x64 src0 row less
extern "C" __global__ AICORE void TPARTMAX_i32_64x64_src0_row_less(__gm__ void *a, __gm__ void *b, __gm__ void *c);

void LaunchTPARTMAX_i32_64x64_src0_row_less(void *a, void *b, void *c, void *stream) {
    TPARTMAX_i32_64x64_src0_row_less<<<1, nullptr, stream>>>((__gm__ void *)a, (__gm__ void *)b, (__gm__ void *)c);
}