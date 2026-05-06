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

// Case 0: f32 16x64
extern "C" __global__ AICORE void TDIV_f32_16x64(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTDIV_f32_16x64(float *a, float *b, float *c, void *stream) {
    TDIV_f32_16x64<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}

// Case 1: f32 32x32
extern "C" __global__ AICORE void TDIV_f32_32x32(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTDIV_f32_32x32(float *a, float *b, float *c, void *stream) {
    TDIV_f32_32x32<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}

// Case 2: ui32 16x64
extern "C" __global__ AICORE void TDIV_ui32_16x64(__gm__ uint32_t *a, __gm__ uint32_t *b, __gm__ uint32_t *c);

void LaunchTDIV_ui32_16x64(uint32_t *a, uint32_t *b, uint32_t *c, void *stream) {
    TDIV_ui32_16x64<<<1, nullptr, stream>>>((__gm__ uint32_t *)a, (__gm__ uint32_t *)b, (__gm__ uint32_t *)c);
}

// Case 3: ui16 16x64
extern "C" __global__ AICORE void TDIV_ui16_16x64(__gm__ uint16_t *a, __gm__ uint16_t *b, __gm__ uint16_t *c);

void LaunchTDIV_ui16_16x64(uint16_t *a, uint16_t *b, uint16_t *c, void *stream) {
    TDIV_ui16_16x64<<<1, nullptr, stream>>>((__gm__ uint16_t *)a, (__gm__ uint16_t *)b, (__gm__ uint16_t *)c);
}

// Case 4: i32 16x64
extern "C" __global__ AICORE void TDIV_i32_16x64(__gm__ int32_t *a, __gm__ int32_t *b, __gm__ int32_t *c);

void LaunchTDIV_i32_16x64(int32_t *a, int32_t *b, int32_t *c, void *stream) {
    TDIV_i32_16x64<<<1, nullptr, stream>>>((__gm__ int32_t *)a, (__gm__ int32_t *)b, (__gm__ int32_t *)c);
}