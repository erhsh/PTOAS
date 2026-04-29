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

// Case: f32_single_1x256_b64
extern "C" __global__ AICORE void TMRGSORT_f32_single_1x256_b64(__gm__ float *src, __gm__ float *dst);

void LaunchTMRGSORT_f32_single_1x256_b64(float *src, float *dst, void *stream) {
    TMRGSORT_f32_single_1x256_b64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

// Case: f32_single_1x512_b128
extern "C" __global__ AICORE void TMRGSORT_f32_single_1x512_b128(__gm__ float *src, __gm__ float *dst);

void LaunchTMRGSORT_f32_single_1x512_b128(float *src, float *dst, void *stream) {
    TMRGSORT_f32_single_1x512_b128<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

// Case: f16_single_1x256_b64
extern "C" __global__ AICORE void TMRGSORT_f16_single_1x256_b64(__gm__ uint16_t *src, __gm__ uint16_t *dst);

void LaunchTMRGSORT_f16_single_1x256_b64(uint16_t *src, uint16_t *dst, void *stream) {
    TMRGSORT_f16_single_1x256_b64<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint16_t *)dst);
}

// Case: f16_single_1x512_b128
extern "C" __global__ AICORE void TMRGSORT_f16_single_1x512_b128(__gm__ uint16_t *src, __gm__ uint16_t *dst);

void LaunchTMRGSORT_f16_single_1x512_b128(uint16_t *src, uint16_t *dst, void *stream) {
    TMRGSORT_f16_single_1x512_b128<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint16_t *)dst);
}

// Case: f32_single_1x1024_b256
extern "C" __global__ AICORE void TMRGSORT_f32_single_1x1024_b256(__gm__ float *src, __gm__ float *dst);

void LaunchTMRGSORT_f32_single_1x1024_b256(float *src, float *dst, void *stream) {
    TMRGSORT_f32_single_1x1024_b256<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}