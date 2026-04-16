// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You can not use the file except of compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <stdint.h>

#ifndef AICORE
#define AICORE [aicore]
#endif

// Case 0: f32 1x32
extern "C" __global__ AICORE void TSORT32_f32_1x32(__gm__ float *src, __gm__ uint32_t *idx, __gm__ float *dst);

void LaunchTSORT32_f32_1x32(float *src, uint32_t *idx, float *dst, void *stream) {
    TSORT32_f32_1x32<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint32_t *)idx, (__gm__ float *)dst);
}

// Case 1: f32 1x64
extern "C" __global__ AICORE void TSORT32_f32_1x64(__gm__ float *src, __gm__ uint32_t *idx, __gm__ float *dst);

void LaunchTSORT32_f32_1x64(float *src, uint32_t *idx, float *dst, void *stream) {
    TSORT32_f32_1x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint32_t *)idx, (__gm__ float *)dst);
}

// Case 2: f32 16x32
extern "C" __global__ AICORE void TSORT32_f32_16x32(__gm__ float *src, __gm__ uint32_t *idx, __gm__ float *dst);

void LaunchTSORT32_f32_16x32(float *src, uint32_t *idx, float *dst, void *stream) {
    TSORT32_f32_16x32<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint32_t *)idx, (__gm__ float *)dst);
}

// Case 3: f32 16x64 shared_idx
extern "C" __global__ AICORE void TSORT32_f32_16x64_shared_idx(__gm__ float *src, __gm__ uint32_t *idx, __gm__ float *dst);

void LaunchTSORT32_f32_16x64_shared_idx(float *src, uint32_t *idx, float *dst, void *stream) {
    TSORT32_f32_16x64_shared_idx<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint32_t *)idx, (__gm__ float *)dst);
}