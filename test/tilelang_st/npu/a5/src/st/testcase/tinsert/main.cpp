// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Host driver for TileLang tinsert ST — case-table driven.
// Each case launches a different kernel variant, reads/writes from per-case subdirectory.
// Numerical comparison is done externally by compare.py.

#include "acl/acl.h"
#include "test_common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>

using namespace PtoTestCommon;

void LaunchTINSERT_f32_8x8_to_16x16_idx0_0(float *src, float *dst_init, float *out, void *stream);
void LaunchTINSERT_f32_8x8_to_16x16_idx4_8(float *src, float *dst_init, float *out, void *stream);
void LaunchTINSERT_f16_16x16_to_32x32_idx8_16(uint16_t *src, uint16_t *dst_init, uint16_t *out, void *stream);
void LaunchTINSERT_f32_16x16_v10_to_32x16_idx4_0_aligned(float *src, float *dst_init, float *out, void *stream);
void LaunchTINSERT_i32_8x8_to_16x16_idx0_0(int32_t *src, int32_t *dst_init, int32_t *out, void *stream);
void LaunchTINSERT_f32_16x16_v6_to_32x16_idx0_0_aligned(float *src, float *dst_init, float *out, void *stream);
void LaunchTINSERT_f32_64x64_to_128x128_idx0_0_large(float *src, float *dst_init, float *out, void *stream);
void LaunchTINSERT_f32_64x64_to_128x128_idx32_32_large(float *src, float *dst_init, float *out, void *stream);
void LaunchTINSERT_f16_32x32_to_64x64_idx16_16_large(uint16_t *src, uint16_t *dst_init, uint16_t *out, void *stream);
void LaunchTINSERT_f32_8x8_to_16x16_idx8_8_edge(float *src, float *dst_init, float *out, void *stream);
void LaunchTINSERT_f32_32x8_to_64x16_idx16_0_row_shape(float *src, float *dst_init, float *out, void *stream);
void LaunchTINSERT_f32_8x32_to_16x64_idx0_16_col_shape(float *src, float *dst_init, float *out, void *stream);
void LaunchTINSERT_i8_32x32_to_64x64_idx0_32(int8_t *src, int8_t *dst_init, int8_t *out, void *stream);
void LaunchTINSERT_i8_16x32_v20_to_32x32_idx0_0_aligned(int8_t *src, int8_t *dst_init, int8_t *out, void *stream);
void LaunchTINSERT_f16_16x16_v12_to_32x32_idx2_0_aligned(uint16_t *src, uint16_t *dst_init, uint16_t *out, void *stream);
void LaunchTINSERT_i32_16x16_v10_to_32x16_idx4_0_aligned(int32_t *src, int32_t *dst_init, int32_t *out, void *stream);
void LaunchTINSERT_f32_16x24_v16_to_32x24_idx4_0_gap_dma(float *src, float *dst_init, float *out, void *stream);
void LaunchTINSERT_f32_8x8_to_16x16_idx0_3_unaligned(float *src, float *dst_init, float *out, void *stream);
void LaunchTINSERT_f16_16x16_to_32x32_idx0_5_unaligned(uint16_t *src, uint16_t *dst_init, uint16_t *out, void *stream);

struct TestCase {
    const char *name;
    void (*launch)(void *src, void *dst_init, void *out, void *stream);
    size_t      srcRows;
    size_t      srcCols;
    size_t      dstRows;
    size_t      dstCols;
    size_t      elemSize;
};

static const TestCase kCases[] = {
    {"f32_8x8_to_16x16_idx0_0",    (void(*)(void*,void*,void*,void*))LaunchTINSERT_f32_8x8_to_16x16_idx0_0,    8,  8,  16, 16, sizeof(float)},
    {"f32_8x8_to_16x16_idx4_8",    (void(*)(void*,void*,void*,void*))LaunchTINSERT_f32_8x8_to_16x16_idx4_8,    8,  8,  16, 16, sizeof(float)},
    {"f16_16x16_to_32x32_idx8_16", (void(*)(void*,void*,void*,void*))LaunchTINSERT_f16_16x16_to_32x32_idx8_16, 16, 16, 32, 32, sizeof(uint16_t)},
    {"f32_16x16_v10_to_32x16_idx4_0_aligned", (void(*)(void*,void*,void*,void*))LaunchTINSERT_f32_16x16_v10_to_32x16_idx4_0_aligned, 16, 16, 32, 16, sizeof(float)},
    {"i32_8x8_to_16x16_idx0_0",    (void(*)(void*,void*,void*,void*))LaunchTINSERT_i32_8x8_to_16x16_idx0_0,    8,  8,  16, 16, sizeof(int32_t)},
    {"f32_16x16_v6_to_32x16_idx0_0_aligned", (void(*)(void*,void*,void*,void*))LaunchTINSERT_f32_16x16_v6_to_32x16_idx0_0_aligned, 16, 16, 32, 16, sizeof(float)},
    {"f32_64x64_to_128x128_idx0_0_large", (void(*)(void*,void*,void*,void*))LaunchTINSERT_f32_64x64_to_128x128_idx0_0_large, 64, 64, 128, 128, sizeof(float)},
    {"f32_64x64_to_128x128_idx32_32_large", (void(*)(void*,void*,void*,void*))LaunchTINSERT_f32_64x64_to_128x128_idx32_32_large, 64, 64, 128, 128, sizeof(float)},
    {"f16_32x32_to_64x64_idx16_16_large", (void(*)(void*,void*,void*,void*))LaunchTINSERT_f16_32x32_to_64x64_idx16_16_large, 32, 32, 64, 64, sizeof(uint16_t)},
    {"f32_8x8_to_16x16_idx8_8_edge", (void(*)(void*,void*,void*,void*))LaunchTINSERT_f32_8x8_to_16x16_idx8_8_edge, 8, 8, 16, 16, sizeof(float)},
    {"f32_32x8_to_64x16_idx16_0_row_shape", (void(*)(void*,void*,void*,void*))LaunchTINSERT_f32_32x8_to_64x16_idx16_0_row_shape, 32, 8, 64, 16, sizeof(float)},
    {"f32_8x32_to_16x64_idx0_16_col_shape", (void(*)(void*,void*,void*,void*))LaunchTINSERT_f32_8x32_to_16x64_idx0_16_col_shape, 8, 32, 16, 64, sizeof(float)},
    {"i8_32x32_to_64x64_idx0_32",    (void(*)(void*,void*,void*,void*))LaunchTINSERT_i8_32x32_to_64x64_idx0_32,    32, 32, 64, 64, sizeof(int8_t)},
    {"i8_16x32_v20_to_32x32_idx0_0_aligned", (void(*)(void*,void*,void*,void*))LaunchTINSERT_i8_16x32_v20_to_32x32_idx0_0_aligned, 16, 32, 32, 32, sizeof(int8_t)},
    {"f16_16x16_v12_to_32x32_idx2_0_aligned", (void(*)(void*,void*,void*,void*))LaunchTINSERT_f16_16x16_v12_to_32x32_idx2_0_aligned, 16, 16, 32, 32, sizeof(uint16_t)},
    {"i32_16x16_v10_to_32x16_idx4_0_aligned", (void(*)(void*,void*,void*,void*))LaunchTINSERT_i32_16x16_v10_to_32x16_idx4_0_aligned, 16, 16, 32, 16, sizeof(int32_t)},
    {"f32_16x24_v16_to_32x24_idx4_0_gap_dma", (void(*)(void*,void*,void*,void*))LaunchTINSERT_f32_16x24_v16_to_32x24_idx4_0_gap_dma, 16, 24, 32, 24, sizeof(float)},
    {"f32_8x8_to_16x16_idx0_3_unaligned", (void(*)(void*,void*,void*,void*))LaunchTINSERT_f32_8x8_to_16x16_idx0_3_unaligned, 8, 8, 16, 16, sizeof(float)},
    {"f16_16x16_to_32x32_idx0_5_unaligned", (void(*)(void*,void*,void*,void*))LaunchTINSERT_f16_16x16_to_32x32_idx0_5_unaligned, 16, 16, 32, 32, sizeof(uint16_t)},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    int rc = 0;
    const size_t srcByteSize = tc.srcRows * tc.srcCols * tc.elemSize;
    const size_t dstByteSize = tc.dstRows * tc.dstCols * tc.elemSize;

    std::printf("[INFO] === case: %s (src=%zux%zu, dst=%zux%zu) ===\n",
                tc.name, tc.srcRows, tc.srcCols, tc.dstRows, tc.dstCols);

    std::string caseDir = std::string("./") + tc.name;

    void *srcHost = nullptr, *dstInitHost = nullptr, *outHost = nullptr;
    void *srcDevice = nullptr, *dstInitDevice = nullptr, *outDevice = nullptr;

    aclrtMallocHost((void **)(&srcHost), srcByteSize);
    aclrtMallocHost((void **)(&dstInitHost), dstByteSize);
    aclrtMallocHost((void **)(&outHost), dstByteSize);

    aclrtMalloc((void **)&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dstInitDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&outDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t srcFileSize = srcByteSize;
    size_t dstInitFileSize = dstByteSize;

    if (!ReadFile((caseDir + "/src_input.bin").c_str(), srcFileSize, srcHost, srcByteSize)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/src_input.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0 && !ReadFile((caseDir + "/dst_init.bin").c_str(), dstInitFileSize, dstInitHost, dstByteSize)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/dst_init.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0) {
        aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(dstInitDevice, dstByteSize, dstInitHost, dstByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

        tc.launch(srcDevice, dstInitDevice, outDevice, stream);

        aclrtSynchronizeStream(stream);
        aclrtMemcpy(outHost, dstByteSize, outDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), outHost, dstByteSize)) {
        std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (srcDevice != nullptr) aclrtFree(srcDevice);
    if (dstInitDevice != nullptr) aclrtFree(dstInitDevice);
    if (outDevice != nullptr) aclrtFree(outDevice);
    if (srcHost != nullptr) aclrtFreeHost(srcHost);
    if (dstInitHost != nullptr) aclrtFreeHost(dstInitHost);
    if (outHost != nullptr) aclrtFreeHost(outHost);

    if (rc == 0)
        std::printf("[INFO] case %s done\n", tc.name);
    return rc;
}

int main(int argc, char *argv[]) {
    const char *caseFilter = (argc > 1) ? argv[1] : nullptr;

    int rc = 0;
    int deviceId = 0;
    aclrtStream stream = nullptr;

    aclInit(nullptr);
    if (const char *envDevice = std::getenv("ACL_DEVICE_ID")) {
        deviceId = std::atoi(envDevice);
    }
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    for (size_t i = 0; i < kNumCases; ++i) {
        if (caseFilter != nullptr && std::strcmp(kCases[i].name, caseFilter) != 0) {
            continue;
        }
        int ret = RunCase(kCases[i], deviceId, stream);
        if (ret != 0) {
            std::fprintf(stderr, "[ERROR] case %s failed\n", kCases[i].name);
            rc = 1;
            break;
        }
    }

    if (stream != nullptr)
        aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return rc;
}