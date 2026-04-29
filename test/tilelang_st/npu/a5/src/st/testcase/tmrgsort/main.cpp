// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Host driver for TileLang tmrgsort ST — case-table driven.
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

// Kernel launch wrappers (defined in launch.cpp)
void LaunchTMRGSORT_f32_single_1x256_b64(float *src, float *dst, void *stream);
void LaunchTMRGSORT_f32_single_1x512_b128(float *src, float *dst, void *stream);
void LaunchTMRGSORT_f16_single_1x256_b64(uint16_t *src, uint16_t *dst, void *stream);
void LaunchTMRGSORT_f16_single_1x512_b128(uint16_t *src, uint16_t *dst, void *stream);
void LaunchTMRGSORT_f32_single_1x1024_b256(float *src, float *dst, void *stream);

using LaunchFn = void (*)(void *, void *, void *);

struct TestCase {
    const char *name;
    LaunchFn    launch;
    size_t      srcRows;
    size_t      srcCols;
    size_t      dstRows;
    size_t      dstCols;
    size_t      elemSize;    // bytes per element
    size_t      structSize;  // 8 bytes per (value, index) pair
};

static const TestCase kCases[] = {
    {"f32_single_1x256_b64",   reinterpret_cast<LaunchFn>(LaunchTMRGSORT_f32_single_1x256_b64),   1, 256,  1, 256,  sizeof(float),   8},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, aclrtStream stream) {
    int rc = 0;
    // Input file: (value, index) pairs, size = srcCols * structSize
    size_t srcFileSize = tc.srcRows * tc.srcCols * tc.structSize;
    // Output file: (value, index) pairs, size = dstCols * structSize
    size_t dstFileSize = tc.dstRows * tc.dstCols * tc.structSize;

    std::printf("[INFO] === case: %s (src=%zux%zu, dst=%zux%zu) ===\n",
                tc.name, tc.srcRows, tc.srcCols, tc.dstRows, tc.dstCols);

    std::string caseDir = std::string("./") + tc.name;

    void *srcHost = nullptr, *dstHost = nullptr;
    void *srcDevice = nullptr, *dstDevice = nullptr;

    aclrtMallocHost((void **)(&srcHost), srcFileSize);
    aclrtMallocHost((void **)(&dstHost), dstFileSize);

    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    if (!ReadFile((caseDir + "/input0.bin").c_str(), srcFileSize, srcHost, srcFileSize)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input0.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0) {
        aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

        tc.launch(srcDevice, dstDevice, stream);

        aclrtSynchronizeStream(stream);
        aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), dstHost, dstFileSize)) {
        std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (srcDevice != nullptr)
        aclrtFree(srcDevice);
    if (dstDevice != nullptr)
        aclrtFree(dstDevice);
    if (srcHost != nullptr)
        aclrtFreeHost(srcHost);
    if (dstHost != nullptr)
        aclrtFreeHost(dstHost);

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
        int ret = RunCase(kCases[i], stream);
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