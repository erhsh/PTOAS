# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

import numpy as np
from cases import CASES
from st_common import validate_cases, setup_case_rng, save_case_data

validate_cases(CASES)

for case in CASES:
    setup_case_rng(case)

    dtype = case["dtype"]
    shape = case["shape"]
    valid_shape = case["valid_shape"]
    dtype_name = dtype.__name__

    vr, vc = valid_shape

    # Generate data based on dtype
    if dtype_name == "float32":
        # Float: random values in reasonable range (avoid extreme)
        input1 = np.random.randint(1, 10, size=shape).astype(dtype)
        input2 = np.random.randint(1, 10, size=shape).astype(dtype)
    elif dtype_name in ("uint32", "int32"):
        # Integer 32-bit: include boundary cases for high-precision testing
        input1 = np.random.randint(1, 1000, size=shape, dtype=dtype)
        input2 = np.random.randint(1, 1000, size=shape, dtype=dtype)

        # Boundary cases to trigger precision issues
        # Row 0: large dividend / small divisor
        if dtype_name == "uint32":
            input1[0, 0] = np.uint32(0xFFFFFFFF)  # max uint32
            input2[0, 0] = np.uint32(1)
            input1[0, 1] = np.uint32(0xFFFFFFFF)
            input2[0, 1] = np.uint32(2)
            input1[0, 2] = np.uint32(0x7FFFFFFF)  # mid value
            input2[0, 2] = np.uint32(0x7FFFFFFF)
        else:  # int32
            input1[0, 0] = np.int32(0x7FFFFFFF)   # max int32
            input2[0, 0] = np.int32(1)
            input1[0, 1] = np.int32(0x7FFFFFFF)
            input2[0, 1] = np.int32(2)
            input1[0, 2] = np.int32(-0x7FFFFFFF)  # negative large
            input2[0, 2] = np.int32(1)
            input1[0, 3] = np.int32(0x7FFFFFFF)
            input2[0, 3] = np.int32(-1)           # negative divisor

        # Row 1: small dividend / large divisor (quotient should be 0)
        input1[1, 0] = dtype(1)
        input2[1, 0] = dtype(1000)
        input1[1, 1] = dtype(100)
        input2[1, 1] = dtype(1000)
    elif dtype_name == "uint16":
        # Integer 16-bit: include boundary cases
        input1 = np.random.randint(1, 100, size=shape, dtype=dtype)
        input2 = np.random.randint(1, 100, size=shape, dtype=dtype)

        # Boundary cases
        input1[0, 0] = np.uint16(0xFFFF)  # max uint16
        input2[0, 0] = np.uint16(1)
        input1[0, 1] = np.uint16(0xFFFF)
        input2[0, 1] = np.uint16(2)
        input1[1, 0] = np.uint16(1)
        input2[1, 0] = np.uint16(0xFFFF)
    else:
        # Generic fallback
        input1 = np.random.randint(1, 10, size=shape).astype(dtype)
        input2 = np.random.randint(1, 10, size=shape).astype(dtype)

    golden = np.zeros(shape, dtype=dtype)
    if dtype_name == "float32":
        # Float: use true division (/)
        golden[:vr, :vc] = (input1[:vr, :vc] / input2[:vr, :vc]).astype(dtype, copy=False)
    else:
        # Integer: use floor division (//)
        golden[:vr, :vc] = (input1[:vr, :vc] // input2[:vr, :vc]).astype(dtype, copy=False)

    save_case_data(case["name"], {"input1": input1, "input2": input2, "golden": golden})
    print(f"[INFO] gen_data: {case['name']} shape={shape} valid_shape={valid_shape} dtype={dtype_name}")