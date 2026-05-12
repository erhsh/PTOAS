#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

import numpy as np
import struct
from cases import CASES
from st_common import validate_cases, setup_case_rng, save_case_data

validate_cases(CASES)

# Threshold values for high precision algorithm (bit pattern as float)
# These define the boundary between normal and subnormal numbers
_F16_THRESHOLD = struct.unpack('!e', bytes.fromhex('03FF'))[0]  # ~6.1e-5
_F32_THRESHOLD = struct.unpack('!f', bytes.fromhex('007FFFFF'))[0]  # ~1.18e-38


def _generate_log_input(dtype, shape, is_high_precision):
    """Generate input data for TLog test.

    For high precision cases, we need to test subnormal numbers (values below threshold)
    to trigger the precision compensation logic.

    Args:
        dtype: numpy dtype (np.float16 or np.float32)
        shape: tuple of (rows, cols)
        is_high_precision: whether this is a HIGH_PRECISION test case

    Returns:
        input array with appropriate data range
    """
    if is_high_precision:
        # High precision test: need values BELOW threshold to trigger compensation
        # threshold defines the boundary: values < threshold are subnormal
        if dtype == np.float16:
            threshold = _F16_THRESHOLD  # ~6.1e-5
            # Strategy: generate values spanning threshold boundary
            # 50% subnormal (below threshold), 50% normal (above threshold)
            n_elements = shape[0] * shape[1]
            n_subnormal = n_elements // 2
            n_normal = n_elements - n_subnormal

            # Subnormal range: threshold/100 to threshold (e.g., 6e-7 to 6e-5)
            subnormal_vals = np.random.uniform(threshold / 100, threshold, n_subnormal)
            # Normal range: threshold to 10.0
            normal_vals = np.random.uniform(threshold, 10.0, n_normal)

            combined = np.concatenate([subnormal_vals, normal_vals])
            np.random.shuffle(combined)
            input = combined.reshape(shape).astype(dtype)
        else:  # np.float32
            threshold = _F32_THRESHOLD  # ~1.18e-38
            n_elements = shape[0] * shape[1]
            n_subnormal = n_elements // 2
            n_normal = n_elements - n_subnormal

            # For f32, threshold is extremely small, use log-uniform distribution
            # Subnormal: log-uniform from threshold/1000 to threshold
            log_low = np.log(threshold / 1000)
            log_high = np.log(threshold)
            subnormal_vals = np.exp(np.random.uniform(log_low, log_high, n_subnormal))
            # Normal: uniform from threshold*10 to 10.0
            normal_vals = np.random.uniform(threshold * 10, 10.0, n_normal)

            combined = np.concatenate([subnormal_vals, normal_vals])
            np.random.shuffle(combined)
            input = combined.reshape(shape).astype(dtype)
    else:
        # Default precision test: normal positive values
        # Avoid values too close to zero to prevent domain issues
        input = np.random.uniform(0.1, 10.0, size=shape).astype(dtype)

    return input


for case in CASES:
    setup_case_rng(case)

    dtype = case["dtype"]
    shape = case["shape"]
    valid_shape = case["valid_shape"]
    is_high_precision = case.get("precision_mode") == "HIGH_PRECISION"

    input = _generate_log_input(dtype, shape, is_high_precision)

    golden = np.zeros(shape, dtype=dtype)
    vr, vc = valid_shape
    golden[:vr, :vc] = np.log(input[:vr, :vc]).astype(dtype, copy=False)

    save_case_data(case["name"], {"input": input, "golden": golden})
    print(f"[INFO] gen_data: {case['name']} shape={shape} valid_shape={valid_shape} dtype={dtype.__name__}"
          f" high_precision={is_high_precision}")
