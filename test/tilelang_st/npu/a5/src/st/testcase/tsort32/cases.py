#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You can not use the file except of compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tsort32 ST test cases.

Each case defines:
  - name:        case identifier, used as subdirectory name and by main.cpp kCases[].
  - dtype:       numpy dtype (e.g. np.float32).
  - src_shape:   (rows, cols) — allocated source tile dimensions.
  - idx_shape:   (rows, cols) — allocated index tile dimensions (can be 1 x cols for shared idx).
  - dst_shape:   (rows, cols) — allocated destination tile dimensions.
                 For f32: dst_cols = src_cols * 4 (buffer allocation, but valid region is src_cols * 2).
                 For f16: dst_cols = src_cols * 2.
  - valid_shape: (valid_rows, valid_cols) — effective computation region.
                 valid_cols must be multiple of 32 (BLOCK_SIZE).
  - idx_vshape:  (idx_valid_rows, idx_valid_cols) — idx valid region.
                 If idx_valid_rows == 1, same idx is used for all rows.
  - dst_vshape:  (dst_valid_rows, dst_valid_cols) — dst valid region.
                 For f32: dst_vcols = src_vcols * 2 (stride coef = 2, interleaved value+index).
  - eps:         tolerance for numpy.allclose (atol and rtol).

tsort32 semantics:
  - Sorts data in 32-element blocks using vbitsort.
  - Output format: interleaved (sorted_value, original_index) pairs with stride coef = 2.
  - For each 32-element block, the output contains sorted values and their original indices.
  - Each pair occupies 2 element positions: [value0, idx0, value1, idx1, ...]

gen_data.py and compare.py both import this list to avoid redundant definitions.
"""

import numpy as np

CASES = [
    {
        "name": "f32_1x32",
        "dtype": np.float32,
        "src_shape": (1, 32),
        "idx_shape": (1, 32),
        "dst_shape": (1, 128),      # buffer allocation (src_cols * 4)
        "valid_shape": (1, 32),
        "idx_vshape": (1, 32),
        "dst_vshape": (1, 64),      # actual valid output: src_cols * stride_coef = 32 * 2
        "eps": 1e-6,
    },
    {
        "name": "f32_1x64",
        "dtype": np.float32,
        "src_shape": (1, 64),
        "idx_shape": (1, 64),
        "dst_shape": (1, 256),      # buffer allocation (src_cols * 4)
        "valid_shape": (1, 64),
        "idx_vshape": (1, 64),
        "dst_vshape": (1, 128),     # actual valid output: src_cols * stride_coef = 64 * 2
        "eps": 1e-6,
    },
    {
        "name": "f32_16x32",
        "dtype": np.float32,
        "src_shape": (16, 32),
        "idx_shape": (16, 32),
        "dst_shape": (16, 128),     # buffer allocation (src_cols * 4)
        "valid_shape": (16, 32),
        "idx_vshape": (16, 32),
        "dst_vshape": (16, 64),     # actual valid output: src_cols * stride_coef = 32 * 2
        "eps": 1e-6,
    },
    {
        "name": "f32_16x64_shared_idx",
        "dtype": np.float32,
        "src_shape": (16, 64),
        "idx_shape": (1, 64),       # shared idx for all rows
        "dst_shape": (16, 256),     # buffer allocation (src_cols * 4)
        "valid_shape": (16, 64),
        "idx_vshape": (1, 64),      # idx_valid_rows = 1 means shared idx
        "dst_vshape": (16, 128),    # actual valid output: src_cols * stride_coef = 64 * 2
        "eps": 1e-6,
    },
]