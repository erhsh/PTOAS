# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tmrgsort ST test cases.

Each case defines:
  - name:        case identifier, used as subdirectory name and by main.cpp kCases[].
  - dtype:       numpy dtype (e.g. np.float32, np.float16).
  - format:      "single" for Format1 (1-list internal block sorting),
                 "multi" for Format2-4 (multi-list merge sort).
  - src_shape:   (rows, cols) - allocated source tile dimensions.
                 For Format1: single input list.
                 For multi-list: list of shapes for each input.
  - dst_shape:   (rows, cols) - allocated destination tile dimensions.
  - valid_shape: (valid_rows, valid_cols) - effective computation region.
  - block_len:   For Format1: block length in elements (must divide src_cols by 4).
  - list_num:    For multi-list: number of input lists (2, 3, or 4).
  - src_cols:    For multi-list: list of valid cols for each input list.
  - topk:        For multi-list: top-k output count.
  - exhausted:   For multi-list: whether to enable exhausted suspension.
  - eps:         tolerance for numpy.allclose (atol and rtol).

tmrgsort semantics:
  - Format1 (single list): Sorts 4 internal blocks of src using vmrgsort4.
    Each block is sorted independently, then merged.
    Output: interleaved (sorted_value, original_index) pairs.
  - Format2-4 (multi-list): Merges 2-4 sorted input lists into one sorted output.
    Each input list must already be sorted (in descending order).
    Output: top-k sorted elements from merged lists.

gen_data.py and compare.py both import this list to avoid redundant definitions.
"""

import numpy as np

CASES = [
    # Format1: single list (internal block sorting)
    # Basic f32 cases
    {
        "name": "f32_single_1x256_b64",
        "dtype": np.float32,
        "format": "single",
        "src_shape": (1, 256),
        "dst_shape": (1, 256),
        "valid_shape": (1, 256),
        "block_len": 64,  # 256 / 4 = 64 elements per block
        "eps": 1e-6,
    }
]