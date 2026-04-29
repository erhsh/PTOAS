#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

import os
import sys
import numpy as np
import struct

# Add parent directory to path for st_common import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from st_common import result_cmp, style_fail, style_pass

from cases import CASES


def read_value_index_pairs(filepath, dtype, count):
    """Read interleaved (value, index) pairs from file.

    Format: value followed by index (uint32).
    For f16: value (2 bytes) + padding (2 bytes) + index (4 bytes) = 8 bytes per pair.
    For f32: value (4 bytes) + index (4 bytes) = 8 bytes per pair.
    """
    values = []
    indices = []

    struct_fmt = 'fI' if dtype == np.float32 else 'e2xI'
    struct_size = struct.calcsize(struct_fmt)

    with open(filepath, 'rb') as f:
        for _ in range(count):
            data = f.read(struct_size)
            if not data:
                break
            unpacked = struct.unpack(struct_fmt, data)
            values.append(unpacked[0])
            indices.append(unpacked[1])

    return np.array(values, dtype=dtype), np.array(indices, dtype=np.uint32)


def main():
    case_filter = sys.argv[1] if len(sys.argv) > 1 else None

    all_passed = True
    for case in CASES:
        if case_filter is not None and case["name"] != case_filter:
            continue

        dtype = case["dtype"]
        valid_shape = case["valid_shape"]
        valid_rows, valid_cols = valid_shape
        block_len = case["block_len"]

        # Number of output pairs = valid_cols (each element produces one (value, index) pair)
        # But actual output is 4-way merge of blocks, so output count = valid_cols
        output_count = valid_cols

        golden_vals, golden_indices = read_value_index_pairs(
            os.path.join(case["name"], "golden.bin"), dtype, output_count
        )
        output_vals, output_indices = read_value_index_pairs(
            os.path.join(case["name"], "output.bin"), dtype, output_count
        )

        # Compare values
        vals_ok = result_cmp(golden_vals, output_vals, case["eps"])
        # Compare indices (allow tolerance for exact match)
        indices_ok = np.allclose(golden_indices, output_indices, atol=0, rtol=0)

        if vals_ok and indices_ok:
            print(style_pass(f"[INFO] {case['name']}: compare passed"))
        else:
            if not vals_ok:
                print(style_fail(f"[ERROR] {case['name']}: values mismatch"))
            if not indices_ok:
                print(style_fail(f"[ERROR] {case['name']}: indices mismatch"))
            all_passed = False

    if not all_passed:
        sys.exit(2)
    print(style_pass("[INFO] all cases passed"))


if __name__ == "__main__":
    main()