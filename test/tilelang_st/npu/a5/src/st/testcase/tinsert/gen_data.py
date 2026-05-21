# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Generate input and golden data for tinsert ST test cases (Vec→Vec ND path).

Semantics: dst[i + idx_row, j + idx_col] = src[i, j]
Golden = dst_init with src inserted at offset.
"""

import os
import sys
import numpy as np

from cases import CASES
from st_common import validate_cases, setup_case_rng, save_case_data


def main():
    validate_cases(CASES)
    case_filter = sys.argv[1] if len(sys.argv) > 1 else None

    for case in CASES:
        if case_filter is not None and case["name"] != case_filter:
            continue

        setup_case_rng(case)

        dtype = case["dtype"]
        src_shape = case["shape"]
        src_valid = case["valid_shape"]
        dst_shape = case["dst_shape"]
        idx_row = case["idx_row"]
        idx_col = case["idx_col"]

        if dtype in (np.uint8, np.int8, np.int32):
            src_data = np.random.randint(-128, 128 if dtype == np.int8 else 256, size=src_shape).astype(dtype)
        else:
            src_data = np.random.rand(*src_shape).astype(dtype)

        if dtype in (np.uint8, np.int8, np.int32):
            dst_init = np.random.randint(-128, 128 if dtype == np.int8 else 256, size=dst_shape).astype(dtype)
        else:
            dst_init = np.random.rand(*dst_shape).astype(dtype)

        golden = dst_init.copy()
        r_end = idx_row + src_valid[0]
        c_end = idx_col + src_valid[1]
        golden[idx_row:r_end, idx_col:c_end] = src_data[:src_valid[0], :src_valid[1]]

        save_case_data(case["name"], {"src_input": src_data, "dst_init": dst_init, "golden": golden})
        print(f"[INFO] gen_data: {case['name']} src={src_shape} dst={dst_shape} idx=({idx_row},{idx_col}) dtype={dtype.__name__}")


if __name__ == "__main__":
    main()