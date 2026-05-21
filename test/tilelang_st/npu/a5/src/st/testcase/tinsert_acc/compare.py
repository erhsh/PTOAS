# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Compare device output against golden for tinsert_acc ST smoke-test cases.

LIMITATION: Only matmul result (via mte_l0c_gm) is compared.
tinsert behavior is a smoke test (no numerical verification from GM).
"""

import os
import sys
import numpy as np

from cases import CASES
from st_common import result_cmp, style_fail, style_pass


def main():
    case_filter = sys.argv[1] if len(sys.argv) > 1 else None

    all_passed = True
    for case in CASES:
        if case_filter is not None and case["name"] != case_filter:
            continue

        case_dir = case["name"]
        shape_c = case["shape_c"]
        golden = np.fromfile(os.path.join(case_dir, "golden.bin"), dtype=np.float32).reshape(shape_c)
        output = np.fromfile(os.path.join(case_dir, "output.bin"), dtype=np.float32).reshape(shape_c)

        ok = result_cmp(golden, output, case["eps"])
        if ok:
            print(style_pass(f"[INFO] {case['name']}: matmul compare passed (tinsert is smoke-test only, note: {case['note']})"))
        else:
            print(style_fail(f"[ERROR] {case['name']}: matmul compare failed"))
            all_passed = False

    if not all_passed:
        sys.exit(2)
    print(style_pass("[INFO] all cases passed (tinsert is smoke-test only)"))


if __name__ == "__main__":
    main()