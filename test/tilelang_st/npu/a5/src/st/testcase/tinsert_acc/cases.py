# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tinsert_acc ST smoke-test cases (Acc→Mat NZ cube kernel).

LIMITATION: In a cube kernel, L1 data is NOT observable from GM.
Only the matmul result (via mte_l0c_gm) is verifiable. The tinsert
behavior (offset, layout, ReLU) is a smoke test only.

Each case defines:
  - name:        case identifier
  - shape_a:     (rows, cols) of matrix A (f16)
  - shape_b:     (rows, cols) of matrix B (f16)
  - shape_c:     (rows, cols) of matmul result (f32)
  - eps:         tolerance for numpy.allclose
  - note:        description of what is smoke-tested vs verified
"""

import numpy as np

CASES = [
    {
        "name": "acc_mat_nz_f32_16x16_idx0_0",
        "shape_a": (16, 16),
        "shape_b": (16, 16),
        "shape_c": (16, 16),
        "eps": 1e-2,
        "note": "tmatmul verified, tinsert Acc→Mat NZ (nz2nz) smoke",
    },
]