# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tinsert ST test cases (Vec→Vec ND path).

Each case defines:
  - name:        case identifier
  - dtype:       numpy dtype
  - shape:       (rows, cols) — source tile dimensions
  - valid_shape: (v_row, v_col) — source valid region
  - dst_shape:   (rows, cols) — destination tile dimensions
  - dst_valid_shape: (v_row, v_col) — destination valid region
  - idx_row:     row offset in dst
  - idx_col:     column offset in dst
  - eps:         tolerance for numpy.allclose

Path coverage:
  DMA path:            valid_col * sizeof(dtype) % 32 == 0, valid != (1,1)
  Aligned vector path: valid_col * sizeof(dtype) % 32 != 0, valid != (1,1)
  Unaligned vector path: indexCol * sizeof(dtype) % 32 != 0, valid != (1,1)
  Scalar path:         valid == (1,1) — known bug, not tested

Dtype coverage:
  f32, f16, i32, i8 — active
  bf16 — commented out (requires ml_dtypes bfloat16 numpy extension)
"""

import numpy as np

CASES = [
    {
        "name": "f32_8x8_to_16x16_idx0_0",
        "dtype": np.float32,
        "shape": (8, 8),
        "valid_shape": (8, 8),
        "dst_shape": (16, 16),
        "dst_valid_shape": (16, 16),
        "idx_row": 0,
        "idx_col": 0,
        "eps": 1e-6,
    },
    {
        "name": "f32_8x8_to_16x16_idx4_8",
        "dtype": np.float32,
        "shape": (8, 8),
        "valid_shape": (8, 8),
        "dst_shape": (16, 16),
        "dst_valid_shape": (16, 16),
        "idx_row": 4,
        "idx_col": 8,
        "eps": 1e-6,
    },
    {
        "name": "f16_16x16_to_32x32_idx8_16",
        "dtype": np.float16,
        "shape": (16, 16),
        "valid_shape": (16, 16),
        "dst_shape": (32, 32),
        "dst_valid_shape": (32, 32),
        "idx_row": 8,
        "idx_col": 16,
        "eps": 1e-3,
    },
    {
        "name": "f32_16x16_v10_to_32x16_idx4_0_aligned",
        "dtype": np.float32,
        "shape": (16, 16),
        "valid_shape": (16, 10),
        "dst_shape": (32, 16),
        "dst_valid_shape": (32, 16),
        "idx_row": 4,
        "idx_col": 0,
        "eps": 1e-6,
    },
    {
        "name": "i32_8x8_to_16x16_idx0_0",
        "dtype": np.int32,
        "shape": (8, 8),
        "valid_shape": (8, 8),
        "dst_shape": (16, 16),
        "dst_valid_shape": (16, 16),
        "idx_row": 0,
        "idx_col": 0,
        "eps": 0,
    },
    # {
    #     "name": "bf16_8x8_to_16x16_idx0_0",
    #     "dtype": np.dtype("bfloat16"),  # requires ml_dtypes
    #     "shape": (8, 8),
    #     "valid_shape": (8, 8),
    #     "dst_shape": (16, 16),
    #     "dst_valid_shape": (16, 16),
    #     "idx_row": 0,
    #     "idx_col": 0,
    #     "eps": 1e-2,
    # },
    # {
    #     "name": "bf16_16x16_to_32x32_idx4_8",
    #     "dtype": np.dtype("bfloat16"),  # requires ml_dtypes
    #     "shape": (16, 16),
    #     "valid_shape": (16, 16),
    #     "dst_shape": (32, 32),
    #     "dst_valid_shape": (32, 32),
    #     "idx_row": 4,
    #     "idx_col": 8,
    #     "eps": 1e-2,
    # },
    {
        "name": "f32_16x16_v6_to_32x16_idx0_0_aligned",
        "dtype": np.float32,
        "shape": (16, 16),
        "valid_shape": (16, 6),
        "dst_shape": (32, 16),
        "dst_valid_shape": (32, 16),
        "idx_row": 0,
        "idx_col": 0,
        "eps": 1e-6,
    },
    {
        "name": "i8_32x32_to_64x64_idx0_32",
        "dtype": np.int8,
        "shape": (32, 32),
        "valid_shape": (32, 32),
        "dst_shape": (64, 64),
        "dst_valid_shape": (64, 64),
        "idx_row": 0,
        "idx_col": 32,
        "eps": 0,
    },
    {
        "name": "i8_16x32_v20_to_32x32_idx0_0_aligned",
        "dtype": np.int8,
        "shape": (16, 32),
        "valid_shape": (16, 20),
        "dst_shape": (32, 32),
        "dst_valid_shape": (32, 32),
        "idx_row": 0,
        "idx_col": 0,
        "eps": 0,
    },
    {
        "name": "f16_16x16_v12_to_32x32_idx2_0_aligned",
        "dtype": np.float16,
        "shape": (16, 16),
        "valid_shape": (16, 12),
        "dst_shape": (32, 32),
        "dst_valid_shape": (32, 32),
        "idx_row": 2,
        "idx_col": 0,
        "eps": 1e-3,
    },
    {
        "name": "i32_16x16_v10_to_32x16_idx4_0_aligned",
        "dtype": np.int32,
        "shape": (16, 16),
        "valid_shape": (16, 10),
        "dst_shape": (32, 16),
        "dst_valid_shape": (32, 16),
        "idx_row": 4,
        "idx_col": 0,
        "eps": 0,
    },
    {
        "name": "f32_16x24_v16_to_32x24_idx4_0_gap_dma",
        "dtype": np.float32,
        "shape": (16, 24),
        "valid_shape": (16, 16),
        "dst_shape": (32, 24),
        "dst_valid_shape": (32, 24),
        "idx_row": 4,
        "idx_col": 0,
        "eps": 1e-6,
    },
    {
        "name": "f32_64x64_to_128x128_idx0_0_large",
        "dtype": np.float32,
        "shape": (64, 64),
        "valid_shape": (64, 64),
        "dst_shape": (128, 128),
        "dst_valid_shape": (128, 128),
        "idx_row": 0,
        "idx_col": 0,
        "eps": 1e-6,
    },
    {
        "name": "f32_64x64_to_128x128_idx32_32_large",
        "dtype": np.float32,
        "shape": (64, 64),
        "valid_shape": (64, 64),
        "dst_shape": (128, 128),
        "dst_valid_shape": (128, 128),
        "idx_row": 32,
        "idx_col": 32,
        "eps": 1e-6,
    },
    {
        "name": "f16_32x32_to_64x64_idx16_16_large",
        "dtype": np.float16,
        "shape": (32, 32),
        "valid_shape": (32, 32),
        "dst_shape": (64, 64),
        "dst_valid_shape": (64, 64),
        "idx_row": 16,
        "idx_col": 16,
        "eps": 1e-3,
    },
    {
        "name": "f32_8x8_to_16x16_idx8_8_edge",
        "dtype": np.float32,
        "shape": (8, 8),
        "valid_shape": (8, 8),
        "dst_shape": (16, 16),
        "dst_valid_shape": (16, 16),
        "idx_row": 8,
        "idx_col": 8,
        "eps": 1e-6,
    },
    {
        "name": "f32_32x8_to_64x16_idx16_0_row_shape",
        "dtype": np.float32,
        "shape": (32, 8),
        "valid_shape": (32, 8),
        "dst_shape": (64, 16),
        "dst_valid_shape": (64, 16),
        "idx_row": 16,
        "idx_col": 0,
        "eps": 1e-6,
    },
    {
        "name": "f32_8x32_to_16x64_idx0_16_col_shape",
        "dtype": np.float32,
        "shape": (8, 32),
        "valid_shape": (8, 32),
        "dst_shape": (16, 64),
        "dst_valid_shape": (16, 64),
        "idx_row": 0,
        "idx_col": 16,
        "eps": 1e-6,
    },
    {
        "name": "f32_8x8_to_16x16_idx0_3_unaligned",
        "dtype": np.float32,
        "shape": (8, 8),
        "valid_shape": (8, 8),
        "dst_shape": (16, 16),
        "dst_valid_shape": (16, 16),
        "idx_row": 0,
        "idx_col": 3,
        "eps": 1e-6,
    },
    {
        "name": "f16_16x16_to_32x32_idx0_5_unaligned",
        "dtype": np.float16,
        "shape": (16, 16),
        "valid_shape": (16, 16),
        "dst_shape": (32, 32),
        "dst_valid_shape": (32, 32),
        "idx_row": 0,
        "idx_col": 5,
        "eps": 1e-3,
    },
]