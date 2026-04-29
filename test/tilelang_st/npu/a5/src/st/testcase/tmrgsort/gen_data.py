#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

import numpy as np
import os
import sys
import struct
import ctypes

# Add parent directory to path for st_common import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from st_common import setup_case_rng, save_case_data

from cases import CASES

BLOCK_NUM = 4
STRUCT_SIZE = 8  # bytes per structure (value + index)


def _to_tuple(shape):
    """Convert shape to tuple if needed."""
    if isinstance(shape, tuple):
        return shape
    return tuple(shape)


def get_elem_divisor(dtype):
    """Get element divisor based on dtype.

    A structure is 8 bytes:
    - f32 (4 bytes): 8 / 4 = 2 elems per struct
    - f16 (2 bytes): 8 / 2 = 4 elems per struct
    """
    if dtype == np.float16:
        return 4
    return 2


def write_value_index_pair(f, value, index, dtype):
    """Write a (value, index) pair to file.

    Format: value followed by index (uint32).
    For f16: value (2 bytes) + padding (2 bytes) + index (4 bytes).
    For f32: value (4 bytes) + index (4 bytes).
    """
    if dtype == np.float32:
        packed_data = struct.pack('fI', float(value), ctypes.c_uint32(index).value)
        f.write(packed_data)
    elif dtype == np.float16:
        # f16: 2 bytes + 2 bytes padding + 4 bytes index
        packed_data = struct.pack('e2xI', float(value), ctypes.c_uint32(index).value)
        f.write(packed_data)


def gen_golden_single(case):
    """Generate golden data for Format1 (single list internal block sorting).

    The input is divided into 4 blocks, each block is sorted independently,
    then the 4 sorted blocks are merged into a single sorted output.

    Output format: interleaved (sorted_value, original_index) pairs.
    """
    dtype = case["dtype"]
    src_shape = _to_tuple(case["src_shape"])
    dst_shape = _to_tuple(case["dst_shape"])
    valid_shape = _to_tuple(case["valid_shape"])
    block_len = case["block_len"]

    src_rows, src_cols = src_shape
    valid_rows, valid_cols = valid_shape
    elem_divisor = get_elem_divisor(dtype)

    # Generate random input data
    input_data = np.random.uniform(low=0.0, high=1.0, size=src_shape).astype(dtype)

    # Generate index data (0, 1, 2, ..., src_cols-1)
    idx_data = np.arange(src_cols, dtype=np.uint32)

    # Compute golden: 4-way merge sort
    # 1. Divide input into 4 blocks
    # 2. Sort each block in descending order
    # 3. Merge 4 sorted blocks into final sorted output

    # Each block has block_len elements
    # block_len_structs = block_len / elem_divisor (number of structures per block)

    # For vmrgsort4 Format1:
    # - Input: 4 blocks of size block_len (in elements)
    # - Each block is sorted, then merged
    # - Output: sorted (value, index) pairs

    # Sort each block independently
    blocks = []
    blocks_idx = []
    for i in range(BLOCK_NUM):
        block_start = i * block_len
        block_end = block_start + block_len
        if block_end > valid_cols:
            block_end = valid_cols

        block_data = input_data[0, block_start:block_end].copy()
        block_idx = idx_data[block_start:block_end].copy()

        # Sort in descending order
        sorted_indices = np.argsort(-block_data)
        sorted_values = block_data[sorted_indices]
        sorted_original_idx = block_idx[sorted_indices]

        blocks.append(sorted_values)
        blocks_idx.append(sorted_original_idx)

    # Merge 4 sorted blocks (descending order, largest first)
    # Using heap-like merge for 4 sorted lists
    merged_values = []
    merged_indices = []

    # Track current position in each block
    pos = [0] * BLOCK_NUM
    remaining = [len(blocks[i]) for i in range(BLOCK_NUM)]

    while any(remaining[i] > 0 for i in range(BLOCK_NUM)):
        # Find the block with the maximum value at current position
        max_val = None
        max_block = -1
        for i in range(BLOCK_NUM):
            if remaining[i] > 0:
                if max_val is None or blocks[i][pos[i]] > max_val:
                    max_val = blocks[i][pos[i]]
                    max_block = i

        if max_block >= 0:
            merged_values.append(blocks[max_block][pos[max_block]])
            merged_indices.append(blocks_idx[max_block][pos[max_block]])
            pos[max_block] += 1
            remaining[max_block] -= 1

    # Write input data - each block already sorted
    # For Format1: vmrgsort4 expects pre-sorted input blocks
    # So we write the blocks in sorted order (matching blocks after sorting)
    os.makedirs(case["name"], exist_ok=True)
    with open(os.path.join(case["name"], "input0.bin"), 'wb') as f:
        # Write each sorted block
        for i in range(BLOCK_NUM):
            for j in range(len(blocks[i])):
                write_value_index_pair(f, blocks[i][j], blocks_idx[i][j], dtype)

    # Write golden data
    with open(os.path.join(case["name"], "golden.bin"), 'wb') as f:
        for val, idx in zip(merged_values, merged_indices):
            write_value_index_pair(f, val, idx, dtype)

    print(f"[INFO] gen_data: {case['name']} src_shape={src_shape} dst_shape={dst_shape} dtype={dtype.__name__}")


def gen_golden_data():
    """Generate golden data for all cases."""
    for case in CASES:
        setup_case_rng(case)

        format_type = case.get("format", "single")
        if format_type == "single":
            gen_golden_single(case)
        else:
            print(f"[WARN] Unsupported format: {format_type} for case {case['name']}")


if __name__ == "__main__":
    gen_golden_data()