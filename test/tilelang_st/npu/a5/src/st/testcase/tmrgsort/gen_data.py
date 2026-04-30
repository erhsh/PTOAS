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
        # f16: directly pack value (np.float16), not float(value)
        # Following pto-isa: struct.pack('e2xI', value, ...)
        packed_data = struct.pack('e2xI', value, ctypes.c_uint32(index).value)
        f.write(packed_data)


def gen_golden_single(case):
    """Generate golden data for Format1 (single list internal block sorting).

    Following pto-isa gen_data.py logic exactly:
    - cols = valid_cols // 2 (STRUCTURE count)
    - list_col = block_len // 2 (STRUCTURES per block)
    - block_lens = list_col * 4 (STRUCTURES per vmrgsort4 call)
    - block_lens_floats = block_len * 4 (FLOATS per vmrgsort4 call)

    Process:
    1. Generate random data (cols structures)
    2. Reshape into blocks (each list_col structures)
    3. Sort each block internally -> input0.bin
    4. Reshape into groups (each block_lens structures)
    5. Globally sort each group -> golden.bin
    """
    dtype = case["dtype"]
    src_shape = _to_tuple(case["src_shape"])
    dst_shape = _to_tuple(case["dst_shape"])
    valid_shape = _to_tuple(case["valid_shape"])
    block_len = case["block_len"]

    src_rows, src_cols = src_shape
    valid_rows, valid_cols = valid_shape
    
    # Structure units (following pto-isa)
    cols = valid_cols // 2  # total structures
    list_col = block_len // 2  # structures per block
    block_lens = list_col * 4  # structures per vmrgsort4 call
    block_lens_floats = block_len * 4  # floats per vmrgsort4 call
    
    repeat_times = cols // block_lens  # vmrgsort4 call times

    # Generate random data (1D array of structures)
    input_arr = np.random.uniform(low=0.0, high=1.0, size=(1, cols)).astype(dtype)
    idx_arr = np.arange(cols, dtype=np.uint32)

    # Step 1: Sort each block internally
    # Reshape to (total_blocks, list_col)
    input_reshaped = input_arr.reshape(-1, list_col)
    idx_reshaped = idx_arr.reshape(-1, list_col)
    
    # Sort each block descending
    sorted_indices = np.argsort(-input_reshaped, kind='stable', axis=1)
    sorted_input = np.take_along_axis(input_reshaped, sorted_indices, axis=1)
    sorted_idx = np.take_along_axis(idx_reshaped, sorted_indices, axis=1)

    # Flatten back -> input0.bin
    flat_input = sorted_input.flatten()
    flat_idx = sorted_idx.flatten()

    # Step 2: Generate golden (globally sort each group)
    # Take complete groups
    input_group = flat_input[:cols // block_lens * block_lens]
    idx_group = flat_idx[:cols // block_lens * block_lens]
    
    # Reshape to (repeat_times, block_lens)
    single_output_reshape = input_group.reshape(-1, block_lens)
    single_idx_reshape = idx_group.reshape(-1, block_lens)
    
    # Globally sort each group descending
    single_sorted_indices = np.argsort(-single_output_reshape, kind='stable', axis=1)
    golden_values = np.take_along_axis(single_output_reshape, single_sorted_indices, axis=1).flatten()
    golden_indices = np.take_along_axis(single_idx_reshape, single_sorted_indices, axis=1).flatten()

    # Handle remaining elements
    if cols % block_lens != 0:
        zeros_output = np.zeros(cols % block_lens, dtype=golden_values.dtype)
        zeros_index = np.zeros(cols % block_lens, dtype=np.uint32)
        golden_values = np.concatenate((golden_values, zeros_output))
        golden_indices = np.concatenate((golden_indices, zeros_index))

    os.makedirs(case["name"], exist_ok=True)
    with open(os.path.join(case["name"], "input0.bin"), 'wb') as f:
        for val, idx in zip(flat_input, flat_idx):
            write_value_index_pair(f, val, idx, dtype)

    with open(os.path.join(case["name"], "golden.bin"), 'wb') as f:
        for val, idx in zip(golden_values, golden_indices):
            write_value_index_pair(f, val, idx, dtype)

    print(f"[INFO] gen_data: {case['name']} src_cols={src_cols} valid_cols={valid_cols} "
          f"cols={cols} list_col={list_col} block_lens={block_lens} repeat_times={repeat_times}")


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