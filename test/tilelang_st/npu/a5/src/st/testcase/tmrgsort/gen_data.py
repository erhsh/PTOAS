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


def gen_golden_multilist(case):
    """Generate golden data for Format2 (multi-list merge sort).
    
    Following pto-isa gen_data.py logic for multi-list:
    1. Generate sorted data for each input list (descending order)
    2. Concatenate all lists and globally sort (descending)
    3. Take top-k elements
    4. If exhausted=true, handle special termination logic
    
    Each input list is pre-sorted in descending order.
    Output is top-k merged sorted elements.
    """
    dtype = case["dtype"]
    list_num = case["list_num"]
    src_cols = case["src_cols"]  # structures per list
    topk = case["topk"]
    exhausted = case.get("exhausted", False)
    
    # Calculate actual cols (in elements) per src
    # Each structure = (value, index) pair = 8 bytes
    # For f32: 2 elements per structure (4 bytes value + 4 bytes index)
    # For f16: 4 elements per structure (2 bytes value + 2 bytes padding + 4 bytes index)
    elem_divisor = get_elem_divisor(dtype)
    
    # Generate sorted data for each input list
    output_arr_list = []
    output_idx_list = []
    last_data = []
    
    total_structures = sum(src_cols)
    
    for i in range(list_num):
        cols_i = src_cols[i]
        # Generate random data for this list
        input_arr = np.random.uniform(low=0.0, high=1.0, size=(1, cols_i)).astype(dtype)
        idx_arr = np.arange(cols_i, dtype=np.uint32).reshape(1, cols_i)  # Reshape to match input_arr
        
        # Sort in descending order
        sorted_indices = np.argsort(-input_arr, kind='stable', axis=1)
        sorted_input = np.take_along_axis(input_arr, sorted_indices, axis=1)
        sorted_idx = np.take_along_axis(idx_arr, sorted_indices, axis=1)
        
        # Flatten
        flat_input_i = sorted_input.flatten()
        flat_idx_i = sorted_idx.flatten()
        
        output_arr_list.append(flat_input_i)
        output_idx_list.append(flat_idx_i)
        
        # Track last element for exhausted case
        if cols_i > 0:
            last_data.append(flat_input_i[-1])
        else:
            last_data.append(0)
    
    # Concatenate and globally sort (descending)
    flat_input_group = np.concatenate(output_arr_list).flatten()
    flat_idx_group = np.concatenate(output_idx_list).flatten()
    
    sorted_indices_global = np.argsort(-flat_input_group, kind='stable')
    sorted_output_global = flat_input_group[sorted_indices_global]
    sorted_idx_global = flat_idx_group[sorted_indices_global]
    
    # Take top-k
    topk_sorted_output = sorted_output_global[:topk]
    topk_sorted_idx = sorted_idx_global[:topk]
    
    # Pad zeros if needed
    zeros_output = np.zeros(total_structures - topk, dtype=topk_sorted_output.dtype)
    zeros_index = np.zeros(total_structures - topk, dtype=np.uint32)
    topk_sorted_output_global = np.concatenate((topk_sorted_output, zeros_output))
    topk_sorted_idx_global = np.concatenate((topk_sorted_idx, zeros_index))
    
    # Handle exhausted case: zero elements after last_data occurrence
    if exhausted and list_num <= 4:
        for i in range(list_num):
            # Find last_data in output and zero elements after it
            zero_index = -1
            for j in range(len(topk_sorted_output_global) - 1, -1, -1):
                if topk_sorted_output_global[j] == last_data[i]:
                    zero_index = j
                    break
            if zero_index >= 0:
                # Zero elements after this index in both output and index arrays
                for j in range(zero_index + 1, len(topk_sorted_output_global)):
                    topk_sorted_output_global[j] = 0
                    topk_sorted_idx_global[j] = 0
    
    # Write input files (input0.bin, input1.bin, etc.)
    os.makedirs(case["name"], exist_ok=True)
    for i in range(list_num):
        input_file = os.path.join(case["name"], f"input{i}.bin")
        with open(input_file, 'wb') as f:
            for val, idx in zip(output_arr_list[i], output_idx_list[i]):
                write_value_index_pair(f, val, idx, dtype)
    
    # Write golden output file
    with open(os.path.join(case["name"], "golden.bin"), 'wb') as f:
        for val, idx in zip(topk_sorted_output_global, topk_sorted_idx_global):
            write_value_index_pair(f, val, idx, dtype)
    
    print(f"[INFO] gen_data: {case['name']} list_num={list_num} "
          f"src_cols={src_cols} total_structures={total_structures} topk={topk} exhausted={exhausted}")


def gen_golden_data():
    """Generate golden data for all cases."""
    for case in CASES:
        setup_case_rng(case)

        format_type = case.get("format", "single")
        if format_type == "single":
            gen_golden_single(case)
        elif format_type == "multi":
            gen_golden_multilist(case)
        else:
            print(f"[WARN] Unsupported format: {format_type} for case {case['name']}")


if __name__ == "__main__":
    gen_golden_data()