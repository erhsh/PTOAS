# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You can not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.tmrgsort"""

import tilelang_dsl as pto

# Constants matching C++ implementation
STRUCT_SIZE = 8
STRUCT_SIZE_SHIFT = 3
BLOCK_NUM = 4


@pto.vkernel(
    target="a5",
    op="pto.tmrgsort"
)
def template_tmrgsort_format1(src: pto.Tile, dst: pto.Tile, block_len: pto.i32):
    """
    TMrgSort Format1: Merge-sort a single source tile in-place.

    Semantics (matching pto-isa TMrgSort.hpp):
    - The source tile is treated as 4 contiguous blocks, each of size blockLen
    - blockLen includes values and indices (e.g., 32 values + indices = blockLen=64 for half)
    - Uses vmrgsort4 to merge these 4 pre-sorted blocks
    - Only supports rows=1
    """
    dtype = dst.element_type
    src_col = src.valid_shape[1]

    # Get pointers
    dst_ptr = dst.as_ptr()
    src_ptr = src.as_ptr()

    # Calculate number of structures and repeat times
    # STRUCT_SIZE = 8 bytes (value + index pair)
    # For half: sizeof(half) = 2, so blockLen * 2 / 8 = blockLen / 4 structures
    # For float: sizeof(float) = 4, so blockLen * 4 / 8 = blockLen / 2 structures
    elem_size_shift = 1 if dtype == pto.f32 else 2
    num_structures = block_len >> elem_size_shift

    # Offset between blocks in elements
    offset = num_structures * STRUCT_SIZE >> elem_size_shift

    # Calculate repeat times based on source size
    repeat_times = src_col // (block_len * BLOCK_NUM)

    # Construct the 4 source block pointers
    src0_ptr = src_ptr
    src1_ptr = pto.addptr(src_ptr, offset)
    src2_ptr = pto.addptr(src_ptr, offset * 2)
    src3_ptr = pto.addptr(src_ptr, offset * 3)

    # Build count value (each block has same number of structures)
    count = num_structures
    count |= (num_structures << 16)
    count |= (num_structures << 32)
    count |= (num_structures << 48)

    # Build config value
    # config bits: [7:0] = repeat_times, [11:8] = mask (0b1111 for 4 lists), [12] = exhausted (0)
    config = repeat_times
    config |= (0b1111 << 8)
    config |= (0 << 12)

    pto.vmrgsort4(dst_ptr, src0_ptr, src1_ptr, src2_ptr, src3_ptr, count, config)

    return


@pto.vkernel(
    target="a5",
    op="pto.tmrgsort"
)
def template_tmrgsort_format2(
    src0: pto.Tile,
    src1: pto.Tile,
    src2: pto.Tile,
    src3: pto.Tile,
    dst: pto.Tile,
    tmp: pto.Tile,
    exhausted: bool = False
):
    """
    TMrgSort Format2: Merge-sort 4 pre-sorted source tiles.

    Semantics (matching pto-isa TMrgSort.hpp):
    - Merges 4 pre-sorted input lists (src0..src3)
    - Uses tmp as intermediate buffer, then copies to dst
    - Each source tile has rows=1
    - exhausted mode controls whether to stop when input lists are exhausted
    - Returns executed count for each list via MrgSortExecutedNumList
    """
    dtype = dst.element_type

    # Get pointers
    dst_ptr = dst.as_ptr()
    tmp_ptr = tmp.as_ptr()
    src0_ptr = src0.as_ptr()
    src1_ptr = src1.as_ptr()
    src2_ptr = src2.as_ptr()
    src3_ptr = src3.as_ptr()

    # Get valid columns for each source
    # For half: STRUCT_SIZE / sizeof(half) = 8 / 2 = 4 elements per structure
    # For float: STRUCT_SIZE / sizeof(float) = 8 / 4 = 2 elements per structure
    elem_num_shift = 1 if dtype == pto.f32 else 2

    src0_col = src0.valid_shape[1] >> elem_num_shift
    src1_col = src1.valid_shape[1] >> elem_num_shift
    src2_col = src2.valid_shape[1] >> elem_num_shift
    src3_col = src3.valid_shape[1] >> elem_num_shift

    # Build count value
    count = src0_col
    count |= (src1_col << 16)
    count |= (src2_col << 32)
    count |= (src3_col << 48)

    # Build config value
    # config bits: [7:0] = repeat_times (1), [11:8] = mask (0b1111), [12] = exhausted flag
    config = 1  # repeat_times = 1
    config |= (0b1111 << 8)  # mask for 4 lists
    config |= ((1 if exhausted else 0) << 12)  # exhausted flag

    # Perform merge sort with tmp as intermediate
    pto.vmrgsort4(tmp_ptr, src0_ptr, src1_ptr, src2_ptr, src3_ptr, count, config)

    # Copy from tmp to dst
    dst_col = dst.valid_shape[1]
    elem_bytes = pto.bytewidth(dtype)
    len_burst = (dst_col * elem_bytes + 32 - 1) // 32
    pto.copy_ubuf_to_ubuf(dst_ptr, tmp_ptr, 1, len_burst, 0, 0)

    return