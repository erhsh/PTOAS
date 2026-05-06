# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.tmrgsort"""

import tilelang_dsl as pto

STRUCT_SIZE = 8  # bytes per structure (value + index)
STRUCT_SIZE_SHIFT = 3  # log2(8)
BLOCK_NUM = 4


@pto.inline_proc
def tmrgsort_1list_instr(dst: pto.Tile, src: pto.Tile,
                         num_structures, repeat_times):
    dtype = dst.element_type
    bw = pto.bytewidth(dtype)  # index type

    offset = num_structures * STRUCT_SIZE // bw

    count = pto.i64(num_structures)
    count = count | (pto.i64(num_structures) << pto.i64(16));
    count = count | (pto.i64(num_structures) << pto.i64(32));
    count = count | (pto.i64(num_structures) << pto.i64(48));

    config = pto.i64(repeat_times)
    config = config | (pto.i64(0b1111) << pto.i64(8))
    config = config | (pto.i64(0b0) << pto.i64(12))

    # Get pointers from tiles
    dst_ptr = dst.as_ptr()
    src_ptr = src.as_ptr()

    # Compute offset pointers for the 4 source blocks (offset is index type)
    src0 = src_ptr
    src1 = pto.addptr(src_ptr, offset)
    src2 = pto.addptr(src_ptr, offset * 2)
    src3 = pto.addptr(src_ptr, offset * 3)

    # Execute vmrgsort4 with pointers
    # pto.vmrgsort4(dst_ptr, src0, src1, src2, src3, count, config)
    pto.vmrgsort4(dst_ptr, src0, src1, src2, src3, pto.i64(count), pto.i64(config))
    return


@pto.inline_proc
def tmrgsort_2list_instr_normal(tmp: pto.Tile, src0: pto.Tile, src1: pto.Tile,
                                  src0_structures: int, src1_structures: int):
    dtype = tmp.element_type
    
    tmp_ptr = tmp.as_ptr()
    src0_ptr = src0.as_ptr()
    src1_ptr = src1.as_ptr()
    
    count = pto.i64(src0_structures)
    count = count | (pto.i64(src1_structures) << pto.i64(16))
    
    repeat_time = 1
    list_mask = 0b0011
    exhausted_bit = 0
    
    config = pto.i64(repeat_time)
    config = config | (pto.i64(list_mask) << pto.i64(8))
    config = config | (pto.i64(exhausted_bit) << pto.i64(12))
    
    pto.vmrgsort4(tmp_ptr, src0_ptr, src1_ptr, src0_ptr, src0_ptr,
                   count, config)
    
    return


@pto.inline_proc
def tmrgsort_2list_instr_exhausted(tmp: pto.Tile, src0: pto.Tile, src1: pto.Tile,
                                    src0_structures: int, src1_structures: int):
    dtype = tmp.element_type
    
    tmp_ptr = tmp.as_ptr()
    src0_ptr = src0.as_ptr()
    src1_ptr = src1.as_ptr()
    
    count = pto.i64(src0_structures)
    count = count | (pto.i64(src1_structures) << pto.i64(16))
    
    repeat_time = 1
    list_mask = 0b0011
    exhausted_bit = 1
    
    config = pto.i64(repeat_time)
    config = config | (pto.i64(list_mask) << pto.i64(8))
    config = config | (pto.i64(exhausted_bit) << pto.i64(12))
    
    pto.vmrgsort4(tmp_ptr, src0_ptr, src1_ptr, src0_ptr, src0_ptr,
                   count, config)
    
    return


@pto.vkernel(
    target="a5",
    op="pto.tmrgsort",
    advanced=True,
)
def template_tmrgsort_1list(src: pto.Tile, block_len: pto.AnyInt, dst: pto.Tile):
    dtype = src.element_type
    src_valid_col = src.valid_shape[1]

    # num_structures = block_len
    num_structures = block_len * pto.bytewidth(dtype) // STRUCT_SIZE
    repeat_times = src_valid_col // (block_len * BLOCK_NUM)
    tmrgsort_1list_instr(dst, src, num_structures, repeat_times)

    return None


@pto.vkernel(
    target="a5",
    op="pto.tmrgsort",
    advanced=True,
)
def template_tmrgsort_2list(src0: pto.Tile, src1: pto.Tile, 
                            tmp: pto.Tile, dst: pto.Tile, ex_vec: pto.AnyInt):
    dtype = dst.element_type
    bw = pto.bytewidth(dtype)
    
    src0_valid_col = src0.valid_shape[1]
    src1_valid_col = src1.valid_shape[1]
    dst_valid_col = dst.valid_shape[1]
    
    if pto.constexpr(bw == 4):
        src0_structures = src0_valid_col // 2
        src1_structures = src1_valid_col // 2
    else:
        src0_structures = src0_valid_col // 4
        src1_structures = src1_valid_col // 4
    
    dst_elements = dst_valid_col
    
    exhausted_str = pto.get_op_attr("exhausted", "0")
    
    if pto.constexpr(exhausted_str == "1"):
        tmrgsort_2list_instr_exhausted(tmp, src0, src1,
                                        src0_structures, src1_structures)
    else:
        tmrgsort_2list_instr_normal(tmp, src0, src1,
                                     src0_structures, src1_structures)
    
    lanes = pto.get_lanes(dtype)
    for col in range(0, dst_elements, lanes):
        remained = dst_elements - col
        mask, remained = pto.make_mask(dtype, remained)
        data = pto.vlds(tmp[0, col:])
        pto.vsts(data, dst[0, col:], mask)
    
    return None