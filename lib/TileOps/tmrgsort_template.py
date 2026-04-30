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
