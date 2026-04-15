"""TileLang DSL template for pto.tsort32"""

import tilelang_dsl as pto

# Constants matching C++ implementation
BLOCK_SIZE = 32
REPEAT_MAX = 255
FLOAT_DST_STRIDE_COEF = 2
HALF_DST_STRIDE_COEF = 4


@pto.vkernel(
    target="a5",
    advanced=True,
    op="pto.tsort32"
)
def template_tsort32(src: pto.Tile, idx: pto.Tile, dst: pto.Tile):
    """
    TSort32: Sort fixed-size 32-element blocks using explicit input indices.

    Semantics (matching pto-isa TSort32.hpp):
    - Sorts data in 32-element blocks (one block = one vbitsort call)
    - Output format: interleaved (value, index) pairs, so dst cols = src cols * 2 for half,
      or dst cols = src cols * 4 for float (due to stride coefficients)
    """
    dtype = dst.element_type
    valid_row, _ = dst.valid_shape
    _, valid_col = src.valid_shape

    # Compute repeat times per row
    repeat_times_per_row = valid_col // BLOCK_SIZE

    # Get pointers
    dst_ptr = dst.as_ptr()
    src_ptr = src.as_ptr()
    idx_ptr = idx.as_ptr()

    # Stride coefficients based on dtype
    type_coef = HALF_DST_STRIDE_COEF
    if pto.constexpr(dtype == pto.f32):
        type_coef = FLOAT_DST_STRIDE_COEF

    # Row stride handling for idx
    idx_row_stride = idx.shape[1]
    idx_valid_shape0, _ = idx.valid_shape
    if idx_valid_shape0 == 1:
        idx_row_stride = 0 

    # Simple case: validCol is multiple of 32 and no need for large repeat handling
    if repeat_times_per_row <= REPEAT_MAX:
        for row in range(0, valid_row, 1):
            row_dst_offset = row * dst.shape[1]
            row_src_offset = row * src.shape[1]
            row_idx_offset = 0
            if idx_row_stride > 0:
                row_idx_offset = row * idx_row_stride

            dst_row_ptr = pto.addptr(dst_ptr, row_dst_offset)
            src_row_ptr = pto.addptr(src_ptr, row_src_offset)
            idx_row_ptr = pto.addptr(idx_ptr, row_idx_offset)

            pto.vbitsort(dst_row_ptr, src_row_ptr, idx_row_ptr, repeat_times_per_row)
    else:
        # Handle large repeat times: split into multiple vbitsort calls
        loop_num = (repeat_times_per_row + REPEAT_MAX - 1) // REPEAT_MAX
        tail_repeat_num = repeat_times_per_row % REPEAT_MAX

        for row in range(0, valid_row, 1):
            row_dst_offset = row * dst.shape[1]
            row_src_offset = row * src.shape[1]
            row_idx_offset = 0
            if idx_row_stride > 0:
                row_idx_offset = row * idx_row_stride

            for j in range(0, loop_num, 1):
                repeat_num = tail_repeat_num
                if j < loop_num - 1:
                    repeat_num = REPEAT_MAX
                if repeat_num == 0:
                    repeat_num = REPEAT_MAX

                block_offset = j * REPEAT_MAX * BLOCK_SIZE

                dst_block_ptr = pto.addptr(dst_ptr, row_dst_offset + block_offset * type_coef)
                src_block_ptr = pto.addptr(src_ptr, row_src_offset + block_offset)
                idx_block_ptr = pto.addptr(idx_ptr, row_idx_offset + block_offset)

                pto.vbitsort(dst_block_ptr, src_block_ptr, idx_block_ptr, repeat_num)

    return