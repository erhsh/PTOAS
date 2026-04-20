"""TileLang DSL template for pto.tpartmin"""

import tilelang_dsl as pto


@pto.vkernel(
    target="a5",
    op="pto.tpartmin",
    advanced=True,
)
def template_tpartmin(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape
    src0_valid_rows, src0_valid_cols = src0.valid_shape
    src1_valid_rows, src1_valid_cols = src1.valid_shape
    lanes = pto.get_lanes(dtype)

    pad_scalar = pto.f32(3.402823466e+38)
    if pto.constexpr(dtype == pto.f16):
        pad_scalar = pto.f16(65504.0)
    elif pto.constexpr(dtype == pto.bf16):
        pad_scalar = pto.bf16(3.3856e+38)
    elif pto.constexpr(dtype == pto.i32):
        pad_scalar = pto.i32(2147483647)
    elif pto.constexpr(dtype == pto.ui32):
        # pad_scalar = pto.ui32(4294967295)
        pass
    elif pto.constexpr(dtype == pto.i16):
        pad_scalar = pto.i16(32767)
    elif pto.constexpr(dtype == pto.ui16):
        # pad_scalar = pto.ui16(65535)
        pass
    elif pto.constexpr(dtype == pto.i8):
        pad_scalar = pto.i8(127)
    elif pto.constexpr(dtype == pto.ui8):
        # pad_scalar = pto.ui8(255)
        pass

    pad_vec = pto.vbr(pad_scalar)

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            # pto.vsts(pad_vec, dst[row, col:], mask)
            pass

    for row in range(0, src0_valid_rows, 1):
        remained = src0_valid_cols
        for col in range(0, src0_valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            vec0 = pto.vlds(src0[row, col:])
            pto.vsts(vec0, dst[row, col:], mask)

    for row in range(0, src1_valid_rows, 1):
        remained = src1_valid_cols
        for col in range(0, src1_valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            vec_dst = pto.vlds(dst[row, col:])
            vec1 = pto.vlds(src1[row, col:])
            result = pto.vmin(vec_dst, vec1, mask)
            pto.vsts(result, dst[row, col:], mask)

    return