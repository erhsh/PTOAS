"""TileLang DSL template for pto.tpartadd"""

import sys
from pathlib import Path
import tilelang_dsl as pto


@pto.vkernel(
    target="a5",
    op="pto.tpartadd"
)
def template_tpartadd(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile):
    """Partial elementwise add with implementation-defined handling of mismatched valid regions.

    For each element (i, j) in the destination valid region:
    - If both inputs are defined at (i,j): dst[i,j] = src0[i,j] + src1[i,j]
    - If only src0 is defined at (i,j): dst[i,j] = src0[i,j]
    - If only src1 is defined at (i,j): dst[i,j] = src1[i,j]

    Note: This template handles the case where at least one source matches dst valid region.
    The implementation follows the semantics defined in TPartAdd.hpp.
    """
    dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(dtype)):
            mask, remained = pto.make_mask(dtype, remained)
            lhs = pto.vlds(src0[row, col:])
            rhs = pto.vlds(src1[row, col:])
            summed = pto.vadd(lhs, rhs, mask)
            pto.vsts(summed, dst[row, col:], mask)
    return