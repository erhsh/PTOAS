"""TileLang DSL template for pto.tpartmin"""

import sys
from pathlib import Path
import tilelang_dsl as pto


@pto.vkernel(
    target="a5",
    op="pto.tpartmin"
)
def template_tpartmin(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    dtype = dst.element_type
    lanes = pto.get_lanes(dtype)

    dst_valid_rows, dst_valid_cols = dst.valid_shape
    src0_valid_rows, src0_valid_cols = src0.valid_shape
    src1_valid_rows, src1_valid_cols = src1.valid_shape

    src0_eq_dst = (src0_valid_rows == dst_valid_rows and src0_valid_cols == dst_valid_cols)
    src1_eq_dst = (src1_valid_rows == dst_valid_rows and src1_valid_cols == dst_valid_cols)

    if src0_eq_dst:
        if src1_eq_dst:
            for row in range(0, dst_valid_rows, 1):
                remained = dst_valid_cols
                for col in range(0, dst_valid_cols, lanes):
                    mask, remained = pto.make_mask(dtype, remained)
                    lhs = pto.vlds(src0[row, col:])
                    rhs = pto.vlds(src1[row, col:])
                    summed = pto.vmin(lhs, rhs, mask)
                    pto.vsts(summed, dst[row, col:], mask)
        else:
            src1_row_lt_dst = (src1_valid_rows < dst_valid_rows and src1_valid_cols == dst_valid_cols)
            src1_col_lt_dst = (src1_valid_rows <= dst_valid_rows and src1_valid_cols < dst_valid_cols)
            if src1_row_lt_dst:
                if src1_valid_cols > 0:
                    for row in range(0, src1_valid_rows, 1):
                        remained = src1_valid_cols
                        for col in range(0, src1_valid_cols, lanes):
                            mask, remained = pto.make_mask(dtype, remained)
                            lhs = pto.vlds(src0[row, col:])
                            rhs = pto.vlds(src1[row, col:])
                            summed = pto.vmin(lhs, rhs, mask)
                            pto.vsts(summed, dst[row, col:], mask)

                for row in range(src1_valid_rows, dst_valid_rows, 1):
                    remained = dst_valid_cols
                    for col in range(0, dst_valid_cols, lanes):
                        mask, remained = pto.make_mask(dtype, remained)
                        val = pto.vlds(src0[row, col:])
                        pto.vsts(val, dst[row, col:], mask)

            if src1_col_lt_dst:
                for row in range(0, dst_valid_rows, 1):
                    remained = dst_valid_cols
                    for col in range(0, dst_valid_cols, lanes):
                        mask, remained = pto.make_mask(dtype, remained)
                        val = pto.vlds(src0[row, col:])
                        pto.vsts(val, dst[row, col:], mask)

                if src1_valid_cols > 0:
                    for row in range(0, src1_valid_rows, 1):
                        remained = src1_valid_cols
                        for col in range(0, src1_valid_cols, lanes):
                            mask, remained = pto.make_mask(dtype, remained)
                            lhs = pto.vlds(src0[row, col:])
                            rhs = pto.vlds(src1[row, col:])
                            summed = pto.vmin(lhs, rhs, mask)
                            pto.vsts(summed, dst[row, col:], mask)
    elif src1_eq_dst:
        src0_row_lt_dst = (src0_valid_rows < dst_valid_rows and src0_valid_cols == dst_valid_cols)
        src0_col_lt_dst = (src0_valid_rows <= dst_valid_rows and src0_valid_cols < dst_valid_cols)

        if src0_row_lt_dst:
            if src0_valid_rows > 0:
                for row in range(0, src0_valid_rows, 1):
                    remained = src0_valid_cols
                    for col in range(0, src0_valid_cols, lanes):
                        mask, remained = pto.make_mask(dtype, remained)
                        lhs = pto.vlds(src0[row, col:])
                        rhs = pto.vlds(src1[row, col:])
                        summed = pto.vmin(lhs, rhs, mask)
                        pto.vsts(summed, dst[row, col:], mask)

            for row in range(src0_valid_rows, dst_valid_rows, 1):
                remained = dst_valid_cols
                for col in range(0, dst_valid_cols, lanes):
                    mask, remained = pto.make_mask(dtype, remained)
                    val = pto.vlds(src1[row, col:])
                    pto.vsts(val, dst[row, col:], mask)

        if src0_col_lt_dst:
            for row in range(0, dst_valid_rows, 1):
                remained = dst_valid_cols
                for col in range(0, dst_valid_cols, lanes):
                    mask, remained = pto.make_mask(dtype, remained)
                    val = pto.vlds(src1[row, col:])
                    pto.vsts(val, dst[row, col:], mask)

            if src0_valid_cols > 0:
                for row in range(0, src0_valid_rows, 1):
                    remained = src0_valid_cols
                    for col in range(0, src0_valid_cols, lanes):
                        mask, remained = pto.make_mask(dtype, remained)
                        lhs = pto.vlds(src0[row, col:])
                        rhs = pto.vlds(src1[row, col:])
                        summed = pto.vmin(lhs, rhs, mask)
                        pto.vsts(summed, dst[row, col:], mask)

    return