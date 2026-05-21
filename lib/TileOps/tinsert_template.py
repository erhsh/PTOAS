# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.tinsert - insert src sub-tile into dst at offset

"""

import tilelang_dsl as pto

_BLOCK_BYTE = 32
_BLOCK_LEN = 256

_PARAM_INDEX = {"src": 0, "indexRow": 1, "indexCol": 2, "dst": 3}

_STR_TO_BYTE_WIDTH = {
    "f32": 4, "f16": 2, "bf16": 2, "i32": 4, "i16": 2, "i8": 1, "si32": 4,
}


def _get_elem_bytes(tile, **kwargs) -> int:
    dtype = tile.dtype
    if dtype is None:
        tile_name = tile._name
        positional_key = f"arg{_PARAM_INDEX.get(tile_name, 0)}_dtype"
        named_key = f"{tile_name}_dtype"
        dtype = kwargs.get(named_key) or kwargs.get(positional_key)
    if dtype is None:
        return 0
    return pto.bytewidth(dtype) if not isinstance(dtype, str) else _STR_TO_BYTE_WIDTH.get(dtype, 0)


def _get_dtype_raw(tile, **kwargs):
    dtype = tile.dtype
    if dtype is not None:
        return dtype
    tile_name = tile._name
    positional_key = f"arg{_PARAM_INDEX.get(tile_name, 0)}_dtype"
    named_key = f"{tile_name}_dtype"
    return kwargs.get(named_key) or kwargs.get(positional_key)


def _ms_value(ms):
    if ms is None:
        return None
    if isinstance(ms, str):
        return ms
    if hasattr(ms, "value"):
        return ms.value
    return str(ms).lower()


def _is_ub(tile) -> bool:
    return _ms_value(tile.memory_space) == "ub"


def _is_mat(tile) -> bool:
    return _ms_value(tile.memory_space) == "mat"


def _is_acc(tile) -> bool:
    return _ms_value(tile.memory_space) == "acc"


def _is_nd_layout(tile) -> bool:
    c = tile.config
    if c is None:
        return False
    return c.b_layout == pto.BLayout.ROW_MAJOR and c.s_layout == pto.SLayout.NONE_BOX


def _is_nz_layout(tile) -> bool:
    c = tile.config
    if c is None:
        return False
    return c.b_layout == pto.BLayout.COL_MAJOR and c.s_layout == pto.SLayout.ROW_MAJOR


def _is_dn_layout(tile) -> bool:
    c = tile.config
    if c is None:
        return False
    return c.b_layout == pto.BLayout.COL_MAJOR and c.s_layout == pto.SLayout.NONE_BOX


def _stride_32b_aligned(tile, **kwargs) -> bool:
    elem_bytes = _get_elem_bytes(tile, **kwargs)
    if elem_bytes == 0:
        return True
    stride = tile.shape[1]
    if hasattr(stride, "value"):
        stride = stride.value
    if stride is None:
        return True
    return stride * elem_bytes % _BLOCK_BYTE == 0


def _c0_size(elem_bytes) -> int:
    return _BLOCK_BYTE // elem_bytes


def _index_col_32b_aligned(indexCol, elem_bytes) -> bool:
    if elem_bytes == 0:
        return True
    ic = indexCol
    if hasattr(ic, "value"):
        ic = ic.value
    if ic is None:
        return True
    return ic * elem_bytes % _BLOCK_BYTE == 0


def _same_dtype(src, dst, **kwargs) -> bool:
    src_dtype = _get_dtype_raw(src, **kwargs)
    dst_dtype = _get_dtype_raw(dst, **kwargs)
    if src_dtype is None or dst_dtype is None:
        return True
    return src_dtype == dst_dtype


def _common_constraint(src, indexRow, indexCol, dst, **kwargs) -> bool:
    if not (_is_ub(src) and _is_ub(dst)):
        return False
    sc = src.config
    dc = dst.config
    if sc is None or dc is None:
        return False
    if sc.s_layout != pto.SLayout.NONE_BOX or dc.s_layout != pto.SLayout.NONE_BOX:
        return False
    return _same_dtype(src, dst, **kwargs)


def _scalar_constraint(src, indexRow, indexCol, dst, **kwargs) -> bool:
    if not _common_constraint(src, indexRow, indexCol, dst, **kwargs):
        return False
    vrows, vcols = src.valid_shape
    return vrows == 1 and vcols == 1


def _valid_col_aligned(src, **kwargs) -> bool:
    elem_bytes = _get_elem_bytes(src, **kwargs)
    if elem_bytes == 0:
        return True
    vcols = src.valid_shape[1]
    if hasattr(vcols, "value"):
        vcols = vcols.value
    if vcols is None:
        return True
    return vcols * elem_bytes % _BLOCK_BYTE == 0


def _nd_aligned_constraint(src, indexRow, indexCol, dst, **kwargs) -> bool:
    if not _common_constraint(src, indexRow, indexCol, dst, **kwargs):
        return False
    elem_bytes = _get_elem_bytes(src, **kwargs)
    if not _index_col_32b_aligned(indexCol, elem_bytes):
        return False
    if not _stride_32b_aligned(src, **kwargs):
        return False
    if not _stride_32b_aligned(dst, **kwargs):
        return False
    return True


def _dma_constraint(src, indexRow, indexCol, dst, **kwargs) -> bool:
    if not _nd_aligned_constraint(src, indexRow, indexCol, dst, **kwargs):
        return False
    if not _valid_col_aligned(src, **kwargs):
        return False
    if _scalar_constraint(src, indexRow, indexCol, dst, **kwargs):
        return False
    return True


def _aligned_constraint(src, indexRow, indexCol, dst, **kwargs) -> bool:
    if not _nd_aligned_constraint(src, indexRow, indexCol, dst, **kwargs):
        return False
    if _valid_col_aligned(src, **kwargs):
        return False
    if _scalar_constraint(src, indexRow, indexCol, dst, **kwargs):
        return False
    return True


def _unaligned_constraint(src, indexRow, indexCol, dst, **kwargs) -> bool:
    if not _common_constraint(src, indexRow, indexCol, dst, **kwargs):
        return False
    if _scalar_constraint(src, indexRow, indexCol, dst, **kwargs):
        return False
    elem_bytes = _get_elem_bytes(src, **kwargs)
    src_stride_aligned = _stride_32b_aligned(src, **kwargs)
    dst_stride_aligned = _stride_32b_aligned(dst, **kwargs)
    index_col_aligned = _index_col_32b_aligned(indexCol, elem_bytes)
    if src_stride_aligned and dst_stride_aligned and index_col_aligned:
        return False
    return True


def _vec_vec_nz_dma_constraint(src, indexRow, indexCol, dst, **kwargs) -> bool:
    if not (_is_ub(src) and _is_ub(dst)):
        return False
    if not (_is_nz_layout(src) and _is_nz_layout(dst)):
        return False
    elem_bytes = _get_elem_bytes(src, **kwargs)
    if not _index_col_32b_aligned(indexCol, elem_bytes):
        return False
    return _same_dtype(src, dst, **kwargs)


def _vec_mat_nd_dma_constraint(src, indexRow, indexCol, dst, **kwargs) -> bool:
    if not (_is_ub(src) and _is_mat(dst)):
        return False
    if not (_is_nd_layout(src) and _is_nd_layout(dst)):
        return False
    elem_bytes = _get_elem_bytes(src, **kwargs)
    if not _index_col_32b_aligned(indexCol, elem_bytes):
        return False
    return _same_dtype(src, dst, **kwargs)


def _vec_mat_nz_dma_constraint(src, indexRow, indexCol, dst, **kwargs) -> bool:
    if not (_is_ub(src) and _is_mat(dst)):
        return False
    if not (_is_nz_layout(src) and _is_nz_layout(dst)):
        return False
    elem_bytes = _get_elem_bytes(src, **kwargs)
    if not _index_col_32b_aligned(indexCol, elem_bytes):
        return False
    return _same_dtype(src, dst, **kwargs)


def _acc_to_vec_nd_constraint(src, indexRow, indexCol, dst, **kwargs) -> bool:
    if not (_is_acc(src) and _is_ub(dst)):
        return False
    if not _is_nd_layout(dst):
        return False
    return _same_dtype(src, dst, **kwargs)


def _acc_to_vec_dn_constraint(src, indexRow, indexCol, dst, **kwargs) -> bool:
    if not (_is_acc(src) and _is_ub(dst)):
        return False
    if not _is_dn_layout(dst):
        return False
    return _same_dtype(src, dst, **kwargs)


def _acc_to_vec_nz_constraint(src, indexRow, indexCol, dst, **kwargs) -> bool:
    if not (_is_acc(src) and _is_ub(dst)):
        return False
    if not _is_nz_layout(dst):
        return False
    return _same_dtype(src, dst, **kwargs)


def _acc_to_mat_nz_constraint(src, indexRow, indexCol, dst, **kwargs) -> bool:
    if not (_is_acc(src) and _is_mat(dst)):
        return False
    if not _is_nz_layout(dst):
        return False
    if not _same_dtype(src, dst, **kwargs):
        return False
    if kwargs.get("pre_relu") is not None:
        return False
    return True


def _acc_to_mat_nz_relu_constraint(src, indexRow, indexCol, dst, **kwargs) -> bool:
    if not (_is_acc(src) and _is_mat(dst)):
        return False
    if not _is_nz_layout(dst):
        return False
    if not _same_dtype(src, dst, **kwargs):
        return False
    if kwargs.get("pre_relu") != "normal_relu":
        return False
    return True


_VEC_DTYPE_TUPLES = [
    (pto.f32, pto.i32, pto.i32, pto.f32),
    (pto.f16, pto.i32, pto.i32, pto.f16),
    (pto.bf16, pto.i32, pto.i32, pto.bf16),
    (pto.i32, pto.i32, pto.i32, pto.i32),
    (pto.i8, pto.i32, pto.i32, pto.i8),
]


@pto.vkernel(
    target="a5",
    op="pto.tinsert",
    dtypes=_VEC_DTYPE_TUPLES,
    constraints=[_scalar_constraint],
    advanced=True,
)
def template_tinsert_scalar(src: pto.Tile, indexRow: pto.i32, indexCol: pto.i32, dst: pto.Tile):
    dst_stride = dst.shape[1]
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()
    src_scalar = pto.load_scalar(src_ptr, 0)
    dst_elem_offset = indexRow * dst_stride + indexCol
    pto.store_scalar(src_scalar, dst_ptr, dst_elem_offset)
    return None


@pto.vkernel(
    target="a5",
    op="pto.tinsert",
    dtypes=_VEC_DTYPE_TUPLES,
    constraints=[_dma_constraint],
    advanced=True,
)
def template_tinsert_dma(src: pto.Tile, indexRow: pto.i32, indexCol: pto.i32, dst: pto.Tile):
    dtype = dst.element_type
    elem_bytes = pto.bytewidth(dtype)

    valid_rows, valid_cols = src.valid_shape
    src_stride = src.shape[1]
    dst_stride = dst.shape[1]

    row_bytes = valid_cols * elem_bytes
    total_bytes = valid_rows * row_bytes
    row_burst_len = row_bytes // _BLOCK_BYTE

    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()

    dst_offset_bytes = (indexRow * dst_stride + indexCol) * elem_bytes
    dst_start = pto.addptr(dst_ptr, dst_offset_bytes // elem_bytes)

    if src_stride == valid_cols and dst_stride == valid_cols and total_bytes >= _BLOCK_BYTE:
        burst_len = total_bytes // _BLOCK_BYTE
        pto.copy_ubuf_to_ubuf(src_ptr, dst_start, 0, 1, burst_len, 0, 0)
    else:
        src_gap = (src_stride - valid_cols) * elem_bytes // _BLOCK_BYTE
        dst_gap = (dst_stride - valid_cols) * elem_bytes // _BLOCK_BYTE
        pto.copy_ubuf_to_ubuf(src_ptr, dst_start, 0, valid_rows, row_burst_len, src_gap, dst_gap)
    return None


@pto.vkernel(
    target="a5",
    op="pto.tinsert",
    dtypes=_VEC_DTYPE_TUPLES,
    constraints=[_aligned_constraint],
    advanced=True,
)
def template_tinsert_aligned_vector(src: pto.Tile, indexRow: pto.i32, indexCol: pto.i32, dst: pto.Tile):
    dtype = dst.element_type
    lanes = pto.get_lanes(dtype)

    valid_rows, valid_cols = src.valid_shape
    src_stride = src.shape[1]
    dst_stride = dst.shape[1]

    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()

    repeat_times = (valid_cols + lanes - 1) // lanes

    for i in range(0, valid_rows, 1):
        remained = valid_cols
        for j in range(0, repeat_times, 1):
            pred, remained = pto.make_mask(dtype, remained)
            src_elem_off = i * src_stride + j * lanes
            dst_elem_off = (indexRow + i) * dst_stride + indexCol + j * lanes
            data = pto.vlds(pto.addptr(src_ptr, src_elem_off), 0)
            pto.vsts(data, pto.addptr(dst_ptr, dst_elem_off), 0, pred)
    return None


@pto.vkernel(
    target="a5",
    op="pto.tinsert",
    dtypes=_VEC_DTYPE_TUPLES,
    constraints=[_unaligned_constraint],
    advanced=True,
)
def template_tinsert_unaligned_vector(src: pto.Tile, indexRow: pto.i32, indexCol: pto.i32, dst: pto.Tile):
    dtype = dst.element_type
    lanes = pto.get_lanes(dtype)
    elem_bytes = pto.bytewidth(dtype)

    valid_rows, valid_cols = src.valid_shape
    src_stride = src.shape[1]
    dst_stride = dst.shape[1]

    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()

    full_repeats = valid_cols // lanes
    remainder = valid_cols % lanes

    for i in range(0, valid_rows, 1):
        ureg = pto.init_align(dtype)
        src_row_off = i * src_stride
        dst_row_ptr = pto.addptr(dst_ptr, (indexRow + i) * dst_stride + indexCol)

        for j in range(0, full_repeats, 1):
            data = pto.vlds(pto.addptr(src_ptr, src_row_off + j * lanes), 0)
            ureg = pto.vstus(ureg, lanes, data, dst_row_ptr)
            dst_row_ptr = pto.addptr(dst_row_ptr, lanes)

        if remainder > 0:
            data = pto.vlds(pto.addptr(src_ptr, src_row_off + full_repeats * lanes), 0)
            ureg = pto.vstus(ureg, remainder, data, dst_row_ptr)

        pto.vstas(ureg, dst_row_ptr, 0)
    return None


@pto.vkernel(
    target="a5",
    op="pto.tinsert",
    dtypes=_VEC_DTYPE_TUPLES,
    constraints=[_vec_vec_nz_dma_constraint],
    advanced=True,
)
def template_tinsert_vec_vec_nz_dma(src: pto.Tile, indexRow: pto.i32, indexCol: pto.i32, dst: pto.Tile):
    dtype = dst.element_type
    elem_bytes = pto.bytewidth(dtype)

    valid_rows, valid_cols = src.valid_shape
    src_rows = src.shape[0]
    dst_rows = dst.shape[0]

    c0_size = 32 // elem_bytes
    burst_num = (valid_cols + c0_size - 1) // c0_size
    burst_len = valid_rows
    src_gap = src_rows - valid_rows
    dst_gap = dst_rows - valid_rows

    dst_offset = dst_rows * indexCol + indexRow * c0_size

    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()
    dst_start = pto.addptr(dst_ptr, dst_offset)

    pto.copy_ubuf_to_ubuf(src_ptr, dst_start, 0, burst_num, burst_len, src_gap, dst_gap)
    return None


@pto.vkernel(
    target="a5",
    op="pto.tinsert",
    dtypes=_VEC_DTYPE_TUPLES,
    constraints=[_vec_mat_nd_dma_constraint],
    advanced=True,
)
def template_tinsert_vec_mat_nd_dma(src: pto.Tile, indexRow: pto.i32, indexCol: pto.i32, dst: pto.Tile):
    dtype = dst.element_type
    elem_bytes = pto.bytewidth(dtype)

    valid_rows, valid_cols = src.valid_shape
    src_stride = src.shape[1]
    dst_stride = dst.shape[1]

    row_bytes = valid_cols * elem_bytes
    total_bytes = valid_rows * row_bytes

    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()

    dst_offset_bytes = (indexRow * dst_stride + indexCol) * elem_bytes
    dst_start = pto.addptr(dst_ptr, dst_offset_bytes // elem_bytes)

    if src_stride == valid_cols and dst_stride == valid_cols and total_bytes >= _BLOCK_BYTE:
        burst_len = total_bytes // _BLOCK_BYTE
        pto.mte_ub_l1(src_ptr, dst_start, burst_len, nburst=(1, 0, 0))
    else:
        row_burst_len = row_bytes // _BLOCK_BYTE
        src_gap = (src_stride - valid_cols) * elem_bytes // _BLOCK_BYTE
        dst_gap = (dst_stride - valid_cols) * elem_bytes // _BLOCK_BYTE
        pto.mte_ub_l1(src_ptr, dst_start, row_burst_len, nburst=(valid_rows, src_gap, dst_gap))
    return None


@pto.vkernel(
    target="a5",
    op="pto.tinsert",
    dtypes=_VEC_DTYPE_TUPLES,
    constraints=[_vec_mat_nz_dma_constraint],
    advanced=True,
)
def template_tinsert_vec_mat_nz_dma(src: pto.Tile, indexRow: pto.i32, indexCol: pto.i32, dst: pto.Tile):
    dtype = dst.element_type
    elem_bytes = pto.bytewidth(dtype)

    valid_rows, valid_cols = src.valid_shape
    src_rows = src.shape[0]
    dst_rows = dst.shape[0]

    c0_size = 32 // elem_bytes
    burst_num = (valid_cols + c0_size - 1) // c0_size
    burst_len = valid_rows
    src_gap = src_rows - valid_rows
    dst_gap = dst_rows - valid_rows

    dst_offset = dst_rows * indexCol + indexRow * c0_size

    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()
    dst_start = pto.addptr(dst_ptr, dst_offset)

    pto.mte_ub_l1(src_ptr, dst_start, burst_len, nburst=(burst_num, src_gap, dst_gap))
    return None


@pto.ckernel(
    target="a5",
    op="pto.tinsert",
    dtypes=[
        (pto.f32, pto.i32, pto.i32, pto.f32),
    ],
    constraints=[_acc_to_vec_nd_constraint],
)
def template_tinsert_acc_to_vec_nd(src: pto.Tile, indexRow: pto.i32, indexCol: pto.i32, dst: pto.Tile):
    dtype = dst.element_type
    elem_bytes = pto.bytewidth(dtype)

    valid_rows, valid_cols = src.valid_shape
    dst_cols = dst.shape[1]

    c0_size = 32 // elem_bytes
    valid_col_aligned = ((valid_cols + c0_size - 1) // c0_size) * c0_size

    src_stride_c0 = ((valid_rows + 256 - 1) // 256) * 256 // c0_size

    dst_stride = dst_cols

    dst_offset = indexRow * dst_cols + indexCol

    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()
    dst_start = pto.addptr(dst_ptr, dst_offset)

    pto.mte_l0c_ub(src_ptr, dst_start, valid_rows, valid_col_aligned, src_stride_c0, dst_stride, 0, layout="nz2nd")
    return None


@pto.ckernel(
    target="a5",
    op="pto.tinsert",
    dtypes=[
        (pto.f32, pto.i32, pto.i32, pto.f32),
    ],
    constraints=[_acc_to_mat_nz_constraint],
)
def template_tinsert_acc_to_mat_nz(src: pto.Tile, indexRow: pto.i32, indexCol: pto.i32, dst: pto.Tile):
    dtype = dst.element_type
    elem_bytes = pto.bytewidth(dtype)

    valid_rows, valid_cols = src.valid_shape
    src_rows = src.shape[0]
    dst_rows = dst.shape[0]

    c0_size = 32 // elem_bytes
    n_size = ((valid_cols + c0_size - 1) // c0_size) * c0_size

    src_stride_c0 = src_rows // c0_size
    dst_stride = dst_rows * c0_size

    dst_offset = dst_rows * indexCol + indexRow * c0_size

    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()
    dst_start = pto.addptr(dst_ptr, dst_offset)

    pto.mte_l0c_l1(src_ptr, dst_start, src_rows, n_size, src_stride_c0, dst_stride, layout=("nz2nz", 0))
    return None


@pto.ckernel(
    target="a5",
    op="pto.tinsert",
    dtypes=[
        (pto.f32, pto.i32, pto.i32, pto.f32),
    ],
    constraints=[_acc_to_vec_dn_constraint],
)
def template_tinsert_acc_to_vec_dn(src: pto.Tile, indexRow: pto.i32, indexCol: pto.i32, dst: pto.Tile):
    dtype = dst.element_type
    elem_bytes = pto.bytewidth(dtype)

    valid_rows, valid_cols = src.valid_shape
    dst_rows = dst.shape[0]

    c0_size = 32 // elem_bytes
    valid_row_aligned = ((valid_rows + c0_size - 1) // c0_size) * c0_size
    valid_col_aligned = ((valid_cols + c0_size - 1) // c0_size) * c0_size

    src_stride = ((valid_rows + _BLOCK_LEN - 1) // _BLOCK_LEN) * _BLOCK_LEN

    dst_stride = dst_rows

    dst_offset = indexCol * dst_rows + indexRow

    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()
    dst_start = pto.addptr(dst_ptr, dst_offset)

    loop0_src_stride = src_stride // c0_size

    pto.mte_l0c_ub(src_ptr, dst_start, valid_row_aligned, valid_col_aligned, src_stride // c0_size, dst_stride, 0,
                   layout=("nz2dn", loop0_src_stride))
    return None


@pto.ckernel(
    target="a5",
    op="pto.tinsert",
    dtypes=[
        (pto.f32, pto.i32, pto.i32, pto.f32),
    ],
    constraints=[_acc_to_vec_nz_constraint],
)
def template_tinsert_acc_to_vec_nz(src: pto.Tile, indexRow: pto.i32, indexCol: pto.i32, dst: pto.Tile):
    dtype = dst.element_type
    elem_bytes = pto.bytewidth(dtype)

    valid_rows, valid_cols = src.valid_shape
    src_rows = src.shape[0]
    dst_rows = dst.shape[0]

    c0_size = 32 // elem_bytes
    valid_col_aligned = ((valid_cols + c0_size - 1) // c0_size) * c0_size
    n_size = valid_col_aligned

    src_stride_c0 = src_rows // c0_size
    dst_stride = dst_rows * c0_size

    dst_offset = dst_rows * indexCol + indexRow * c0_size

    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()
    dst_start = pto.addptr(dst_ptr, dst_offset)

    loop0_src_stride = src_stride_c0

    pto.mte_l0c_ub(src_ptr, dst_start, src_rows, n_size, src_stride_c0, dst_stride, 0,
                   layout=("nz2nz", loop0_src_stride))
    return None


@pto.ckernel(
    target="a5",
    op="pto.tinsert",
    dtypes=[
        (pto.f32, pto.i32, pto.i32, pto.f32),
    ],
    constraints=[_acc_to_vec_nd_constraint],
)
def template_tinsert_acc_to_vec_nd_relu(src: pto.Tile, indexRow: pto.i32, indexCol: pto.i32, dst: pto.Tile):
    dtype = dst.element_type
    elem_bytes = pto.bytewidth(dtype)

    valid_rows, valid_cols = src.valid_shape
    dst_cols = dst.shape[1]

    c0_size = 32 // elem_bytes
    valid_col_aligned = ((valid_cols + c0_size - 1) // c0_size) * c0_size

    src_stride_c0 = ((valid_rows + 256 - 1) // 256) * 256 // c0_size

    dst_stride = dst_cols

    dst_offset = indexRow * dst_cols + indexCol

    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()
    dst_start = pto.addptr(dst_ptr, dst_offset)

    pto.mte_l0c_ub(src_ptr, dst_start, valid_rows, valid_col_aligned, src_stride_c0, dst_stride, 0,
                   layout="nz2nd", pre_relu=("normal_relu", None, None))
    return None


@pto.ckernel(
    target="a5",
    op="pto.tinsert",
    dtypes=[
        (pto.f32, pto.i32, pto.i32, pto.f32),
    ],
    constraints=[_acc_to_mat_nz_relu_constraint],
)
def template_tinsert_acc_to_mat_nz_relu(src: pto.Tile, indexRow: pto.i32, indexCol: pto.i32, dst: pto.Tile):
    dtype = dst.element_type
    elem_bytes = pto.bytewidth(dtype)

    valid_rows, valid_cols = src.valid_shape
    src_rows = src.shape[0]
    dst_rows = dst.shape[0]

    c0_size = 32 // elem_bytes
    n_size = ((valid_cols + c0_size - 1) // c0_size) * c0_size

    src_stride_c0 = src_rows // c0_size
    dst_stride = dst_rows * c0_size

    dst_offset = dst_rows * indexCol + indexRow * c0_size

    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()
    dst_start = pto.addptr(dst_ptr, dst_offset)

    pto.mte_l0c_l1(src_ptr, dst_start, src_rows, n_size, src_stride_c0, dst_stride,
                   layout=("nz2nz", 0), pre_relu=("normal_relu", None, None))
    return None