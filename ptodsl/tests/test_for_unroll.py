# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from ptodsl import pto


@pto.jit(target="a5", backend="vpto", mode="explicit")
def full_unroll_for_probe():
    with pto.for_(0, 4, step=1, unroll="full") as _:
        pto.pipe_barrier("ALL")


@pto.jit(target="a5", backend="vpto", mode="explicit")
def full_unroll_carry_probe():
    value = pto.const(0, dtype=pto.i32)
    loop = pto.for_(0, 4, step=1, unroll="full").carry(value=value)
    with loop:
        loop.update(value=loop.value + pto.const(1, dtype=pto.i32))
    _ = loop.final("value")


@pto.jit(target="a5", backend="vpto")
def auto_unroll_runtime_range_probe():
    for _ in pto.runtime_range(0, 4, 1, unroll="auto"):
        pto.pipe_barrier("ALL")


@pto.jit(target="a5", backend="vpto")
def auto_unroll_runtime_range_carry_probe():
    value = pto.const(0, dtype=pto.i32)
    for _ in pto.runtime_range(0, 4, 1, unroll="auto"):
        value = value + pto.const(1, dtype=pto.i32)
    _ = value


@pto.jit(target="a5", backend="vpto")
def default_runtime_range_probe():
    for _ in pto.runtime_range(4):
        pto.pipe_barrier("ALL")


@pto.jit(target="a5", backend="vpto")
def explicit_none_runtime_range_probe():
    for _ in pto.runtime_range(4, unroll=None):
        pto.pipe_barrier("ALL")


def test_for_full_unroll_attribute():
    text = full_unroll_for_probe.compile().mlir_text()
    assert 'pto.unroll = "full"' in text


def test_carry_for_full_unroll_attribute():
    text = full_unroll_carry_probe.compile().mlir_text()
    assert 'pto.unroll = "full"' in text


def test_runtime_range_auto_unroll_attribute():
    text = auto_unroll_runtime_range_probe.compile().mlir_text()
    assert 'pto.unroll = "auto"' in text


def test_runtime_range_auto_unroll_carry_attribute():
    text = auto_unroll_runtime_range_carry_probe.compile().mlir_text()
    assert 'pto.unroll = "auto"' in text


def test_runtime_range_default_has_no_unroll_attribute():
    text = default_runtime_range_probe.compile().mlir_text()
    assert "pto.unroll" not in text


def test_runtime_range_explicit_none_has_no_unroll_attribute():
    text = explicit_none_runtime_range_probe.compile().mlir_text()
    assert "pto.unroll" not in text


def test_for_rejects_unknown_unroll_mode():
    try:
        pto.for_(0, 4, step=1, unroll="partial")
    except ValueError as exc:
        assert "None, 'auto', or 'full'" in str(exc)
    else:
        raise AssertionError("expected invalid unroll mode to fail")


def test_runtime_range_rejects_unknown_unroll_mode():
    try:
        pto.runtime_range(0, 4, unroll="partial")
    except ValueError as exc:
        assert "None or 'auto'" in str(exc)
    else:
        raise AssertionError("expected invalid runtime range hint to fail")
