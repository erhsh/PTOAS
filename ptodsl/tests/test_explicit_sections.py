#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# This software is provided on an "AS IS" basis, without warranties of any kind.

from ptodsl import pto


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_raises(callback, exc_type, *message_fragments: str) -> None:
    try:
        callback()
    except exc_type as exc:
        text = str(exc)
        for fragment in message_fragments:
            expect(fragment in text, f"expected diagnostic fragment {fragment!r} in {text!r}")
    else:
        raise AssertionError(f"expected {exc_type.__name__} to be raised")


@pto.jit(target="a5", mode="explicit", insert_sync=False)
def explicit_mixed_section_probe(
    lhs: pto.ptr(pto.bf16, "gm"),
    rhs: pto.ptr(pto.bf16, "gm"),
    dst: pto.ptr(pto.f32, "gm"),
    event_id: pto.index,
):
    zero = pto.const(0, dtype=pto.i64)
    lhs_l1 = pto.castptr(zero, pto.ptr(pto.bf16, "mat"))
    rhs_l1 = pto.castptr(zero, pto.ptr(pto.bf16, "mat"))
    lhs_l0 = pto.castptr(zero, pto.ptr(pto.bf16, "left"))
    rhs_l0 = pto.castptr(zero, pto.ptr(pto.bf16, "right"))
    acc = pto.castptr(zero, pto.ptr(pto.f32, "acc"))
    out = pto.castptr(zero, pto.ptr(pto.f32, "vec"))

    with pto.cube_section():
        pto.mte_gm_l1(lhs, lhs_l1, 32, nburst=(1, 0, 0))
        pto.mte_gm_l1(rhs, rhs_l1, 32, nburst=(1, 0, 0))
        pto.mte_l1_l0a(lhs_l1, lhs_l0, 16, 16)
        pto.mte_l1_l0b(rhs_l1, rhs_l0, 16, 16)
        pto.mad(lhs_l0, rhs_l0, acc, 16, 16, 16)
        pto.set_intra_flag(pto.Pipe.FIX, event_id)

    with pto.vector_section():
        pto.wait_intra_flag(pto.Pipe.MTE3, event_id)
        pto.mte_l0c_ub(acc, out, 16, 16, 16, 16, split=pto.SplitMode.M)
        pto.mte_ub_gm(out, dst, 1024, nburst=(1, 0, 0))


@pto.jit(target="a5", mode="auto")
def automatic_section_probe():
    with pto.vector_section():
        pto.const(0)


@pto.jit(target="a5", mode="explicit")
def nested_section_probe():
    with pto.cube_section():
        with pto.vector_section():
            pto.const(0)


@pto.jit(target="a5", mode="explicit")
def escaped_section_value_probe():
    with pto.vector_section():
        leaked = pto.const(1, dtype=pto.i64)
    pto.castptr(leaked, pto.ptr(pto.f32, "ub"))


@pto.jit(target="a5", mode="explicit", kernel_kind="cube")
def matching_kind_section_probe():
    with pto.cube_section():
        pto.const(0)


@pto.jit(target="a5", mode="explicit", kernel_kind="vector")
def mismatched_kind_section_probe():
    with pto.cube_section():
        pto.const(0)


@pto.jit(target="a5", mode="explicit", backend="emitc")
def emitc_section_probe():
    with pto.vector_section():
        pto.const(0)


def use_section_outside_trace():
    with pto.cube_section():
        pass


def main() -> None:
    explicit_mixed_section_probe.verify()
    text = explicit_mixed_section_probe.mlir_text()
    expect(text.count("pto.section.cube") == 1, text)
    expect(text.count("pto.section.vector") == 1, text)
    expect("pto.kernel_kind" not in text, text)
    expect("func.call" not in text, text)
    expect("func.func private @inline_" not in text, text)
    expect("pto.mte_gm_l1" in text and "pto.mte_l0c_ub" in text, text)
    expect("pto.sync.set" in text and "pto.sync.wait" in text, text)

    matching_kind_section_probe.verify()
    matching_text = matching_kind_section_probe.mlir_text()
    expect("pto.kernel_kind = #pto.kernel_kind<cube>" in matching_text, matching_text)
    expect("pto.section.cube" in matching_text, matching_text)

    expect_raises(
        automatic_section_probe.verify,
        RuntimeError,
        "pto.vector_section()",
        'mode="explicit"',
    )
    expect_raises(
        nested_section_probe.verify,
        RuntimeError,
        "cannot be nested",
    )
    expect_raises(
        escaped_section_value_probe.verify,
        RuntimeError,
        "escape the scope boundary",
    )
    expect_raises(
        mismatched_kind_section_probe.verify,
        RuntimeError,
        "kernel_kind='vector'",
    )
    expect_raises(
        emitc_section_probe.verify,
        RuntimeError,
        'backend="vpto"',
    )
    expect_raises(
        use_section_outside_trace,
        RuntimeError,
        "pto.cube_section()",
        'mode="explicit"',
    )


if __name__ == "__main__":
    main()
