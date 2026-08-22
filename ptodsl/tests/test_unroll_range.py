import pytest

from ptodsl import pto


@pto.jit(target="a5")
def ast_unroll_range_probe():
    total = pto.const(0, dtype=pto.i32)
    one = pto.const(1, dtype=pto.i32)
    for _ in pto.unroll_range(0, 4, 1):
        total = total + one
    _ = total


@pto.jit(target="a5")
def ast_plain_range_probe():
    total = pto.const(0, dtype=pto.i32)
    one = pto.const(1, dtype=pto.i32)
    for _ in range(0, 4, 1):
        total = total + one
    _ = total


def test_unroll_range_lowers_to_annotated_scf_for_with_carry():
    text = ast_unroll_range_probe.compile().mlir_text()
    assert "scf.for" in text
    assert 'pto.unroll = "full"' in text


def test_plain_range_remains_unannotated():
    text = ast_plain_range_probe.compile().mlir_text()
    assert "scf.for" in text
    assert "pto.unroll" not in text


def test_for_rejects_unsupported_unroll_mode():
    with pytest.raises(ValueError, match="only supports None or 'full'"):
        pto.for_(0, 4, step=1, unroll="factor")
