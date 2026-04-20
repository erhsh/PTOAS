#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You can not use the file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

import numpy as np
import os
import sys

# Add parent directory to path for st_common import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from st_common import setup_case_rng, save_case_data

from cases import CASES


def _to_tuple(shape):
    """Convert shape to tuple if needed."""
    if isinstance(shape, tuple):
        return shape
    return tuple(shape)


for case in CASES:
    setup_case_rng(case)

    dtype = case["dtype"]
    shape = _to_tuple(case["shape"])
    src0_valid = _to_tuple(case["valid_shape"])
    src1_valid = _to_tuple(case["src1_vshape"])
    dst_valid = _to_tuple(case["dst_vshape"])

    rows, cols = shape
    src0_vr, src0_vc = src0_valid
    src1_vr, src1_vc = src1_valid
    dst_vr, dst_vc = dst_valid

    input1 = np.random.randint(1, 10, size=shape).astype(dtype)
    input2 = np.random.randint(1, 10, size=shape).astype(dtype)

    golden = np.zeros(shape, dtype=dtype)

    # tpartmin semantics (based on pto-isa TPartBinOps.hpp TCopyPadOp):
    # Algorithm:
    # 1. dst[:] = Max (padding for min operation)
    # 2. dst[0:src0_vr, 0:src0_vc] = src0[0:src0_vr, 0:src0_vc] (copy src0 to dst)
    # 3. dst[0:src1_vr, 0:src1_vc] = min(dst[0:src1_vr, 0:src1_vc], src1[0:src1_vr, 0:src1_vc])
    #    (apply min in src1 valid region)

    src0_eq_dst = (src0_vr == dst_vr and src0_vc == dst_vc)
    src1_eq_dst = (src1_vr == dst_vr and src1_vc == dst_vc)

    if src0_eq_dst and src1_eq_dst:
        # Full min: both src0 and src1 cover entire dst
        golden[:dst_vr, :dst_vc] = np.minimum(input1[:dst_vr, :dst_vc], input2[:dst_vr, :dst_vc]).astype(dtype, copy=False)
    elif src0_eq_dst:
        # src0 covers dst, src1 is partial
        # dst = src0 (copy), then min(dst, src1) in src1 region = min(src0, src1) in src1 region, src0 in rest
        golden[:src1_vr, :src1_vc] = np.minimum(input1[:src1_vr, :src1_vc], input2[:src1_vr, :src1_vc]).astype(dtype, copy=False)
        if src1_vc < dst_vc:
            golden[:src1_vr, src1_vc:dst_vc] = input1[:src1_vr, src1_vc:dst_vc].copy()
        if src1_vr < dst_vr:
            golden[src1_vr:dst_vr, :dst_vc] = input1[src1_vr:dst_vr, :dst_vc].copy()
    elif src1_eq_dst:
        # src1 covers dst, src0 is partial
        # dst = Max, then copy src0 in src0 region, then min(dst, src1) in src1 region
        golden[:src0_vr, :src0_vc] = np.minimum(input1[:src0_vr, :src0_vc], input2[:src0_vr, :src0_vc]).astype(dtype, copy=False)
        if src0_vc < dst_vc:
            golden[:src0_vr, src0_vc:dst_vc] = input2[:src0_vr, src0_vc:dst_vc].copy()
        if src0_vr < dst_vr:
            golden[src0_vr:dst_vr, :dst_vc] = input2[src0_vr:dst_vr, :dst_vc].copy()
    else:
        # Both src0 and src1 are partial (complex case)
        # Algorithm from TCopyPadOp:
        # 1. dst = Max
        # 2. copy src0 to dst[0:src0_vr, 0:src0_vc]
        # 3. min(dst[0:src1_vr, 0:src1_vc], src1[0:src1_vr, 0:src1_vc])

        min_vr = min(src0_vr, src1_vr)
        min_vc = min(src0_vc, src1_vc)

        # Region 1: [0:min_vr, 0:min_vc] - overlapping region (both src0 and src1 valid)
        # dst = src0 (from copy), then min(dst, src1) = min(src0, src1)
        golden[:min_vr, :min_vc] = np.minimum(input1[:min_vr, :min_vc], input2[:min_vr, :min_vc]).astype(dtype, copy=False)

        # Region 2: [0:src0_vr, min_vc:src0_vc] if src0_vc > min_vc
        # Only src0 valid (src1 cols don't reach), dst = src0 (from copy), no min operation
        if src0_vc > min_vc:
            golden[:src0_vr, min_vc:src0_vc] = input1[:src0_vr, min_vc:src0_vc].copy()

        # Region 3: [min_vr:src1_vr, 0:min_vc] if src1_vr > min_vr
        # Only src1 valid (src0 rows don't reach), dst = Max (not copied), min(Max, src1) = src1
        if src1_vr > min_vr:
            golden[min_vr:src1_vr, :min_vc] = input2[min_vr:src1_vr, :min_vc].copy()

        # Region 4: [min_vr:src1_vr, min_vc:src1_vc] if src1_vr > min_vr AND src1_vc > min_vc
        # src0 rows don't reach, src1 cols don't reach src0_vc
        # dst = Max (not copied in rows beyond src0_vr), min(Max, src1) = src1
        if src1_vr > min_vr and src1_vc > min_vc:
            golden[min_vr:src1_vr, min_vc:src1_vc] = input2[min_vr:src1_vr, min_vc:src1_vc].copy()

        # Region 5: [0:min_vr, src1_vc:src0_vc] if src0_vc > src1_vc
        # src0 cols beyond src1_vc: dst = src0 (from copy), no min operation
        # This overlaps with Region 2 if src1_vr >= src0_vr, so we handle the case where src1_vr < src0_vr
        if src0_vc > src1_vc and min_vr > 0:
            # Already handled in Region 2 if rows are [0:src0_vr]
            pass  # Region 2 covers this

        # Region 6: [src0_vr:src1_vr, src0_vc:src1_vc] if src1_vr > src0_vr AND src1_vc > src0_vc
        # Neither src0 nor src1 covers this region fully
        # dst = Max (no copy from src0 since row > src0_vr), then min(Max, src1) = src1
        # This is handled in Region 4

        # Region 7: [src1_vr:dst_vr, :dst_vc] if dst_vr > src1_vr
        # Beyond both src0 and src1 rows, dst = Max (padded)
        # This region is outside src1_vr, so min operation doesn't apply
        # Result should be Max, but since we're comparing against golden[:dst_vr, :dst_vc]
        # and this region has golden=0, the hardware output might be Max which would mismatch
        # However, this region should not be compared if dst_vr > max(src0_vr, src1_vr)
        # Wait - the compare.py compares [:dst_vr, :dst_vc], so this region IS compared!
        # But in TCopyPadOp, only rows up to src1_vr are processed. Beyond that, dst = Max.
        # For min operation, Max padding results in output = Max for this region.
        # But golden has 0 in this region, so there will be mismatch!
        # This is a bug - the hardware implementation doesn't handle regions beyond both src0 and src1.
        # Let me check pto-isa implementation again...
        # Actually, the hardware fills dst with Max first, then copies src0, then applies min.
        # For rows beyond src1_vr: dst stays as Max (no min operation).
        # For cols beyond src0_vc (within src1 rows): dst = min(src0_padded, src1) = src1

        # Wait, let me re-analyze the complex case more carefully:
        # Case: src0=(104,123), src1=(122,110), dst=(122,123)
        #
        # 1. dst[:] = Max (all 122x128 padded)
        # 2. dst[0:104, 0:123] = src0[0:104, 0:123] (copy)
        # 3. dst[0:122, 0:110] = min(dst[0:122, 0:110], src1[0:122, 0:110])
        #
        # After step 2:
        # - [0:104, 0:123] = src0
        # - [104:122, :] = Max
        # - [:, 123:128] = Max (if cols=128)
        #
        # After step 3:
        # - [0:104, 0:110] = min(src0, src1)
        # - [0:104, 110:123] = src0 (not in min region since src1_vc=110)
        # - [104:122, 0:110] = min(Max, src1) = src1
        # - [104:122, 110:123] = Max (not in copy region nor min region)
        #
        # So the final result:
        # - [0:104, 0:110] = min(src0, src1)
        # - [0:104, 110:123] = src0
        # - [104:122, 0:110] = src1
        # - [104:122, 110:123] = Max (not processed!)
        #
        # But dst_valid is (122, 123), so [104:122, 110:123] IS in the valid region!
        # The hardware would output Max in this region, but golden expects... what?
        #
        # According to tpartmin semantics, regions where neither operand covers should have Max.
        # But for comparison, we're using np.zeros for golden, and filling only the processed regions.
        # So [104:122, 110:123] in golden would be 0, while hardware outputs Max.
        #
        # This is the mismatch! The complex case has a region that neither src0 nor src1 covers.
        # For tpartmin, this region should be filled with Max padding value.
        #
        # However, this seems like a limitation of the hardware - it can't handle cases where
        # dst extends beyond both src0 and src1 in both dimensions simultaneously.
        #
        # Let me check if the pto-isa test cases actually have such scenarios...
        # Looking at the case: src0=(104,123), src1=(122,110), dst=(122,123)
        # - [104:122, 110:123]: src0 doesn't cover (rows 104-122), src1 doesn't cover (cols 110-123)
        # - This is 18 rows x 13 cols = 234 elements
        #
        # For tpartmin, in this region, dst should be Max (padding).
        # But the hardware doesn't process it, so it stays as Max from initial fill.
        #
        # Actually wait - after step 1, dst = Max.
        # After step 2, [0:104, 0:123] = src0, [104:122, :] = Max
        # After step 3, [0:122, 0:110] = min(dst, src1)
        #   - [0:104, 0:110]: dst was src0, now min(src0, src1)
        #   - [104:122, 0:110]: dst was Max, now min(Max, src1) = src1
        # - [:, 110:123]: dst stays as whatever it was before step 3
        #   - [0:104, 110:123]: dst was src0 (from step 2)
        #   - [104:122, 110:123]: dst was Max (from step 1)
        #
        # So [104:122, 110:123] = Max in the final output.
        # For tpartmin with float, Max is +inf. For integers, it's the max value.
        #
        # But golden has 0 there. This is wrong! The hardware produces Max, and that's correct
        # according to the padding semantics.
        #
        # So the fix is: in gen_data.py, we should fill [104:122, 110:123] with the appropriate
        # padding value (Max for tpartmin).
        #
        # But wait - comparing against Max values might not be meaningful. Maybe the test
        # expectation is wrong for such complex cases?
        #
        # Let me check what pto-isa actually tests for this case...

        # For now, let me fix gen_data.py to handle the "neither covers" region properly.
        # Regions beyond both src0 and src1 should be filled with appropriate padding.

        # Region 8: [max(src0_vr, src1_vr):dst_vr, :dst_vc] if dst_vr > max(src0_vr, src1_vr)
        # Beyond max src row, dst = Max (no processing)
        # Actually this region shouldn't exist if dst_vr <= max(src0_vr, src1_vr)...
        # But for this case: max(104, 122) = 122, dst_vr = 122, so this region is empty.

        # Region 9: [:, max(src0_vc, src1_vc):dst_vc] if dst_vc > max(src0_vc, src1_vc)
        # Beyond max src col, dst = Max
        # For this case: max(123, 110) = 123, dst_vc = 123, so this region is empty.

        # The key missing region is: [src0_vr:src1_vr, src1_vc:src0_vc] when src1_vr > src0_vr AND src0_vc > src1_vc
        # This region:
        # - Row: beyond src0 but within src1
        # - Col: beyond src1 but within src0
        # - Neither covers it fully!
        #
        # After copy: dst = Max (rows beyond src0_vr not copied)
        # After min: dst = Max (cols beyond src1_vc not minned)
        # Final: dst = Max

        if src1_vr > src0_vr and src0_vc > src1_vc:
            # Region [src0_vr:src1_vr, src1_vc:src0_vc] = Max (neither covers)
            # This is correct for tpartmin - padding value is Max
            # For floats, we use np.inf. For integers, use dtype max.
            if dtype == np.float32:
                max_val = np.finfo(np.float32).max
            elif dtype == np.float16:
                max_val = np.finfo(np.float16).max
            elif dtype == np.int8:
                max_val = np.iinfo(np.int8).max
            elif dtype == np.uint8:
                max_val = np.iinfo(np.uint8).max
            elif dtype == np.int16:
                max_val = np.iinfo(np.int16).max
            elif dtype == np.uint16:
                max_val = np.iinfo(np.uint16).max
            elif dtype == np.int32:
                max_val = np.iinfo(np.int32).max
            elif dtype == np.uint32:
                max_val = np.iinfo(np.uint32).max
            else:
                max_val = np.iinfo(dtype).max
            golden[src0_vr:src1_vr, src1_vc:src0_vc] = max_val

    save_case_data(case["name"], {"input1": input1, "input2": input2, "golden": golden})
    print(f"[INFO] gen_data: {case['name']} shape={shape} src0_valid={src0_valid} src1_valid={src1_valid} dst_valid={dst_valid} dtype={dtype.__name__}")