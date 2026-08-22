// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOUnrollSIMTForPass.cpp -------------------------------------------===//
//
// Unroll explicitly annotated scf.for loops inside SIMT contexts to eliminate
// divergent control flow before LLVM lowering.
//
// Only loops carrying the `{pto.unroll = "full"}` attribute are unrolled.
// A SIMT context is either a pto.simt_entry function or an inline
// pto.section.simt region that has not been outlined yet.  General-purpose
// loops outside these contexts are not affected.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOUNROLLSIMTFOR
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "pto-unroll-simt-for"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Name of the unroll annotation placed on scf::ForOp by users.
static constexpr llvm::StringLiteral kUnrollAttrName = "pto.unroll";
static constexpr llvm::StringLiteral kUnrollFullValue = "full";
static constexpr llvm::StringLiteral kUnrollAutoValue = "auto";

/// Check whether the loop has the explicit "full unroll" annotation.
static bool hasUnrollFullAttr(scf::ForOp forOp) {
  if (auto attr = forOp->getAttrOfType<StringAttr>(kUnrollAttrName)) {
    return attr.getValue() == kUnrollFullValue;
  }
  return false;
}

static bool hasUnrollAutoAttr(scf::ForOp forOp) {
  if (auto attr = forOp->getAttrOfType<StringAttr>(kUnrollAttrName)) {
    return attr.getValue() == kUnrollAutoValue;
  }
  return false;
}

struct AutoCost {
  int64_t bodyOps = 0;
  int64_t ivDependentOps = 0;
  int64_t memoryOps = 0;
  int64_t valueResults = 0;
  int64_t expandedOps = 0;
  int64_t pressure = 0;
  int64_t benefit = 0;
  int64_t cost = 0;
  std::string reason;
};

static AutoCost evaluateAutoCost(scf::ForOp forOp, int64_t tripCount,
                                 int64_t maxTripCount,
                                 int64_t maxExpandedOps,
                                 int64_t maxPressure) {
  AutoCost result;
  llvm::DenseSet<Value> ivDependent;
  ivDependent.insert(forOp.getInductionVar());
  for (Operation &op : forOp.getBody()->without_terminator()) {
    op.walk([&](Operation *nested) {
      ++result.bodyOps;
      result.valueResults += nested->getNumResults();
      StringRef name = nested->getName().getStringRef();
      const bool isMemoryOp = name.contains("load") || name.contains("store") ||
                              name.contains("alloca");
      if (isMemoryOp) {
        ++result.memoryOps;
      }
      bool depends = llvm::any_of(nested->getOperands(), [&](Value operand) {
        return ivDependent.contains(operand);
      });
      if (depends) {
        ++result.ivDependentOps;
        for (Value value : nested->getResults()) {
          ivDependent.insert(value);
        }
      }
    });
  }

  if (tripCount > maxTripCount) {
    result.reason = "trip-count-safety-ceiling";
    return result;
  }
  if (result.bodyOps > 0 && tripCount > maxExpandedOps / result.bodyOps) {
    result.reason = "expanded-op-budget";
    return result;
  }
  result.expandedOps = result.bodyOps * tripCount;
  result.pressure = result.valueResults + 2 * result.memoryOps;
  if (result.pressure > maxPressure) {
    result.reason = "pressure-budget";
    return result;
  }

  // The dynamic benefit accounts for loop control and IV-dependent scalar
  // address/control expressions removed by folding. Code growth, repeated
  // memory scaffold, and a fixed instruction-cache setup cost oppose it.
  result.benefit = tripCount * (3 + result.ivDependentOps) * 8;
  result.cost = result.bodyOps * (tripCount - 1) +
                2 * result.memoryOps * tripCount + 4096;
  result.reason = result.benefit >= result.cost ? "profitable" : "not-profitable";
  return result;
}

/// Check whether this function is a SIMT entry.
static bool isSIMTEntry(func::FuncOp func) {
  return func->hasAttr(pto::kPTOSimtEntryAttrName);
}

/// Check whether a function contains an inline SIMT section.
static bool containsInlineSIMTSection(func::FuncOp func) {
  WalkResult result =
      func.walk([](pto::SectionSimtOp) { return WalkResult::interrupt(); });
  return result.wasInterrupted();
}

/// Check whether a loop is inside either supported SIMT representation.
static bool isInSIMTContext(scf::ForOp forOp) {
  func::FuncOp func = forOp->getParentOfType<func::FuncOp>();
  if (!func) {
    return false;
  }
  return isSIMTEntry(func) ||
         static_cast<bool>(forOp->getParentOfType<pto::SectionSimtOp>());
}

// ---------------------------------------------------------------------------
// Rewrite pattern
// ---------------------------------------------------------------------------

namespace {

struct UnrollSIMTForPattern : public OpRewritePattern<scf::ForOp> {
  UnrollSIMTForPattern(MLIRContext *ctx, StringRef mode,
                       int64_t maxTripCount, int64_t maxExpandedOps,
                       int64_t maxPressure, bool functionBudgetExceeded,
                       std::string *report)
      : OpRewritePattern<scf::ForOp>(ctx), mode(mode.str()),
        maxTripCount(maxTripCount), maxExpandedOps(maxExpandedOps),
        maxPressure(maxPressure), functionBudgetExceeded(functionBudgetExceeded),
        report(report) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    // Only apply inside an outlined or inline SIMT context.
    if (!isInSIMTContext(forOp)) {
      return failure();
    }

    const bool forceFull = hasUnrollFullAttr(forOp);
    const bool autoCandidate = hasUnrollAutoAttr(forOp);
    if (!forceFull && !autoCandidate) {
      return failure();
    }

    std::optional<int64_t> lb = getConstantIntValue(forOp.getLowerBound());
    std::optional<int64_t> ub = getConstantIntValue(forOp.getUpperBound());
    std::optional<int64_t> step = getConstantIntValue(forOp.getStep());
    if (!lb || !ub || !step || *step <= 0 || *ub <= *lb) {
      return failure();
    }

    int64_t tripCount = (*ub - *lb + *step - 1) / *step;
    if (tripCount <= 0) {
      return failure();
    }

    if (autoCandidate) {
      AutoCost autoCost = evaluateAutoCost(forOp, tripCount, maxTripCount,
                                           maxExpandedOps, maxPressure);
      if (functionBudgetExceeded && autoCost.reason == "profitable") {
        autoCost.reason = "function-expanded-op-budget";
      }
      const bool accepted = autoCost.reason == "profitable";
      if (report) {
        llvm::raw_string_ostream os(*report);
        os << "PTOUnrollSIMTFor auto trip=" << tripCount
           << " body_ops=" << autoCost.bodyOps
           << " iv_ops=" << autoCost.ivDependentOps
           << " memory_ops=" << autoCost.memoryOps
           << " expanded_ops=" << autoCost.expandedOps
           << " pressure=" << autoCost.pressure
           << " benefit=" << autoCost.benefit << " cost=" << autoCost.cost
           << " decision=" << (accepted ? "unroll" : "keep")
           << " reason=" << autoCost.reason << "\n";
      }
      if (mode != "on" || !accepted) {
        return failure();
      }
    }

    LLVM_DEBUG(llvm::dbgs()
               << "PTOUnrollSIMTFor: unrolling annotated scf.for tripCount="
               << tripCount << " at " << forOp.getLoc() << "\n");

    // loopUnrollByFactor returns failure if the loop carries iteration
    // arguments that have uses outside the loop (live-out values).  In that
    // case we cannot unroll.
    if (failed(loopUnrollByFactor(forOp, static_cast<uint64_t>(tripCount)))) {
      return failure();
    }

    return success();
  }

private:
  std::string mode;
  int64_t maxTripCount;
  int64_t maxExpandedOps;
  int64_t maxPressure;
  bool functionBudgetExceeded;
  std::string *report;
};

} // namespace

// ---------------------------------------------------------------------------
// Pass definition
// ---------------------------------------------------------------------------

namespace {

struct PTOUnrollSIMTFor : public pto::impl::PTOUnrollSIMTForBase<PTOUnrollSIMTFor> {
  using pto::impl::PTOUnrollSIMTForBase<
      PTOUnrollSIMTFor>::PTOUnrollSIMTForBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (mode != "on" && mode != "off" && mode != "analyze") {
      func.emitError("invalid auto-unroll mode '") << mode
          << "'; expected off, analyze, or on";
      return signalPassFailure();
    }
    const bool hasSIMTContext = isSIMTEntry(func) || containsInlineSIMTSection(func);
    if (!hasSIMTContext) {
      return;
    }

    std::string report;
    int64_t profitableExpandedOps = 0;
    func.walk([&](scf::ForOp forOp) {
      const bool isAutoCandidate =
          isInSIMTContext(forOp) && hasUnrollAutoAttr(forOp);
      if (!isAutoCandidate) {
        return;
      }
      std::optional<int64_t> lb = getConstantIntValue(forOp.getLowerBound());
      std::optional<int64_t> ub = getConstantIntValue(forOp.getUpperBound());
      std::optional<int64_t> step = getConstantIntValue(forOp.getStep());
      if (!lb || !ub || !step || *step <= 0 || *ub <= *lb) {
        return;
      }
      int64_t tripCount = (*ub - *lb + *step - 1) / *step;
      AutoCost cost = evaluateAutoCost(forOp, tripCount, maxAutoTripCount,
                                       maxAutoExpandedOps, maxAutoPressure);
      if (cost.reason == "profitable") {
        if (cost.expandedOps > maxAutoExpandedOps - profitableExpandedOps) {
          profitableExpandedOps = maxAutoExpandedOps + 1;
        } else {
          profitableExpandedOps += cost.expandedOps;
        }
      }
    });
    const bool functionBudgetExceeded =
        profitableExpandedOps > maxAutoExpandedOps;
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<UnrollSIMTForPattern>(ctx, mode, maxAutoTripCount,
                                      maxAutoExpandedOps, maxAutoPressure,
                                      functionBudgetExceeded,
                                      mode == "analyze" ? &report : nullptr);

    GreedyRewriteConfig config;
    config.maxIterations = 10; // loops may nest
    config.strictMode = GreedyRewriteStrictness::ExistingOps;

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns), config))) {
      signalPassFailure();
    }
    if (!report.empty()) {
      static std::mutex reportMutex;
      std::lock_guard<std::mutex> lock(reportMutex);
      llvm::errs() << report;
    }
  }
};

} // namespace

// ---------------------------------------------------------------------------
// Pass constructor
// ---------------------------------------------------------------------------

std::unique_ptr<Pass> mlir::pto::createPTOUnrollSIMTForPass(
    const PTOUnrollSIMTForOptions &options) {
  return std::make_unique<PTOUnrollSIMTFor>(options);
}
