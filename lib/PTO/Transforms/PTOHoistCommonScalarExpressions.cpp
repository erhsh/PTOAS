// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOHoistCommonScalarExpressions.cpp - Cross-branch scalar CSE -----===//

#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"

#include <optional>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOHOISTCOMMONSCALAREXPRESSIONS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr unsigned kMaxClonedExpressionOps = 64;
constexpr unsigned kMinimumWeightedSaving = 2;

// The weights are deliberately approximate. They are used only to reject
// rewrites that cannot remove more scalar work than they introduce.
static std::optional<unsigned> getScalarOpWeight(Operation *op) {
  if (op->getNumRegions() != 0 || op->getNumResults() != 1 || !isPure(op)) {
    return std::nullopt;
  }

  auto isScalarIntegerLike = [](Type type) {
    return type.isIndex() || isa<IntegerType>(type);
  };
  if (!llvm::all_of(op->getOperandTypes(), isScalarIntegerLike) ||
      !llvm::all_of(op->getResultTypes(), isScalarIntegerLike)) {
    return std::nullopt;
  }

  StringRef name = op->getName().getStringRef();
  std::optional<unsigned> weight =
      llvm::StringSwitch<std::optional<unsigned>>(name)
          .Case("arith.constant", 0)
          .Cases("arith.extsi", "arith.extui", "arith.trunci", 2)
          .Cases("arith.index_cast", "arith.index_castui", 2)
          .Cases("arith.addi", "arith.subi", "arith.muli", 1)
          .Cases("arith.andi", "arith.ori", "arith.xori", 1)
          .Cases("arith.shli", "arith.shrsi", "arith.shrui", 2)
          .Case("arith.cmpi", 1)
          .Cases("arith.divsi", "arith.divui", 8)
          .Cases("arith.remsi", "arith.remui", 8)
          .Default(std::nullopt);
  if (!weight) {
    return std::nullopt;
  }

  // Division and remainder ops are modeled as pure by Arith, but zero divisors
  // and signed minimum values with a -1 divisor are still undefined. Require a
  // known-safe divisor before moving these operations.
  if (name == "arith.divsi" || name == "arith.divui" || name == "arith.remsi" ||
      name == "arith.remui") {
    auto constant = op->getOperand(1).getDefiningOp<arith::ConstantOp>();
    auto value =
        constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr();
    if (!value || value.getValue().isZero()) {
      return std::nullopt;
    }
    bool isSignedDivisionOrRemainder =
        name == "arith.divsi" || name == "arith.remsi";
    if (isSignedDivisionOrRemainder && value.getValue().isAllOnes()) {
      return std::nullopt;
    }
  }
  return weight;
}

static bool isEligibleScalarOp(Operation *op) {
  return getScalarOpWeight(op).has_value();
}

static llvm::hash_code hashExpression(Value value,
                                      DenseMap<Value, llvm::hash_code> &memo) {
  if (auto it = memo.find(value); it != memo.end()) {
    return it->second;
  }
  Operation *def = value.getDefiningOp();
  if (!def || !isEligibleScalarOp(def)) {
    return hash_value(value);
  }
  llvm::hash_code hash = OperationEquivalence::computeHash(
      def, [&](Value operand) { return hashExpression(operand, memo); },
      OperationEquivalence::ignoreHashValue,
      OperationEquivalence::IgnoreLocations);
  memo.try_emplace(value, hash);
  return hash;
}

static bool equivalentExpressions(
    Value lhs, Value rhs,
    DenseMap<std::pair<Value, Value>, bool> &equivalenceMemo) {
  if (lhs == rhs) {
    return true;
  }
  std::pair<Value, Value> key(lhs, rhs);
  if (auto it = equivalenceMemo.find(key); it != equivalenceMemo.end()) {
    return it->second;
  }

  Operation *lhsDef = lhs.getDefiningOp();
  Operation *rhsDef = rhs.getDefiningOp();
  if (!lhsDef || !rhsDef || !isEligibleScalarOp(lhsDef) ||
      !isEligibleScalarOp(rhsDef)) {
    equivalenceMemo.try_emplace(key, false);
    return false;
  }

  // Scalar expressions are acyclic. Seed the memo to guard against malformed
  // IR and to avoid repeatedly comparing shared sub-DAGs.
  equivalenceMemo.try_emplace(key, false);
  bool equivalent = OperationEquivalence::isEquivalentTo(
      lhsDef, rhsDef,
      [&](Value lhsOperand, Value rhsOperand) {
        return success(
            equivalentExpressions(lhsOperand, rhsOperand, equivalenceMemo));
      },
      nullptr, OperationEquivalence::IgnoreLocations);
  equivalenceMemo[key] = equivalent;
  return equivalent;
}

struct ExpressionGroup {
  SmallVector<Operation *> operations;
};

static SmallVector<ExpressionGroup> collectExpressionGroups(func::FuncOp func) {
  SmallVector<ExpressionGroup> groups;
  DenseMap<size_t, SmallVector<unsigned>> groupsByHash;
  DenseMap<Value, llvm::hash_code> hashMemo;

  func.walk([&](Operation *op) {
    if (!isEligibleScalarOp(op) || op->use_empty()) {
      return;
    }
    size_t hash =
        static_cast<size_t>(hashExpression(op->getResult(0), hashMemo));
    SmallVector<unsigned> &bucket = groupsByHash[hash];
    for (unsigned groupIndex : bucket) {
      DenseMap<std::pair<Value, Value>, bool> equivalenceMemo;
      if (equivalentExpressions(
              groups[groupIndex].operations.front()->getResult(0),
              op->getResult(0), equivalenceMemo)) {
        groups[groupIndex].operations.push_back(op);
        return;
      }
    }
    bucket.push_back(groups.size());
    groups.push_back({{op}});
  });
  return groups;
}

static SmallVector<Operation *> getExecutionBarriers(Operation *op,
                                                     func::FuncOp func) {
  SmallVector<Operation *> barriers;
  for (Operation *parent = op->getParentOp(); parent && parent != func;
       parent = parent->getParentOp()) {
    // scf.if is the only boundary this pass is allowed to cross. Exact barrier
    // equality keeps expressions inside loops, SIMT scopes, execute_region,
    // and every unknown region-bearing operation.
    if (parent->getNumRegions() != 0 && !isa<scf::IfOp>(parent)) {
      barriers.push_back(parent);
    }
  }
  return barriers;
}

static Region *getDirectChildRegion(Operation *descendant,
                                    Operation *ancestor) {
  Operation *cursor = descendant;
  while (cursor && cursor->getParentOp() != ancestor) {
    cursor = cursor->getParentOp();
  }
  return cursor ? cursor->getParentRegion() : nullptr;
}

static bool areMutuallyExclusive(Operation *lhs, Operation *rhs) {
  for (Operation *parent = lhs->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto ifOp = dyn_cast<scf::IfOp>(parent);
    if (!ifOp || !ifOp->isProperAncestor(rhs)) {
      continue;
    }
    Region *lhsRegion = getDirectChildRegion(lhs, ifOp);
    Region *rhsRegion = getDirectChildRegion(rhs, ifOp);
    if (lhsRegion && rhsRegion && lhsRegion != rhsRegion) {
      return true;
    }
  }
  return false;
}

static bool hasCoExecutablePair(ArrayRef<Operation *> operations) {
  for (auto [index, lhs] : llvm::enumerate(operations)) {
    for (Operation *rhs : operations.drop_front(index + 1)) {
      if (!areMutuallyExclusive(lhs, rhs)) {
        return true;
      }
    }
  }
  return false;
}

static SmallVector<Block *> getBlockAncestors(Operation *op,
                                              func::FuncOp func) {
  SmallVector<Block *> blocks;
  for (Block *block = op->getBlock(); block;) {
    blocks.push_back(block);
    Operation *parent = block->getParentOp();
    if (!parent || parent == func) {
      break;
    }
    block = parent->getBlock();
  }
  return blocks;
}

static Block *findDeepestCommonBlock(ArrayRef<Operation *> operations,
                                     func::FuncOp func) {
  SmallVector<Block *> firstAncestors =
      getBlockAncestors(operations.front(), func);
  for (Block *candidate : firstAncestors) {
    bool common = llvm::all_of(operations.drop_front(), [&](Operation *op) {
      SmallVector<Block *> ancestors = getBlockAncestors(op, func);
      return llvm::is_contained(ancestors, candidate);
    });
    if (common) {
      return candidate;
    }
  }
  return nullptr;
}

static Operation *getAnchorInBlock(Operation *op, Block *block) {
  Operation *anchor = op;
  while (anchor && anchor->getBlock() != block) {
    anchor = anchor->getParentOp();
  }
  return anchor;
}

static Operation *findInsertionPoint(ArrayRef<Operation *> operations,
                                     func::FuncOp func) {
  Block *block = findDeepestCommonBlock(operations, func);
  if (!block) {
    return nullptr;
  }
  Operation *insertionPoint = nullptr;
  for (Operation *op : operations) {
    Operation *anchor = getAnchorInBlock(op, block);
    if (!anchor) {
      return nullptr;
    }
    if (!insertionPoint || anchor->isBeforeInBlock(insertionPoint)) {
      insertionPoint = anchor;
    }
  }
  return insertionPoint;
}

static bool canMaterializeBefore(Value value, Operation *insertionPoint,
                                 DominanceInfo &dominance,
                                 DenseMap<Value, bool> &memo) {
  if (dominance.dominates(value, insertionPoint)) {
    return true;
  }
  if (auto it = memo.find(value); it != memo.end()) {
    return it->second;
  }
  memo.try_emplace(value, false);
  Operation *def = value.getDefiningOp();
  if (!def || !isEligibleScalarOp(def)) {
    return false;
  }
  bool available = llvm::all_of(def->getOperands(), [&](Value operand) {
    return canMaterializeBefore(operand, insertionPoint, dominance, memo);
  });
  memo[value] = available;
  return available;
}

static bool collectCloneOps(Value value, Operation *insertionPoint,
                            DominanceInfo &dominance,
                            SmallPtrSetImpl<Operation *> &cloneOps) {
  if (dominance.dominates(value, insertionPoint)) {
    return true;
  }
  Operation *def = value.getDefiningOp();
  if (!def || !isEligibleScalarOp(def)) {
    return false;
  }
  if (!cloneOps.insert(def).second) {
    return true;
  }
  if (cloneOps.size() > kMaxClonedExpressionOps) {
    return false;
  }
  return llvm::all_of(def->getOperands(), [&](Value operand) {
    return collectCloneOps(operand, insertionPoint, dominance, cloneOps);
  });
}

static void collectExpressionDAG(Value value,
                                 SmallPtrSetImpl<Operation *> &dagOps) {
  Operation *def = value.getDefiningOp();
  if (!def || !isEligibleScalarOp(def) || !dagOps.insert(def).second) {
    return;
  }
  for (Value operand : def->getOperands()) {
    collectExpressionDAG(operand, dagOps);
  }
}

static SmallPtrSet<Operation *, 32>
findRemovableOps(ArrayRef<Operation *> roots) {
  SmallPtrSet<Operation *, 32> dagOps;
  SmallPtrSet<Operation *, 32> removable;
  for (Operation *root : roots) {
    removable.insert(root);
    collectExpressionDAG(root->getResult(0), dagOps);
  }

  bool changed;
  do {
    changed = false;
    for (Operation *op : dagOps) {
      if (removable.contains(op)) {
        continue;
      }
      if (llvm::all_of(op->getUsers(), [&](Operation *user) {
            return removable.contains(user);
          })) {
        changed |= removable.insert(op).second;
      }
    }
  } while (changed);
  return removable;
}

static unsigned getWeightedCost(const SmallPtrSetImpl<Operation *> &ops) {
  unsigned cost = 0;
  for (Operation *op : ops) {
    cost += *getScalarOpWeight(op);
  }
  return cost;
}

static bool hasDominatingOccurrence(ArrayRef<Operation *> operations,
                                    DominanceInfo &dominance) {
  return llvm::any_of(operations, [&](Operation *candidate) {
    Value value = candidate->getResult(0);
    return llvm::all_of(operations, [&](Operation *other) {
      return candidate == other || dominance.dominates(value, other);
    });
  });
}

struct HoistCandidate {
  SmallVector<Operation *> occurrences;
  Operation *insertionPoint = nullptr;
  unsigned weightedSaving = 0;
  unsigned cloneCost = 0;
};

static std::optional<HoistCandidate>
findBestCandidate(func::FuncOp func, DominanceInfo &dominance) {
  std::optional<HoistCandidate> best;
  for (ExpressionGroup &group : collectExpressionGroups(func)) {
    if (group.operations.size() < 2 ||
        hasDominatingOccurrence(group.operations, dominance) ||
        !hasCoExecutablePair(group.operations)) {
      continue;
    }

    SmallVector<Operation *> barriers =
        getExecutionBarriers(group.operations.front(), func);
    if (!llvm::all_of(ArrayRef(group.operations).drop_front(),
                      [&](Operation *op) {
                        return getExecutionBarriers(op, func) == barriers;
                      })) {
      continue;
    }

    Operation *insertionPoint = findInsertionPoint(group.operations, func);
    if (!insertionPoint) {
      continue;
    }
    DenseMap<Value, bool> availabilityMemo;
    if (!canMaterializeBefore(group.operations.front()->getResult(0),
                              insertionPoint, dominance, availabilityMemo)) {
      continue;
    }

    SmallPtrSet<Operation *, 32> cloneOps;
    if (!collectCloneOps(group.operations.front()->getResult(0), insertionPoint,
                         dominance, cloneOps)) {
      continue;
    }
    SmallPtrSet<Operation *, 32> removable = findRemovableOps(group.operations);
    unsigned cloneCost = getWeightedCost(cloneOps);
    unsigned removableCost = getWeightedCost(removable);
    if (removableCost < cloneCost + kMinimumWeightedSaving) {
      continue;
    }

    HoistCandidate candidate{group.operations, insertionPoint,
                             removableCost - cloneCost, cloneCost};
    if (!best || candidate.weightedSaving > best->weightedSaving ||
        (candidate.weightedSaving == best->weightedSaving &&
         candidate.cloneCost > best->cloneCost)) {
      best = std::move(candidate);
    }
  }
  return best;
}

static Value materializeBefore(Value value, Operation *insertionPoint,
                               DominanceInfo &dominance, OpBuilder &builder,
                               DenseMap<Value, Value> &memo) {
  if (dominance.dominates(value, insertionPoint)) {
    return value;
  }
  if (auto it = memo.find(value); it != memo.end()) {
    return it->second;
  }

  Operation *def = value.getDefiningOp();
  SmallVector<Value> operands;
  operands.reserve(def->getNumOperands());
  for (Value operand : def->getOperands()) {
    operands.push_back(
        materializeBefore(operand, insertionPoint, dominance, builder, memo));
  }
  builder.setInsertionPoint(insertionPoint);
  Operation *clone = builder.clone(*def);
  clone->setOperands(operands);
  Value result = clone->getResult(0);
  memo.try_emplace(value, result);
  return result;
}

static void eraseDeadExpressionDAGs(ArrayRef<Operation *> roots) {
  SmallPtrSet<Operation *, 32> dagOps;
  for (Operation *root : roots) {
    collectExpressionDAG(root->getResult(0), dagOps);
  }

  while (true) {
    Operation *dead = nullptr;
    for (Operation *op : dagOps) {
      if (isOpTriviallyDead(op)) {
        dead = op;
        break;
      }
    }
    if (!dead) {
      return;
    }
    dagOps.erase(dead);
    dead->erase();
  }
}

struct PTOHoistCommonScalarExpressionsPass
    : public pto::impl::PTOHoistCommonScalarExpressionsBase<
          PTOHoistCommonScalarExpressionsPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    DominanceInfo dominance(func);
    while (std::optional<HoistCandidate> candidate =
               findBestCandidate(func, dominance)) {
      OpBuilder builder(candidate->insertionPoint);
      DenseMap<Value, Value> materialized;
      Value hoisted = materializeBefore(
          candidate->occurrences.front()->getResult(0),
          candidate->insertionPoint, dominance, builder, materialized);
      for (Operation *op : candidate->occurrences) {
        op->getResult(0).replaceAllUsesWith(hoisted);
      }
      eraseDeadExpressionDAGs(candidate->occurrences);
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOHoistCommonScalarExpressionsPass() {
  return std::make_unique<PTOHoistCommonScalarExpressionsPass>();
}
