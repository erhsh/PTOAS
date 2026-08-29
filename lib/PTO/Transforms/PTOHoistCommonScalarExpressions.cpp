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
#include "llvm/Support/Debug.h"

#include <cstdint>
#include <optional>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOHOISTCOMMONSCALAREXPRESSIONS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-hoist-common-scalar-expressions"

namespace {

constexpr unsigned kMaxClonedExpressionOps = 64;
constexpr unsigned kMinimumWeightedSaving = 2;
constexpr unsigned kMinimumExpectedWeightedSaving = 1;
constexpr unsigned kMaxCostModelPredicates = 8;
constexpr unsigned kCostScale = 1U << kMaxCostModelPredicates;
constexpr unsigned kConditionalRegionsPerLiveRangeUnit = 4;
constexpr unsigned kSpeculationRiskDivisor = 8;
constexpr unsigned kLiveRangeRiskDivisor = 2;
constexpr unsigned kMinimumOccurrencesToAmortizeLiveRange = 4;
constexpr unsigned kMaxUnguardedLiveRangeSpanUnits = 2;
constexpr unsigned kGuardControlFlowCost = 2;

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

  // Remainder ops are modeled as pure by Arith, but a zero divisor is still
  // undefined. Require a known-safe divisor before moving division/remainder.
  if (name == "arith.divsi" || name == "arith.divui" || name == "arith.remsi" ||
      name == "arith.remui") {
    auto constant = op->getOperand(1).getDefiningOp<arith::ConstantOp>();
    auto value =
        constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr();
    if (!value || value.getValue().isZero()) {
      return std::nullopt;
    }
    if (name == "arith.divsi" && value.getValue().isAllOnes()) {
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

struct PredicateLiteral {
  Value predicate;
  bool value;
};

struct NormalizedPredicate {
  Value predicate;
  bool negated = false;
};

static std::optional<bool> getI1Constant(Value value) {
  if (!value.getType().isInteger(1)) {
    return std::nullopt;
  }
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  auto integer =
      constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr();
  if (!integer) {
    return std::nullopt;
  }
  return !integer.getValue().isZero();
}

// Canonicalize the boolean negation forms emitted by the frontends. In
// particular, arith.cmpi eq, %condition, false is the common representation of
// logical-not after SCF canonicalization.
static NormalizedPredicate normalizePredicate(Value predicate) {
  bool negated = false;
  for (unsigned depth = 0; depth < kMaxClonedExpressionOps; ++depth) {
    if (auto xorOp = predicate.getDefiningOp<arith::XOrIOp>()) {
      Value lhs = xorOp->getOperand(0);
      Value rhs = xorOp->getOperand(1);
      std::optional<bool> lhsConstant = getI1Constant(lhs);
      std::optional<bool> rhsConstant = getI1Constant(rhs);
      if (lhsConstant || rhsConstant) {
        bool constant = lhsConstant ? *lhsConstant : *rhsConstant;
        predicate = lhsConstant ? rhs : lhs;
        negated ^= constant;
        continue;
      }
    }

    if (auto cmp = predicate.getDefiningOp<arith::CmpIOp>()) {
      arith::CmpIPredicate comparison = cmp.getPredicate();
      if (comparison != arith::CmpIPredicate::eq &&
          comparison != arith::CmpIPredicate::ne) {
        break;
      }
      Value lhs = cmp->getOperand(0);
      Value rhs = cmp->getOperand(1);
      std::optional<bool> lhsConstant = getI1Constant(lhs);
      std::optional<bool> rhsConstant = getI1Constant(rhs);
      if (!lhsConstant && !rhsConstant) {
        break;
      }
      bool constant = lhsConstant ? *lhsConstant : *rhsConstant;
      predicate = lhsConstant ? rhs : lhs;
      negated ^= comparison == arith::CmpIPredicate::eq ? !constant : constant;
      continue;
    }
    break;
  }
  return {predicate, negated};
}

static SmallVector<PredicateLiteral> getControlPredicates(Operation *op) {
  SmallVector<PredicateLiteral> predicates;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto ifOp = dyn_cast<scf::IfOp>(parent);
    if (!ifOp) {
      continue;
    }
    Region *region = getDirectChildRegion(op, ifOp);
    if (!region) {
      continue;
    }
    bool inThenRegion = region == &ifOp.getThenRegion();
    bool inElseRegion = region == &ifOp.getElseRegion();
    if (!inThenRegion && !inElseRegion) {
      continue;
    }
    NormalizedPredicate normalized = normalizePredicate(ifOp.getCondition());
    predicates.push_back(
        {normalized.predicate, inThenRegion != normalized.negated});
  }
  return predicates;
}

static bool equivalentPredicates(Value lhs, Value rhs) {
  if (lhs == rhs) {
    return true;
  }
  DenseMap<std::pair<Value, Value>, bool> equivalenceMemo;
  return equivalentExpressions(lhs, rhs, equivalenceMemo);
}

static bool areMutuallyExclusive(Operation *lhs, Operation *rhs) {
  SmallVector<PredicateLiteral> lhsPredicates = getControlPredicates(lhs);
  SmallVector<PredicateLiteral> rhsPredicates = getControlPredicates(rhs);
  for (const PredicateLiteral &lhsPredicate : lhsPredicates) {
    for (const PredicateLiteral &rhsPredicate : rhsPredicates) {
      if (lhsPredicate.value != rhsPredicate.value &&
          equivalentPredicates(lhsPredicate.predicate,
                               rhsPredicate.predicate)) {
        return true;
      }
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

struct PredicateRequirement {
  unsigned variable;
  bool value;
};

struct ConditionalOpCost {
  unsigned weight;
  SmallVector<PredicateRequirement> requirements;
};

struct ConditionalCostEstimate {
  uint64_t expectedCostScaled;
  unsigned minimumCost;
};

static std::optional<bool>
findPredicateValue(Value predicate, ArrayRef<PredicateLiteral> predicates) {
  for (const PredicateLiteral &candidate : predicates) {
    if (equivalentPredicates(predicate, candidate.predicate)) {
      return candidate.value;
    }
  }
  return std::nullopt;
}

static std::optional<PredicateLiteral>
findCommonDominatingPredicate(ArrayRef<Operation *> operations,
                              Operation *insertionPoint,
                              DominanceInfo &dominance) {
  SmallVector<PredicateLiteral> insertionPredicates =
      getControlPredicates(insertionPoint);
  for (const PredicateLiteral &candidate :
       getControlPredicates(operations.front())) {
    if (getI1Constant(candidate.predicate) ||
        findPredicateValue(candidate.predicate, insertionPredicates) ||
        !dominance.dominates(candidate.predicate, insertionPoint)) {
      continue;
    }
    bool common = llvm::all_of(operations.drop_front(), [&](Operation *op) {
      std::optional<bool> value =
          findPredicateValue(candidate.predicate, getControlPredicates(op));
      return value && *value == candidate.value;
    });
    if (common) {
      return candidate;
    }
  }
  return std::nullopt;
}

static unsigned getOrAddPredicateVariable(Value predicate,
                                          SmallVectorImpl<Value> &variables) {
  for (auto [index, candidate] : llvm::enumerate(variables)) {
    if (equivalentPredicates(predicate, candidate)) {
      return index;
    }
  }
  variables.push_back(predicate);
  return variables.size() - 1;
}

// Estimate the dynamic cost before hoisting under an unbiased distribution of
// unknown predicates. minimumCost captures the least-work path and is used to
// measure the worst additional work caused by speculation.
static std::optional<ConditionalCostEstimate>
estimateConditionalCost(const SmallPtrSetImpl<Operation *> &ops,
                        Operation *insertionPoint,
                        ArrayRef<PredicateLiteral> assumedPredicates = {}) {
  SmallVector<PredicateLiteral> insertionPredicates =
      getControlPredicates(insertionPoint);
  insertionPredicates.append(assumedPredicates.begin(),
                             assumedPredicates.end());
  SmallVector<Value> variables;
  SmallVector<ConditionalOpCost> conditionalCosts;

  for (Operation *op : ops) {
    unsigned weight = *getScalarOpWeight(op);
    if (weight == 0) {
      continue;
    }

    bool reachable = true;
    SmallVector<PredicateRequirement> requirements;
    for (const PredicateLiteral &literal : getControlPredicates(op)) {
      if (std::optional<bool> constant = getI1Constant(literal.predicate)) {
        if (*constant != literal.value) {
          reachable = false;
          break;
        }
        continue;
      }
      if (std::optional<bool> insertionValue =
              findPredicateValue(literal.predicate, insertionPredicates)) {
        if (*insertionValue != literal.value) {
          reachable = false;
          break;
        }
        continue;
      }

      unsigned variable =
          getOrAddPredicateVariable(literal.predicate, variables);
      auto existing = llvm::find_if(requirements, [&](const auto &requirement) {
        return requirement.variable == variable;
      });
      if (existing != requirements.end()) {
        if (existing->value != literal.value) {
          reachable = false;
          break;
        }
        continue;
      }
      requirements.push_back({variable, literal.value});
    }
    if (variables.size() > kMaxCostModelPredicates) {
      return std::nullopt;
    }
    if (reachable) {
      conditionalCosts.push_back({weight, std::move(requirements)});
    }
  }

  unsigned assignmentCount = 1U << variables.size();
  uint64_t totalCost = 0;
  std::optional<unsigned> minimumCost;
  for (unsigned assignment = 0; assignment < assignmentCount; ++assignment) {
    unsigned cost = 0;
    for (const ConditionalOpCost &conditionalCost : conditionalCosts) {
      bool executes = llvm::all_of(
          conditionalCost.requirements, [&](const auto &requirement) {
            return static_cast<bool>(assignment &
                                     (1U << requirement.variable)) ==
                   requirement.value;
          });
      if (executes) {
        cost += conditionalCost.weight;
      }
    }
    totalCost += cost;
    minimumCost = minimumCost ? std::min(*minimumCost, cost) : cost;
  }

  return ConditionalCostEstimate{totalCost * (kCostScale / assignmentCount),
                                 minimumCost.value_or(0)};
}

static unsigned getResultRegisterUnits(Operation *op) {
  Type type = op->getResult(0).getType();
  unsigned bitWidth = 32;
  if (auto integer = dyn_cast<IntegerType>(type)) {
    bitWidth = integer.getWidth();
  } else if (isa<IndexType>(type)) {
    bitWidth = 64;
  }
  return std::max(1U, (bitWidth + 31) / 32);
}

// A hoisted result remains live across every structured branch between its
// definition and the last occurrence. Estimate one register-width unit per
// four conditional regions; the estimate is used as a hard safety bound and a
// candidate tie-breaker because backend register pressure has cliff behavior
// that is not well represented by a linear arithmetic cost.
static unsigned getLiveRangePenalty(ArrayRef<Operation *> operations,
                                    Operation *insertionPoint) {
  Block *block = insertionPoint->getBlock();
  Operation *lastAnchor = insertionPoint;
  for (Operation *op : operations) {
    Operation *anchor = getAnchorInBlock(op, block);
    if (!anchor) {
      return kMaxClonedExpressionOps;
    }
    if (lastAnchor->isBeforeInBlock(anchor)) {
      lastAnchor = anchor;
    }
  }

  unsigned conditionalRegions = 0;
  for (Operation *cursor = insertionPoint; cursor;
       cursor = cursor->getNextNode()) {
    if (isa<scf::IfOp>(cursor)) {
      ++conditionalRegions;
    }
    if (cursor == lastAnchor) {
      break;
    }
  }
  if (conditionalRegions == 0) {
    return 0;
  }
  unsigned spanUnits =
      (conditionalRegions + kConditionalRegionsPerLiveRangeUnit - 1) /
      kConditionalRegionsPerLiveRangeUnit;
  return spanUnits * getResultRegisterUnits(operations.front());
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
  uint64_t estimatedBenefit = 0;
  unsigned cloneCost = 0;
  unsigned liveRangePenalty = 0;
  std::optional<PredicateLiteral> guard;
};

static bool isBetterCandidate(const HoistCandidate &candidate,
                              const std::optional<HoistCandidate> &best) {
  if (!best || candidate.estimatedBenefit != best->estimatedBenefit) {
    return !best || candidate.estimatedBenefit > best->estimatedBenefit;
  }
  if (candidate.liveRangePenalty != best->liveRangePenalty) {
    return candidate.liveRangePenalty < best->liveRangePenalty;
  }
  if (candidate.guard.has_value() != best->guard.has_value()) {
    return candidate.guard.has_value();
  }
  return candidate.cloneCost < best->cloneCost;
}

static bool sameOccurrences(ArrayRef<Operation *> lhs,
                            ArrayRef<Operation *> rhs) {
  return lhs.size() == rhs.size() && llvm::equal(lhs, rhs);
}

static SmallVector<SmallVector<Operation *>>
collectCandidateOccurrenceSets(const ExpressionGroup &group) {
  SmallVector<SmallVector<Operation *>> sets;
  sets.push_back(group.operations);
  for (Operation *op : group.operations) {
    for (const PredicateLiteral &literal : getControlPredicates(op)) {
      SmallVector<Operation *> subset;
      for (Operation *candidate : group.operations) {
        std::optional<bool> value = findPredicateValue(
            literal.predicate, getControlPredicates(candidate));
        if (value && *value == literal.value) {
          subset.push_back(candidate);
        }
      }
      if (subset.size() < 2 ||
          llvm::any_of(sets, [&](ArrayRef<Operation *> existing) {
            return sameOccurrences(subset, existing);
          })) {
        continue;
      }
      sets.push_back(std::move(subset));
    }
  }
  return sets;
}

static void considerCandidateSet(ArrayRef<Operation *> operations,
                                 func::FuncOp func, DominanceInfo &dominance,
                                 std::optional<HoistCandidate> &best) {
  if (operations.size() < 2 || hasDominatingOccurrence(operations, dominance) ||
      !hasCoExecutablePair(operations)) {
    return;
  }

  SmallVector<Operation *> barriers =
      getExecutionBarriers(operations.front(), func);
  if (!llvm::all_of(operations.drop_front(), [&](Operation *op) {
        return getExecutionBarriers(op, func) == barriers;
      })) {
    return;
  }

  Operation *insertionPoint = findInsertionPoint(operations, func);
  if (!insertionPoint) {
    return;
  }
  DenseMap<Value, bool> availabilityMemo;
  if (!canMaterializeBefore(operations.front()->getResult(0), insertionPoint,
                            dominance, availabilityMemo)) {
    return;
  }

  SmallPtrSet<Operation *, 32> cloneOps;
  if (!collectCloneOps(operations.front()->getResult(0), insertionPoint,
                       dominance, cloneOps)) {
    return;
  }
  SmallPtrSet<Operation *, 32> removable = findRemovableOps(operations);
  unsigned cloneCost = getWeightedCost(cloneOps);
  unsigned removableCost = getWeightedCost(removable);
  if (removableCost < cloneCost + kMinimumWeightedSaving) {
    return;
  }

  std::optional<ConditionalCostEstimate> conditionalCost =
      estimateConditionalCost(removable, insertionPoint);
  if (!conditionalCost) {
    return;
  }
  unsigned speculationPenalty = cloneCost > conditionalCost->minimumCost
                                    ? cloneCost - conditionalCost->minimumCost
                                    : 0;
  unsigned liveRangePenalty = getLiveRangePenalty(operations, insertionPoint);
  LLVM_DEBUG({
    llvm::dbgs() << "scalar-cse candidate root="
                 << operations.front()->getName()
                 << " occurrences=" << operations.size()
                 << " clone=" << cloneCost << " removable=" << removableCost
                 << " expected="
                 << static_cast<double>(conditionalCost->expectedCostScaled) /
                        kCostScale
                 << " minimum=" << conditionalCost->minimumCost
                 << " live=" << liveRangePenalty << "\n";
  });
  // expectedCostScaled already compares the always-executed clone against the
  // probability-weighted old work. Add a smaller tail-risk premium. A short
  // fanout still pays for its live range because one or two saved copies rarely
  // offset backend register-pressure effects. Four or more occurrences may
  // amortize that linear charge through their expected arithmetic saving, but
  // every candidate remains subject to the hard live-range bound below.
  uint64_t liveRangeCost =
      operations.size() < kMinimumOccurrencesToAmortizeLiveRange
          ? static_cast<uint64_t>(liveRangePenalty) * kCostScale /
                kLiveRangeRiskDivisor
          : 0;
  uint64_t unguardedCost = static_cast<uint64_t>(cloneCost) * kCostScale +
                           static_cast<uint64_t>(speculationPenalty) *
                               kCostScale / kSpeculationRiskDivisor +
                           liveRangeCost;
  uint64_t unguardedMinimumSaving =
      static_cast<uint64_t>(kMinimumExpectedWeightedSaving) * kCostScale;
  unsigned maximumLiveRangePenalty = kMaxUnguardedLiveRangeSpanUnits *
                                     getResultRegisterUnits(operations.front());
  if (liveRangePenalty <= maximumLiveRangePenalty &&
      conditionalCost->expectedCostScaled >
          unguardedCost + unguardedMinimumSaving) {
    HoistCandidate candidate{SmallVector<Operation *>(operations),
                             insertionPoint,
                             conditionalCost->expectedCostScaled -
                                 unguardedCost,
                             cloneCost,
                             liveRangePenalty,
                             std::nullopt};
    if (isBetterCandidate(candidate, best)) {
      best = std::move(candidate);
    }
  }

  std::optional<PredicateLiteral> commonGuard =
      findCommonDominatingPredicate(operations, insertionPoint, dominance);
  if (!commonGuard) {
    LLVM_DEBUG(llvm::dbgs() << "  no common dominating predicate\n");
    return;
  }
  std::optional<ConditionalCostEstimate> guardedConditionalCost =
      estimateConditionalCost(removable, insertionPoint,
                              ArrayRef<PredicateLiteral>(*commonGuard));
  if (!guardedConditionalCost) {
    return;
  }
  unsigned guardedSpeculationPenalty =
      cloneCost > guardedConditionalCost->minimumCost
          ? cloneCost - guardedConditionalCost->minimumCost
          : 0;
  unsigned guardOverhead = kGuardControlFlowCost + !commonGuard->value;
  LLVM_DEBUG({
    llvm::dbgs() << "  guarded minimum=" << guardedConditionalCost->minimumCost
                 << " speculation=" << guardedSpeculationPenalty
                 << " overhead=" << guardOverhead << "\n";
  });
  // The cloned DAG only executes when the common guard is true, while the
  // synthetic branch and its result's live range exist on every path. Scale
  // both the cloned work and the required saving by the guard probability,
  // but keep the control-flow and register-pressure charges unconditional.
  uint64_t guardedCost =
      static_cast<uint64_t>(cloneCost + guardedSpeculationPenalty) *
          (kCostScale / 2) +
      static_cast<uint64_t>(guardOverhead + liveRangePenalty) * kCostScale;
  uint64_t guardedMinimumSaving =
      static_cast<uint64_t>(kMinimumWeightedSaving) * (kCostScale / 2);
  if (conditionalCost->expectedCostScaled >=
      guardedCost + guardedMinimumSaving) {
    HoistCandidate candidate{SmallVector<Operation *>(operations),
                             insertionPoint,
                             conditionalCost->expectedCostScaled - guardedCost,
                             cloneCost,
                             liveRangePenalty,
                             commonGuard};
    if (isBetterCandidate(candidate, best)) {
      best = std::move(candidate);
    }
  }
}

static std::optional<HoistCandidate>
findBestCandidate(func::FuncOp func, DominanceInfo &dominance) {
  std::optional<HoistCandidate> best;
  for (const ExpressionGroup &group : collectExpressionGroups(func)) {
    for (const SmallVector<Operation *> &operations :
         collectCandidateOccurrenceSets(group)) {
      considerCandidateSet(operations, func, dominance, best);
    }
  }
  return best;
}

static Value materializeAt(Value value, Operation *availabilityPoint,
                           DominanceInfo &dominance, OpBuilder &builder,
                           DenseMap<Value, Value> &memo) {
  if (dominance.dominates(value, availabilityPoint)) {
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
        materializeAt(operand, availabilityPoint, dominance, builder, memo));
  }
  Operation *clone = builder.clone(*def);
  clone->setOperands(operands);
  Value result = clone->getResult(0);
  memo.try_emplace(value, result);
  return result;
}

static Value materializeGuarded(const HoistCandidate &candidate,
                                DominanceInfo &dominance, OpBuilder &builder,
                                DenseMap<Value, Value> &memo) {
  Location location = candidate.occurrences.front()->getLoc();
  Value condition = candidate.guard->predicate;
  if (!candidate.guard->value) {
    Value trueValue =
        builder.create<arith::ConstantOp>(location, builder.getBoolAttr(true));
    condition = builder.create<arith::XOrIOp>(location, condition, trueValue);
  }

  Type resultType = candidate.occurrences.front()->getResult(0).getType();
  auto ifOp =
      builder.create<scf::IfOp>(location, TypeRange{resultType}, condition,
                                /*withElseRegion=*/true);
  {
    OpBuilder::InsertionGuard insertionGuard(builder);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    Value materialized =
        materializeAt(candidate.occurrences.front()->getResult(0),
                      candidate.insertionPoint, dominance, builder, memo);
    builder.create<scf::YieldOp>(location, materialized);

    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    IntegerAttr zero = builder.getIntegerAttr(resultType, 0);
    Value zeroValue =
        builder.create<arith::ConstantOp>(location, resultType, zero);
    builder.create<scf::YieldOp>(location, zeroValue);
  }
  return ifOp.getResult(0);
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
      Value hoisted;
      if (candidate->guard) {
        hoisted =
            materializeGuarded(*candidate, dominance, builder, materialized);
      } else {
        hoisted = materializeAt(candidate->occurrences.front()->getResult(0),
                                candidate->insertionPoint, dominance, builder,
                                materialized);
      }
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
