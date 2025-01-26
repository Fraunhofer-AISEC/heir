#ifndef LIB_ANALYSIS_OPERATIONCOUNT_OPERATIONCOUNTANALYSIS_H_
#define LIB_ANALYSIS_OPERATIONCOUNT_OPERATIONCOUNTANALYSIS_H_


#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"        // from @llvm-project

namespace mlir {
namespace heir {

class OperationCount {
 public:
  OperationCount() : initialized(false), addCount(0), keySwitchCount(0) {}
  explicit OperationCount(int addCount, int keySwitchCount)
      : initialized(true), addCount(addCount), keySwitchCount(keySwitchCount) {}
  
  void countOperation(Operation *op);

  int getAddCount() const {
    assert(isInitialized() && "OperationCount not initialized");
    return addCount;
  }

  int getKeySwitchCount() const {
    assert(isInitialized() && "OperationCount not initialized");
    return keySwitchCount;
  }

  OperationCount incrementKeySwitch() const {
    assert(isInitialized() && "OperationCount not initialized");
    return OperationCount(addCount, keySwitchCount + 1);
  }

  bool isInitialized() const { return initialized; }

  bool operator==(const OperationCount &rhs) const {
    return initialized == rhs.initialized && addCount == rhs.addCount &&
           keySwitchCount == rhs.keySwitchCount;
  }

  OperationCount operator+(const OperationCount &rhs) const {
    assert(isInitialized() && rhs.isInitialized() &&
           "OperationCount not initialized");
    return OperationCount(addCount + rhs.addCount,
                          keySwitchCount + rhs.keySwitchCount);
  }

  OperationCount max(const OperationCount &rhs) const {
    assert(isInitialized() && rhs.isInitialized() &&
           "OperationCount not initialized");
    return OperationCount(std::max(addCount, rhs.addCount),
                          std::max(keySwitchCount, rhs.keySwitchCount));
  }

  static OperationCount join(const OperationCount &lhs,
                             const OperationCount &rhs) {
    OperationCount result;

    if (!lhs.isInitialized()) {
      return rhs;
    }

    if (!rhs.isInitialized()) {
      return lhs;
    }

    return lhs.max(rhs);

    return result;
  }

  void print(llvm::raw_ostream &os) const {
    if (isInitialized()) {
      os << "OperationCount(addCount=" << addCount
         << ", keySwitchCount=" << keySwitchCount << ")";
    } else {
      os << "OperationCount(uninitialized)";
    }
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const OperationCount &count) {
    count.print(os);
    return os;
  }

 private:
  bool initialized;
  int addCount;
  int keySwitchCount;
};

class OperationCountLattice : public dataflow::Lattice<OperationCount> {
 public:
  using Lattice::Lattice;
};

class OperationCountAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<OperationCountLattice>,
      public SecretnessAnalysisDependent<OperationCountAnalysis> {
 public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;
  friend class SecretnessAnalysisDependent<OperationCountAnalysis>;

  LogicalResult visitOperation(
      Operation *op, ArrayRef<const OperationCountLattice *> operands,
      ArrayRef<OperationCountLattice *> results) override;

  void setToEntryState(OperationCountLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(OperationCount()));
  }
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_OPERATIONCOUNT_OPERATIONCOUNTANALYSIS_H_