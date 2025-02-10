#ifndef LIB_ANALYSIS_OPERATIONCOUNT_OPERATIONCOUNTANALYSIS_H_
#define LIB_ANALYSIS_OPERATIONCOUNT_OPERATIONCOUNTANALYSIS_H_


#include <cstdint>
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"        // from @llvm-project

namespace mlir {
namespace heir {

class OperationCount {
 public:
  OperationCount() : initialized(false), ciphertextCount(0), keySwitchCount(0) {}
  explicit OperationCount(int ciphertextCount, int keySwitchCount)
      : initialized(true), ciphertextCount(ciphertextCount), keySwitchCount(keySwitchCount) {}
  
  void countOperation(Operation *op);

  int getCiphertextCount() const {
    assert(isInitialized() && "OperationCount not initialized");
    return ciphertextCount;
  }

  int getKeySwitchCount() const {
    assert(isInitialized() && "OperationCount not initialized");
    return keySwitchCount;
  }

  OperationCount incrementKeySwitch() const {
    assert(isInitialized() && "OperationCount not initialized");
    return OperationCount(ciphertextCount, keySwitchCount + 1);
  }

  bool isInitialized() const { return initialized; }

  bool operator==(const OperationCount &rhs) const {
    return initialized == rhs.initialized && ciphertextCount == rhs.ciphertextCount &&
           keySwitchCount == rhs.keySwitchCount;
  }

  OperationCount operator+(const OperationCount &rhs) const {
    assert(isInitialized() && rhs.isInitialized() &&
           "OperationCount not initialized");
    return OperationCount(ciphertextCount + rhs.ciphertextCount,
                          keySwitchCount + rhs.keySwitchCount);
  }

  static OperationCount max(const OperationCount &lhs, const OperationCount &rhs) {
    assert(lhs.isInitialized() && rhs.isInitialized() &&
           "OperationCount not initialized");
    return OperationCount(std::max(lhs.ciphertextCount, rhs.ciphertextCount),
                          std::max(lhs.keySwitchCount, rhs.keySwitchCount));
  }

  static OperationCount join(const OperationCount &lhs,
                             const OperationCount &rhs) {
    if (!lhs.isInitialized()) {
      return rhs;
    }

    if (!rhs.isInitialized()) {
      return lhs;
    }

    return OperationCount::max(lhs, rhs);
  }

  void print(llvm::raw_ostream &os) const {
    if (isInitialized()) {
      os << "OperationCount(ciphertextCount=" << ciphertextCount
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
  // "Number of ciphertexts with base noise" that flow into the respective value
  int ciphertextCount;
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

void annotateCountParams(Operation *top, DataFlowSolver *solver,
                         int ringDimension, int plaintextModulus);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_OPERATIONCOUNT_OPERATIONCOUNTANALYSIS_H_