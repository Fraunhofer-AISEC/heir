#ifndef LIB_ANALYSIS_SCHEMEINFOANALYSIS_CKKSSCHEMEINFO_H_
#define LIB_ANALYSIS_SCHEMEINFOANALYSIS_CKKSSCHEMEINFO_H_

#include <algorithm>
#include <cassert>
#include <optional>

#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {

// Minimal CKKS scheme info: track multiplicative depth level only.
class CKKSSchemeInfo {
 public:
  CKKSSchemeInfo()
      : initialized(true),
        level(0) {}

  explicit CKKSSchemeInfo(int lvl)
      : initialized(true),
        level(lvl) {}

  int getLevel() const {
    assert(isInitialized() && "CKKSSchemeInfo not initialized");
    return level;
  }

  bool isInitialized() const { return initialized; }

  bool operator==(const CKKSSchemeInfo &rhs) const {
    return initialized == rhs.initialized &&
           level == rhs.level;
  }

  CKKSSchemeInfo operator+(const CKKSSchemeInfo &rhs) const {
    if (!isInitialized() && !rhs.isInitialized()) {
      return *this;  // return the current object
    }

    if (isInitialized() && !rhs.isInitialized()) {
      return *this;  // return the current object
    }

    if (!isInitialized() && rhs.isInitialized()) {
      return rhs;  // return the rhs object
    }

    // Both are initialized
    return CKKSSchemeInfo(
        std::max(level, rhs.level)
    );
  }

  static CKKSSchemeInfo max(const CKKSSchemeInfo &lhs,
                            const CKKSSchemeInfo &rhs) {
    return CKKSSchemeInfo(
        std::max(lhs.level, rhs.level)
    );
  }

  static CKKSSchemeInfo join(const CKKSSchemeInfo &lhs,
                             const CKKSSchemeInfo &rhs) {
    if (!lhs.isInitialized()) {
      return rhs;
    }

    if (!rhs.isInitialized()) {
      return lhs;
    }

    return max(lhs, rhs);
  }

  void print(llvm::raw_ostream &os) const {
    if (isInitialized()) {
      os << "CKKSSchemeInfo(level=" << level
         << ")";
    } else {
      os << "CKKSSchemeInfo(uninitialized)";
    }
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const CKKSSchemeInfo &state) {
    state.print(os);
    return os;
  }

 private:
  bool initialized;
  int level;
};


class CKKSSchemeInfoLattice : public dataflow::Lattice<CKKSSchemeInfo> {
 public:
  using Lattice::Lattice;
};

class CKKSSchemeInfoAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<CKKSSchemeInfoLattice> {
 public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const CKKSSchemeInfoLattice *> operands,
                               ArrayRef<CKKSSchemeInfoLattice *> results) override;

  void setToEntryState(CKKSSchemeInfoLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(CKKSSchemeInfo()));
  }
};

// Runtime estimation entry for CKKS.
int computeApproximateRuntimeCKKS(Operation *top, DataFlowSolver *solver);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_SCHEMEINFOANALYSIS_CKKSSCHEMEINFO_H_