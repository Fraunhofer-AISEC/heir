#ifndef LIB_ANALYSIS_SCHEMEINFOANALYSIS_CGGISCHEMEINFO_H_
#define LIB_ANALYSIS_SCHEMEINFOANALYSIS_CGGISCHEMEINFO_H_

#include <algorithm>
#include <cassert>

#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {

class CGGISchemeInfo {
 public:
  CGGISchemeInfo()
      : initialized(true),
        level(0) {}

  explicit CGGISchemeInfo(int lvl)
      : initialized(true),
        level(lvl) {}

  int getLevel() const {
    assert(isInitialized() && "CGGISchemeInfo not initialized");
    return level;
  }

  bool isInitialized() const { return initialized; }

  bool operator==(const CGGISchemeInfo &rhs) const {
    return initialized == rhs.initialized &&
           level == rhs.level;
  }

  CGGISchemeInfo operator+(const CGGISchemeInfo &rhs) const {
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
    return CGGISchemeInfo(
        std::max(level, rhs.level)
    );
  }
 

  static CGGISchemeInfo max(const CGGISchemeInfo &lhs,
                                 const CGGISchemeInfo &rhs) {
    return CGGISchemeInfo(
    std::max(lhs.level, rhs.level)
    );
  }

  static CGGISchemeInfo join(const CGGISchemeInfo &lhs,
                                  const CGGISchemeInfo &rhs) {
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
      os << "CGGISchemeInfo(level=" << level
         << ")";
    } else {
      os << "CGGISchemeInfo(uninitialized)";
    }
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const CGGISchemeInfo &state) {
    state.print(os);
    return os;
  }
 private:
  bool initialized;
  int level;
};


class CGGISchemeInfoLattice : public dataflow::Lattice<CGGISchemeInfo> {
 public:
  using Lattice::Lattice;
};

class CGGISchemeInfoAnalysis : public dataflow::SparseForwardDataFlowAnalysis<CGGISchemeInfoLattice> {
 public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const CGGISchemeInfoLattice *> operands,
                               ArrayRef<CGGISchemeInfoLattice *> results) override;

  void setToEntryState(CGGISchemeInfoLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(CGGISchemeInfo()));
  }
};

int computeApproximateRuntimeCGGI(Operation *top, DataFlowSolver *solver) ;

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_SCHEMEINFOANALYSIS_CGGISCHEMEINFO_H_
