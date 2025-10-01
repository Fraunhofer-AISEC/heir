#ifndef LIB_ANALYSIS_SCHEMEINFOANALYSIS_BGVSCHEMEINFO_H_
#define LIB_ANALYSIS_SCHEMEINFOANALYSIS_BGVSCHEMEINFO_H_

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

class BGVSchemeInfo {
 public:
  BGVSchemeInfo()
      : initialized(true),
        level(0) {}

  explicit BGVSchemeInfo(int lvl)
      : initialized(true),
        level(lvl) {}

  int getLevel() const {
    assert(isInitialized() && "BGVSchemeInfo not initialized");
    return level;
  }

  bool isInitialized() const { return initialized; }

  bool operator==(const BGVSchemeInfo &rhs) const {
    return initialized == rhs.initialized &&
           level == rhs.level;
  }

  BGVSchemeInfo operator+(const BGVSchemeInfo &rhs) const {
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
    return BGVSchemeInfo(
        std::max(level, rhs.level)
    );
  }
 

  static BGVSchemeInfo max(const BGVSchemeInfo &lhs,
                                 const BGVSchemeInfo &rhs) {
    return BGVSchemeInfo(
    std::max(lhs.level, rhs.level)
    );
  }

  static BGVSchemeInfo join(const BGVSchemeInfo &lhs,
                                  const BGVSchemeInfo &rhs) {
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
      os << "BGVSchemeInfo(level=" << level
         << ")";
    } else {
      os << "BGVSchemeInfo(uninitialized)";
    }
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const BGVSchemeInfo &state) {
    state.print(os);
    return os;
  }
 private:
  bool initialized;
  int level;
};


class BGVSchemeInfoLattice : public dataflow::Lattice<BGVSchemeInfo> {
 public:
  using Lattice::Lattice;
};

class BGVSchemeInfoAnalysis : public dataflow::SparseForwardDataFlowAnalysis<BGVSchemeInfoLattice> {
 public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const BGVSchemeInfoLattice *> operands,
                               ArrayRef<BGVSchemeInfoLattice *> results) override;

  void setToEntryState(BGVSchemeInfoLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(BGVSchemeInfo()));
  }
};

int computeApproximateRuntimeBGV(Operation *top, DataFlowSolver *solver) ;

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_SCHEMEINFOANALYSIS_BGVSCHEMEINFO_H_
