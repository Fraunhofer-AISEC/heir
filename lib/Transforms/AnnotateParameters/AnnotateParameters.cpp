#include "lib/Transforms/AnnotateParameters/AnnotateParameters.h"

#include <utility>

#include "lib/Analysis/OperationCountAnalysis/OperationCountAnalysis.h"
#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"           // from @llvm-project


namespace mlir {
namespace heir {

#define GEN_PASS_DEF_ANNOTATEPARAMETERS
#include "lib/Transforms/AnnotateParameters/AnnotateParameters.h.inc"

struct AnnotateParameters : impl::AnnotateParametersBase<AnnotateParameters> {
  using AnnotateParametersBase::AnnotateParametersBase;

  void runOnOperation() override {
    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<SecretnessAnalysis>();
    solver.load<LevelAnalysis>();
    solver.load<OperationCountAnalysis>();

    auto result = solver.initializeAndRun(getOperation());

    if (failed(result)) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    annotateCountParams(getOperation(), &solver, ringDimension, plaintextModulus, algorithm);
  }
};

}  // namespace heir
}  // namespace mlir
