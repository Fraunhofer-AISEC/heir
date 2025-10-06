#include "lib/Transforms/AnnotateSchemeInfo/AnnotateSchemeInfo.h"

#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/MulDepthAnalysis/MulDepthAnalysis.h"
#include "lib/Analysis/SchemeInfoAnalysis/BGV/BGVSchemeInfo.h"
#include "lib/Analysis/SchemeInfoAnalysis/CGGI/CGGISchemeInfo.h"
#include "lib/Analysis/SchemeInfoAnalysis/SchemeInfoAnalysis.h"
#include "lib/Analysis/SchemeSelectionAnalysis/SchemeSelectionAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Transforms/AnnotateModule/AnnotateModule.h"
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"              // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {

#define DEBUG_TYPE "AnnotateSchemeInfo"

#define GEN_PASS_DEF_ANNOTATESCHEMEINFO
#include "lib/Transforms/AnnotateSchemeInfo/AnnotateSchemeInfo.h.inc"

struct AnnotateSchemeInfo : impl::AnnotateSchemeInfoBase<AnnotateSchemeInfo> {
  using AnnotateSchemeInfoBase::AnnotateSchemeInfoBase;

  void runOnOperation() override {
    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<SecretnessAnalysis>();

    // Perform Analysis for all available schemes
    solver.load<BGVSchemeInfoAnalysis>();
	solver.load<CGGISchemeInfoAnalysis>();

    auto result = solver.initializeAndRun(getOperation());

    if (failed(result)) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    auto runtimeBGV = computeApproximateRuntimeBGV(getOperation(), &solver);
    LLVM_DEBUG(llvm::dbgs() << "Approximate runtime for BGV: " << runtimeBGV << "ms.\n");

	auto runtimeCGGI = computeApproximateRuntimeCGGI(getOperation(), &solver);
    LLVM_DEBUG(llvm::dbgs() << "Approximate runtime for CGGI: " << runtimeBGV << "ms.\n");

    // Insert comparison between schemes here
    std::string scheme;
	if (runtimeBGV < runtimeCGGI) {
		scheme = BGV;
	} else {
		scheme = CGGI;
	}

    OpPassManager pipeline("builtin.module");
    AnnotateModuleOptions annotateModuleOptions;
    annotateModuleOptions.scheme = scheme;
    pipeline.addPass(createAnnotateModule(annotateModuleOptions));

    (void)runPipeline(pipeline, getOperation());
  }
};
}  // namespace heir
}  // namespace mlir
