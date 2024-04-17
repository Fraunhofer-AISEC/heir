#include "include/Target/OpenFhePke/OpenFhePkeHeaderEmitter.h"

#include "include/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "include/Dialect/LWE/IR/LWEDialect.h"
#include "include/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "include/Dialect/Polynomial/IR/PolynomialDialect.h"
#include "include/Target/OpenFhePke/OpenFheUtils.h"
#include "lib/Target/OpenFhePke/OpenFhePkeTemplates.h"
#include "lib/Target/Utils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"           // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"   // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

void registerToOpenFhePkeHeaderTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-openfhe-pke-header",
      "Emit a header corresponding to the C++ file generated by "
      "--emit-openfhe-pke",
      [](Operation *op, llvm::raw_ostream &output) {
        return translateToOpenFhePkeHeader(op, output);
      },
      [](DialectRegistry &registry) {
        registry.insert<arith::ArithDialect, func::FuncDialect,
                        openfhe::OpenfheDialect, lwe::LWEDialect,
                        ::mlir::heir::polynomial::PolynomialDialect>();
      });
}

LogicalResult translateToOpenFhePkeHeader(Operation *op,
                                          llvm::raw_ostream &os) {
  SelectVariableNames variableNames(op);
  OpenFhePkeHeaderEmitter emitter(os, &variableNames);
  return emitter.translate(*op);
}

LogicalResult OpenFhePkeHeaderEmitter::translate(Operation &op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation &, LogicalResult>(op)
          .Case<ModuleOp>([&](auto op) { return printOperation(op); })
          .Case<func::FuncOp>([&](auto op) { return printOperation(op); })
          .Default([&](Operation &) {
            return op.emitOpError("unable to find printer for op");
          });

  if (failed(status)) {
    op.emitOpError(llvm::formatv("Failed to translate op {0}", op.getName()));
    return failure();
  }
  return success();
}

LogicalResult OpenFhePkeHeaderEmitter::printOperation(ModuleOp moduleOp) {
  os << kModulePrelude << "\n";
  for (Operation &op : moduleOp) {
    if (failed(translate(op))) {
      return failure();
    }
  }
  return success();
}

LogicalResult OpenFhePkeHeaderEmitter::printOperation(func::FuncOp funcOp) {
  // If keeping this consistent alongside OpenFheEmitter gets annoying,
  // extract to a shared function in a base class.
  if (funcOp.getNumResults() != 1) {
    return funcOp.emitOpError() << "Only functions with a single return type "
                                   "are supported, but this function has "
                                << funcOp.getNumResults();
    return failure();
  }

  Type result = funcOp.getResultTypes()[0];
  if (failed(emitType(result))) {
    return funcOp.emitOpError() << "Failed to emit type " << result;
  }

  os << " " << funcOp.getName() << "(";

  for (Value arg : funcOp.getArguments()) {
    if (failed(convertType(arg.getType()))) {
      return funcOp.emitOpError() << "Failed to emit type " << arg.getType();
    }
  }

  os << commaSeparatedValues(funcOp.getArguments(), [&](Value value) {
    auto res = convertType(value.getType());
    return res.value() + " " + variableNames->getNameForValue(value);
  });
  os << ");\n";

  return success();
}

LogicalResult OpenFhePkeHeaderEmitter::emitType(Type type) {
  auto result = convertType(type);
  if (failed(result)) {
    return failure();
  }
  os << result;
  return success();
}

OpenFhePkeHeaderEmitter::OpenFhePkeHeaderEmitter(
    raw_ostream &os, SelectVariableNames *variableNames)
    : os(os), variableNames(variableNames) {}
}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
