#ifndef LIB_DIALECT_RNS_IR_RNSTYPEINTERFACES_H_
#define LIB_DIALECT_RNS_IR_RNSTYPEINTERFACES_H_

#include "lib/Dialect/RNS/IR/RNSDialect.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/RNS/IR/RNSTypeInterfaces.h.inc"

namespace mlir {

class DialectRegistry;

namespace heir {
namespace rns {

void registerExternalRNSTypeInterfaces(DialectRegistry &registry);

}  // namespace rns
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_RNS_IR_RNSTYPEINTERFACES_H_
