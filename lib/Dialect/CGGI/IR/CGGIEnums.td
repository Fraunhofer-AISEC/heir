#ifndef LIB_DIALECT_CGGI_IR_CGGIENUMS_TD_
#define LIB_DIALECT_CGGI_IR_CGGIENUMS_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinDialect.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/EnumAttr.td"

include "lib/Dialect/CGGI/IR/BooleanGates.td"

def BOOL_GATE_AND: I32EnumAttrCase<"AND", AND_GATE>;
def BOOL_GATE_NAND: I32EnumAttrCase<"NAND", NAND_GATE>;
def BOOL_GATE_OR: I32EnumAttrCase<"OR", OR_GATE>;
def BOOL_GATE_NOR: I32EnumAttrCase<"NOR", NOR_GATE>;
def BOOL_GATE_XOR: I32EnumAttrCase<"XOR", XOR_GATE>;
def BOOL_GATE_XNOR: I32EnumAttrCase<"XNOR", XNOR_GATE>;
def BOOL_GATE_NOT: I32EnumAttrCase<"NOT", NOT_GATE>;

def CGGI_BooleanGateEnumAttr : I32EnumAttr<"CGGIBoolGateEnum",
    "An enum attribute representing a CGGI boolean gate using u8 int",
    [ BOOL_GATE_AND, BOOL_GATE_NAND, BOOL_GATE_OR,
      BOOL_GATE_NOR, BOOL_GATE_XOR, BOOL_GATE_XNOR,
      BOOL_GATE_NOT
    ]>
{
  let cppNamespace = "::mlir::heir::cggi";

}

#endif  // LIB_DIALECT_CGGI_IR_CGGIENUMS_TD_
