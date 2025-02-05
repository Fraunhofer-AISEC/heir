#ifndef LIB_DIALECT_MGMT_IR_MGMTATTRIBUTES_TD_
#define LIB_DIALECT_MGMT_IR_MGMTATTRIBUTES_TD_

include "lib/Dialect/Mgmt/IR/MgmtDialect.td"
include "mlir/IR/OpBase.td"

class Mgmt_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Mgmt_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def Mgmt_MgmtAttr : Mgmt_Attr<"Mgmt", "mgmt"> {
  let summary = "Container attribute for all mgmt parameter";
  let description = [{
    This attribute is used to store all mgmt parameters.

    The attribute is a struct with the following fields:
      - `level` : the level of the ciphertext, from L to 0
      - `dimension` : the dimension of the ciphertext, defaults to 2

    Internally, this attribute is used by secret-to-<scheme>
    for determining the level and dimension of the ciphertext.

    It should be populated by --secret-with-mgmt-<scheme> before
    going through the secret-to-<scheme> pass.

    Example:
    ```
    #mgmt = #mgmt.mgmt<level = 1> // dimension defaults to 2
    #mgmt1 = #mgmt.mgmt<level = 1, dimension = 3>
    %0 = secret.generic ins(%arg0, %arg1 : !secret.secret<i16>) attrs = {mgmt.mgmt = #mgmt} {
       ...
    } -> !secret.secret<i16>
    ```
  }];

  let parameters = (ins
    "int": $level,
    DefaultValuedParameter<"int", "2">:$dimension
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Mgmt_OpenfheParamsAttr : Mgmt_Attr<"OpenfheParams", "openfhe_params"> {
  let summary = "Container attribute for some OpenFHE-specific parameters";
  let description = [{
    This attribute is used to store some OpenFHE-specific parameters.

    The attribute is a struct with the following fields:
      - `multiplicativeDepth` : the multiplicative depth
      - `ringDimension` : the ring dimension
      - `firstModeSize` : the first modulus size
      - `scalingModSize` : the scaling modulus size

    When this attribute presents, the lowering of openfhe pass
    will use these parameters to set the corresponding OpenFHE
    parameters.

    Example:
    ```
    #openfhe_params = #mgmt.openfhe_params<firstModSize = 20, multiplicativeDepth = 3, scalingModSize = 38>
    ```
  }];

  let parameters = (ins
    "int": $multiplicativeDepth,
    "int": $ringDimension,
    "int": $scalingModSize,
    "int": $firstModSize
  );
  let assemblyFormat = "`<` struct(params) `>`";
}


#endif  // LIB_DIALECT_MGMT_IR_MGMTATTRIBUTES_TD_
