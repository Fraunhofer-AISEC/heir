#ifndef LIB_TRANSFORMS_ANNOTATEPARAMETERS_ANNOTATEPARAMETERS_TD_
#define LIB_TRANSFORMS_ANNOTATEPARAMETERS_ANNOTATEPARAMETERS_TD_

include "mlir/Pass/PassBase.td"

def AnnotateParameters : Pass<"annotate-parameters"> {
  let summary = "Annotate parameters in the IR";
  let description = [{
  This pass runs the parameter analysis and annotates the IR with the results,
  saving it into the function's attribute dictionary as parameter.
  }];
  let options = [
    Option<"ringDimension", "ring-dimension", "int", "0", "Specify the ring dimension for the parameters">,
    Option<"plaintextModulus", "plaintext-modulus", "int", "0", "Specify the plaintext modulus for the parameters">,
    Option<"algorithm", "algorithm", "std::string", "\"OpenFHE\"", "Specify the algorithm to determine the moduli">
  ];
}

#endif  // LIB_TRANSFORMS_ANNOTATEPARAMETERS_ANNOTATEPARAMETERS_TD_
