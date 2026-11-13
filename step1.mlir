module attributes {scheme.bgv} {
  func.func @func(%arg0: !secret.secret<tensor<8xi16>>, %arg1: !secret.secret<tensor<8xi16>>, %arg2: !secret.secret<tensor<8xi16>>, %arg3: !secret.secret<tensor<8xi16>>, %arg4: !secret.secret<tensor<8xi16>>, %arg5: !secret.secret<tensor<8xi16>>, %arg6: !secret.secret<tensor<8xi16>>, %arg7: !secret.secret<tensor<8xi16>>, %arg8: !secret.secret<tensor<8xi16>>, %arg9: !secret.secret<tensor<8xi16>>, %arg10: !secret.secret<tensor<8xi16>>, %arg11: !secret.secret<tensor<8xi16>>, %arg12: !secret.secret<tensor<8xi16>>, %arg13: !secret.secret<tensor<8xi16>>, %arg14: !secret.secret<tensor<8xi16>>, %arg15: !secret.secret<tensor<8xi16>>, %arg16: !secret.secret<tensor<8xi16>>, %arg17: !secret.secret<tensor<8xi16>>, %arg18: !secret.secret<tensor<8xi16>>, %arg19: !secret.secret<tensor<8xi16>>, %arg20: !secret.secret<tensor<8xi16>>, %arg21: !secret.secret<tensor<8xi16>>, %arg22: !secret.secret<tensor<8xi16>>, %arg23: !secret.secret<tensor<8xi16>>, %arg24: !secret.secret<tensor<8xi16>>, %arg25: !secret.secret<tensor<8xi16>>, %arg26: !secret.secret<tensor<8xi16>>, %arg27: !secret.secret<tensor<8xi16>>, %arg28: !secret.secret<tensor<8xi16>>, %arg29: !secret.secret<tensor<8xi16>>, %arg30: !secret.secret<tensor<8xi16>>, %arg31: !secret.secret<tensor<8xi16>>, %arg32: !secret.secret<tensor<8xi16>>, %arg33: !secret.secret<tensor<8xi16>>, %arg34: !secret.secret<tensor<8xi16>>, %arg35: !secret.secret<tensor<8xi16>>, %arg36: !secret.secret<tensor<8xi16>>, %arg37: !secret.secret<tensor<8xi16>>, %arg38: !secret.secret<tensor<8xi16>>, %arg39: !secret.secret<tensor<8xi16>>, %arg40: !secret.secret<tensor<8xi16>>, %arg41: !secret.secret<tensor<8xi16>>, %arg42: !secret.secret<tensor<8xi16>>, %arg43: !secret.secret<tensor<8xi16>>, %arg44: !secret.secret<tensor<8xi16>>, %arg45: !secret.secret<tensor<8xi16>>, %arg46: !secret.secret<tensor<8xi16>>, %arg47: !secret.secret<tensor<8xi16>>, %arg48: !secret.secret<tensor<8xi16>>, %arg49: !secret.secret<tensor<8xi16>>, %arg50: !secret.secret<tensor<8xi16>>, %arg51: !secret.secret<tensor<8xi16>>, %arg52: !secret.secret<tensor<8xi16>>, %arg53: !secret.secret<tensor<8xi16>>, %arg54: !secret.secret<tensor<8xi16>>, %arg55: !secret.secret<tensor<8xi16>>, %arg56: !secret.secret<tensor<8xi16>>, %arg57: !secret.secret<tensor<8xi16>>, %arg58: !secret.secret<tensor<8xi16>>, %arg59: !secret.secret<tensor<8xi16>>, %arg60: !secret.secret<tensor<8xi16>>, %arg61: !secret.secret<tensor<8xi16>>, %arg62: !secret.secret<tensor<8xi16>>, %arg63: !secret.secret<tensor<8xi16>>) -> !secret.secret<tensor<8xi16>> {
    %0 = secret.generic ins(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg30, %arg31 : !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>) attrs = {__argattrs = [{mgmt.mgmt = #mgmt.mgmt<level = 4>}, {mgmt.mgmt = #mgmt.mgmt<level = 4>}, {mgmt.mgmt = #mgmt.mgmt<level = 4>}, {mgmt.mgmt = #mgmt.mgmt<level = 4>}, {mgmt.mgmt = #mgmt.mgmt<level = 4>}, {mgmt.mgmt = #mgmt.mgmt<level = 4>}, {mgmt.mgmt = #mgmt.mgmt<level = 4>}, {mgmt.mgmt = #mgmt.mgmt<level = 4>}, {mgmt.mgmt = #mgmt.mgmt<level = 4>}, {mgmt.mgmt = #mgmt.mgmt<level = 4>}, {mgmt.mgmt = #mgmt.mgmt<level = 4>}, {mgmt.mgmt = #mgmt.mgmt<level = 4>}, {mgmt.mgmt = #mgmt.mgmt<level = 4>}, {mgmt.mgmt = #mgmt.mgmt<level = 4>}, {mgmt.mgmt = #mgmt.mgmt<level = 4>}, {mgmt.mgmt = #mgmt.mgmt<level = 4>}, {mgmt.mgmt = #mgmt.mgmt<level = 4>}, {mgmt.mgmt = #mgmt.mgmt<level = 4>}, {mgmt.mgmt = #mgmt.mgmt<level = 4>}, {mgmt.mgmt = #mgmt.mgmt<level = 4>}, {mgmt.mgmt = #mgmt.mgmt<level = 4>}, {mgmt.mgmt = #mgmt.mgmt<level = 4>}, {mgmt.mgmt = #mgmt.mgmt<level = 4>}, {mgmt.mgmt = #mgmt.mgmt<level = 4>}, {mgmt.mgmt = #mgmt.mgmt<level = 4>}, {mgmt.mgmt = #mgmt.mgmt<level = 4>}, {mgmt.mgmt = #mgmt.mgmt<level = 4>}, {mgmt.mgmt = #mgmt.mgmt<level = 4>}, {mgmt.mgmt = #mgmt.mgmt<level = 4>}, {mgmt.mgmt = #mgmt.mgmt<level = 4>}, {mgmt.mgmt = #mgmt.mgmt<level = 4>}, {mgmt.mgmt = #mgmt.mgmt<level = 4>}], __resattrs = [{mgmt.mgmt = #mgmt.mgmt<level = 0>}]} {
    ^body(%input0: tensor<8xi16>, %input1: tensor<8xi16>, %input2: tensor<8xi16>, %input3: tensor<8xi16>, %input4: tensor<8xi16>, %input5: tensor<8xi16>, %input6: tensor<8xi16>, %input7: tensor<8xi16>, %input8: tensor<8xi16>, %input9: tensor<8xi16>, %input10: tensor<8xi16>, %input11: tensor<8xi16>, %input12: tensor<8xi16>, %input13: tensor<8xi16>, %input14: tensor<8xi16>, %input15: tensor<8xi16>, %input16: tensor<8xi16>, %input17: tensor<8xi16>, %input18: tensor<8xi16>, %input19: tensor<8xi16>, %input20: tensor<8xi16>, %input21: tensor<8xi16>, %input22: tensor<8xi16>, %input23: tensor<8xi16>, %input24: tensor<8xi16>, %input25: tensor<8xi16>, %input26: tensor<8xi16>, %input27: tensor<8xi16>, %input28: tensor<8xi16>, %input29: tensor<8xi16>, %input30: tensor<8xi16>, %input31: tensor<8xi16>):
      %1 = arith.muli %input0, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 4, dimension = 3>} : tensor<8xi16>
      %2 = mgmt.relinearize %1 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %3 = arith.muli %input1, %input1 {mgmt.mgmt = #mgmt.mgmt<level = 4, dimension = 3>} : tensor<8xi16>
      %4 = mgmt.relinearize %3 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %5 = arith.muli %input2, %input2 {mgmt.mgmt = #mgmt.mgmt<level = 4, dimension = 3>} : tensor<8xi16>
      %6 = mgmt.relinearize %5 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %7 = arith.muli %input3, %input3 {mgmt.mgmt = #mgmt.mgmt<level = 4, dimension = 3>} : tensor<8xi16>
      %8 = mgmt.relinearize %7 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %9 = arith.muli %input4, %input4 {mgmt.mgmt = #mgmt.mgmt<level = 4, dimension = 3>} : tensor<8xi16>
      %10 = mgmt.relinearize %9 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %11 = arith.muli %input5, %input5 {mgmt.mgmt = #mgmt.mgmt<level = 4, dimension = 3>} : tensor<8xi16>
      %12 = mgmt.relinearize %11 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %13 = arith.muli %input6, %input6 {mgmt.mgmt = #mgmt.mgmt<level = 4, dimension = 3>} : tensor<8xi16>
      %14 = mgmt.relinearize %13 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %15 = arith.muli %input7, %input7 {mgmt.mgmt = #mgmt.mgmt<level = 4, dimension = 3>} : tensor<8xi16>
      %16 = mgmt.relinearize %15 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %17 = arith.muli %input8, %input8 {mgmt.mgmt = #mgmt.mgmt<level = 4, dimension = 3>} : tensor<8xi16>
      %18 = mgmt.relinearize %17 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %19 = arith.muli %input9, %input9 {mgmt.mgmt = #mgmt.mgmt<level = 4, dimension = 3>} : tensor<8xi16>
      %20 = mgmt.relinearize %19 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %21 = arith.muli %input10, %input10 {mgmt.mgmt = #mgmt.mgmt<level = 4, dimension = 3>} : tensor<8xi16>
      %22 = mgmt.relinearize %21 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %23 = arith.muli %input11, %input11 {mgmt.mgmt = #mgmt.mgmt<level = 4, dimension = 3>} : tensor<8xi16>
      %24 = mgmt.relinearize %23 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %25 = arith.muli %input12, %input12 {mgmt.mgmt = #mgmt.mgmt<level = 4, dimension = 3>} : tensor<8xi16>
      %26 = mgmt.relinearize %25 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %27 = arith.muli %input13, %input13 {mgmt.mgmt = #mgmt.mgmt<level = 4, dimension = 3>} : tensor<8xi16>
      %28 = mgmt.relinearize %27 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %29 = arith.muli %input14, %input14 {mgmt.mgmt = #mgmt.mgmt<level = 4, dimension = 3>} : tensor<8xi16>
      %30 = mgmt.relinearize %29 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %31 = arith.muli %input15, %input15 {mgmt.mgmt = #mgmt.mgmt<level = 4, dimension = 3>} : tensor<8xi16>
      %32 = mgmt.relinearize %31 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %33 = arith.muli %input16, %input16 {mgmt.mgmt = #mgmt.mgmt<level = 4, dimension = 3>} : tensor<8xi16>
      %34 = mgmt.relinearize %33 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %35 = arith.muli %input17, %input17 {mgmt.mgmt = #mgmt.mgmt<level = 4, dimension = 3>} : tensor<8xi16>
      %36 = mgmt.relinearize %35 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %37 = arith.muli %input18, %input18 {mgmt.mgmt = #mgmt.mgmt<level = 4, dimension = 3>} : tensor<8xi16>
      %38 = mgmt.relinearize %37 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %39 = arith.muli %input19, %input19 {mgmt.mgmt = #mgmt.mgmt<level = 4, dimension = 3>} : tensor<8xi16>
      %40 = mgmt.relinearize %39 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %41 = arith.muli %input20, %input20 {mgmt.mgmt = #mgmt.mgmt<level = 4, dimension = 3>} : tensor<8xi16>
      %42 = mgmt.relinearize %41 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %43 = arith.muli %input21, %input21 {mgmt.mgmt = #mgmt.mgmt<level = 4, dimension = 3>} : tensor<8xi16>
      %44 = mgmt.relinearize %43 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %45 = arith.muli %input22, %input22 {mgmt.mgmt = #mgmt.mgmt<level = 4, dimension = 3>} : tensor<8xi16>
      %46 = mgmt.relinearize %45 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %47 = arith.muli %input23, %input23 {mgmt.mgmt = #mgmt.mgmt<level = 4, dimension = 3>} : tensor<8xi16>
      %48 = mgmt.relinearize %47 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %49 = arith.muli %input24, %input24 {mgmt.mgmt = #mgmt.mgmt<level = 4, dimension = 3>} : tensor<8xi16>
      %50 = mgmt.relinearize %49 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %51 = arith.muli %input25, %input25 {mgmt.mgmt = #mgmt.mgmt<level = 4, dimension = 3>} : tensor<8xi16>
      %52 = mgmt.relinearize %51 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %53 = arith.muli %input26, %input26 {mgmt.mgmt = #mgmt.mgmt<level = 4, dimension = 3>} : tensor<8xi16>
      %54 = mgmt.relinearize %53 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %55 = arith.muli %input27, %input27 {mgmt.mgmt = #mgmt.mgmt<level = 4, dimension = 3>} : tensor<8xi16>
      %56 = mgmt.relinearize %55 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %57 = arith.muli %input28, %input28 {mgmt.mgmt = #mgmt.mgmt<level = 4, dimension = 3>} : tensor<8xi16>
      %58 = mgmt.relinearize %57 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %59 = arith.muli %input29, %input29 {mgmt.mgmt = #mgmt.mgmt<level = 4, dimension = 3>} : tensor<8xi16>
      %60 = mgmt.relinearize %59 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %61 = arith.muli %input30, %input30 {mgmt.mgmt = #mgmt.mgmt<level = 4, dimension = 3>} : tensor<8xi16>
      %62 = mgmt.relinearize %61 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %63 = arith.muli %input31, %input31 {mgmt.mgmt = #mgmt.mgmt<level = 4, dimension = 3>} : tensor<8xi16>
      %64 = mgmt.relinearize %63 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %65 = arith.addi %2, %2 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %66 = arith.addi %4, %6 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %67 = arith.addi %65, %66 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %68 = arith.addi %8, %10 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %69 = arith.addi %12, %14 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %70 = arith.addi %68, %69 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %71 = arith.addi %67, %70 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %72 = arith.addi %16, %18 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %73 = arith.addi %20, %22 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %74 = arith.addi %72, %73 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %75 = arith.addi %24, %26 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %76 = arith.addi %28, %30 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %77 = arith.addi %75, %76 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %78 = arith.addi %74, %77 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %79 = arith.addi %71, %78 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %80 = arith.addi %32, %34 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %81 = arith.addi %36, %38 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %82 = arith.addi %80, %81 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %83 = arith.addi %40, %42 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %84 = arith.addi %44, %46 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %85 = arith.addi %83, %84 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %86 = arith.addi %82, %85 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %87 = arith.addi %48, %50 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %88 = arith.addi %52, %54 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %89 = arith.addi %87, %88 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %90 = arith.addi %56, %58 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %91 = arith.addi %60, %62 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %92 = arith.addi %91, %64 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %93 = arith.addi %90, %92 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %94 = arith.addi %89, %93 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %95 = arith.addi %86, %94 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %96 = arith.addi %79, %95 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<8xi16>
      %97 = mgmt.modreduce %96 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %98 = arith.muli %97, %97 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<8xi16>
      %99 = mgmt.relinearize %98 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %100 = mgmt.modreduce %2 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %101 = arith.muli %100, %100 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<8xi16>
      %102 = mgmt.relinearize %101 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %103 = mgmt.modreduce %4 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %104 = arith.muli %103, %103 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<8xi16>
      %105 = mgmt.relinearize %104 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %106 = mgmt.modreduce %6 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %107 = arith.muli %106, %106 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<8xi16>
      %108 = mgmt.relinearize %107 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %109 = mgmt.modreduce %8 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %110 = arith.muli %109, %109 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<8xi16>
      %111 = mgmt.relinearize %110 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %112 = mgmt.modreduce %10 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %113 = arith.muli %112, %112 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<8xi16>
      %114 = mgmt.relinearize %113 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %115 = mgmt.modreduce %12 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %116 = arith.muli %115, %115 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<8xi16>
      %117 = mgmt.relinearize %116 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %118 = mgmt.modreduce %14 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %119 = arith.muli %118, %118 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<8xi16>
      %120 = mgmt.relinearize %119 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %121 = mgmt.modreduce %16 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %122 = arith.muli %121, %121 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<8xi16>
      %123 = mgmt.relinearize %122 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %124 = mgmt.modreduce %18 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %125 = arith.muli %124, %124 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<8xi16>
      %126 = mgmt.relinearize %125 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %127 = mgmt.modreduce %20 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %128 = arith.muli %127, %127 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<8xi16>
      %129 = mgmt.relinearize %128 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %130 = mgmt.modreduce %22 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %131 = arith.muli %130, %130 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<8xi16>
      %132 = mgmt.relinearize %131 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %133 = mgmt.modreduce %24 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %134 = arith.muli %133, %133 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<8xi16>
      %135 = mgmt.relinearize %134 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %136 = mgmt.modreduce %26 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %137 = arith.muli %136, %136 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<8xi16>
      %138 = mgmt.relinearize %137 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %139 = mgmt.modreduce %28 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %140 = arith.muli %139, %139 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<8xi16>
      %141 = mgmt.relinearize %140 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %142 = mgmt.modreduce %30 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %143 = arith.muli %142, %142 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<8xi16>
      %144 = mgmt.relinearize %143 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %145 = mgmt.modreduce %32 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %146 = arith.muli %145, %145 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<8xi16>
      %147 = mgmt.relinearize %146 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %148 = mgmt.modreduce %34 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %149 = arith.muli %148, %148 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<8xi16>
      %150 = mgmt.relinearize %149 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %151 = mgmt.modreduce %36 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %152 = arith.muli %151, %151 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<8xi16>
      %153 = mgmt.relinearize %152 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %154 = mgmt.modreduce %38 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %155 = arith.muli %154, %154 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<8xi16>
      %156 = mgmt.relinearize %155 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %157 = mgmt.modreduce %40 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %158 = arith.muli %157, %157 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<8xi16>
      %159 = mgmt.relinearize %158 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %160 = mgmt.modreduce %42 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %161 = arith.muli %160, %160 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<8xi16>
      %162 = mgmt.relinearize %161 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %163 = mgmt.modreduce %44 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %164 = arith.muli %163, %163 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<8xi16>
      %165 = mgmt.relinearize %164 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %166 = mgmt.modreduce %46 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %167 = arith.muli %166, %166 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<8xi16>
      %168 = mgmt.relinearize %167 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %169 = mgmt.modreduce %48 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %170 = arith.muli %169, %169 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<8xi16>
      %171 = mgmt.relinearize %170 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %172 = mgmt.modreduce %50 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %173 = arith.muli %172, %172 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<8xi16>
      %174 = mgmt.relinearize %173 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %175 = mgmt.modreduce %52 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %176 = arith.muli %175, %175 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<8xi16>
      %177 = mgmt.relinearize %176 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %178 = mgmt.modreduce %54 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %179 = arith.muli %178, %178 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<8xi16>
      %180 = mgmt.relinearize %179 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %181 = mgmt.modreduce %56 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %182 = arith.muli %181, %181 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<8xi16>
      %183 = mgmt.relinearize %182 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %184 = mgmt.modreduce %58 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %185 = arith.muli %184, %184 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<8xi16>
      %186 = mgmt.relinearize %185 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %187 = mgmt.modreduce %60 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %188 = arith.muli %187, %187 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<8xi16>
      %189 = mgmt.relinearize %188 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %190 = mgmt.modreduce %62 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %191 = arith.muli %190, %190 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<8xi16>
      %192 = mgmt.relinearize %191 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %193 = mgmt.modreduce %64 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %194 = arith.muli %193, %193 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<8xi16>
      %195 = mgmt.relinearize %194 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %196 = arith.addi %99, %102 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %197 = arith.addi %105, %108 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %198 = arith.addi %196, %197 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %199 = arith.addi %111, %114 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %200 = arith.addi %117, %120 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %201 = arith.addi %199, %200 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %202 = arith.addi %198, %201 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %203 = arith.addi %123, %126 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %204 = arith.addi %129, %132 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %205 = arith.addi %203, %204 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %206 = arith.addi %135, %138 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %207 = arith.addi %141, %144 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %208 = arith.addi %206, %207 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %209 = arith.addi %205, %208 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %210 = arith.addi %202, %209 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %211 = arith.addi %147, %150 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %212 = arith.addi %153, %156 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %213 = arith.addi %211, %212 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %214 = arith.addi %159, %162 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %215 = arith.addi %165, %168 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %216 = arith.addi %214, %215 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %217 = arith.addi %213, %216 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %218 = arith.addi %171, %174 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %219 = arith.addi %177, %180 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %220 = arith.addi %218, %219 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %221 = arith.addi %183, %186 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %222 = arith.addi %189, %192 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %223 = arith.addi %222, %195 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %224 = arith.addi %221, %223 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %225 = arith.addi %220, %224 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %226 = arith.addi %217, %225 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %227 = arith.addi %210, %226 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<8xi16>
      %228 = mgmt.modreduce %227 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %229 = arith.muli %228, %228 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<8xi16>
      %230 = mgmt.relinearize %229 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %231 = mgmt.modreduce %102 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %232 = arith.muli %231, %231 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<8xi16>
      %233 = mgmt.relinearize %232 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %234 = mgmt.modreduce %105 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %235 = arith.muli %234, %234 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<8xi16>
      %236 = mgmt.relinearize %235 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %237 = mgmt.modreduce %108 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %238 = arith.muli %237, %237 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<8xi16>
      %239 = mgmt.relinearize %238 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %240 = mgmt.modreduce %111 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %241 = arith.muli %240, %240 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<8xi16>
      %242 = mgmt.relinearize %241 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %243 = mgmt.modreduce %114 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %244 = arith.muli %243, %243 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<8xi16>
      %245 = mgmt.relinearize %244 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %246 = mgmt.modreduce %117 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %247 = arith.muli %246, %246 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<8xi16>
      %248 = mgmt.relinearize %247 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %249 = mgmt.modreduce %120 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %250 = arith.muli %249, %249 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<8xi16>
      %251 = mgmt.relinearize %250 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %252 = mgmt.modreduce %123 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %253 = arith.muli %252, %252 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<8xi16>
      %254 = mgmt.relinearize %253 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %255 = mgmt.modreduce %126 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %256 = arith.muli %255, %255 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<8xi16>
      %257 = mgmt.relinearize %256 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %258 = mgmt.modreduce %129 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %259 = arith.muli %258, %258 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<8xi16>
      %260 = mgmt.relinearize %259 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %261 = mgmt.modreduce %132 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %262 = arith.muli %261, %261 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<8xi16>
      %263 = mgmt.relinearize %262 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %264 = mgmt.modreduce %135 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %265 = arith.muli %264, %264 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<8xi16>
      %266 = mgmt.relinearize %265 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %267 = mgmt.modreduce %138 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %268 = arith.muli %267, %267 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<8xi16>
      %269 = mgmt.relinearize %268 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %270 = mgmt.modreduce %141 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %271 = arith.muli %270, %270 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<8xi16>
      %272 = mgmt.relinearize %271 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %273 = mgmt.modreduce %144 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %274 = arith.muli %273, %273 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<8xi16>
      %275 = mgmt.relinearize %274 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %276 = mgmt.modreduce %147 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %277 = arith.muli %276, %276 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<8xi16>
      %278 = mgmt.relinearize %277 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %279 = mgmt.modreduce %150 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %280 = arith.muli %279, %279 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<8xi16>
      %281 = mgmt.relinearize %280 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %282 = mgmt.modreduce %153 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %283 = arith.muli %282, %282 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<8xi16>
      %284 = mgmt.relinearize %283 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %285 = mgmt.modreduce %156 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %286 = arith.muli %285, %285 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<8xi16>
      %287 = mgmt.relinearize %286 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %288 = mgmt.modreduce %159 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %289 = arith.muli %288, %288 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<8xi16>
      %290 = mgmt.relinearize %289 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %291 = mgmt.modreduce %162 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %292 = arith.muli %291, %291 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<8xi16>
      %293 = mgmt.relinearize %292 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %294 = mgmt.modreduce %165 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %295 = arith.muli %294, %294 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<8xi16>
      %296 = mgmt.relinearize %295 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %297 = mgmt.modreduce %168 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %298 = arith.muli %297, %297 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<8xi16>
      %299 = mgmt.relinearize %298 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %300 = mgmt.modreduce %171 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %301 = arith.muli %300, %300 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<8xi16>
      %302 = mgmt.relinearize %301 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %303 = mgmt.modreduce %174 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %304 = arith.muli %303, %303 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<8xi16>
      %305 = mgmt.relinearize %304 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %306 = mgmt.modreduce %177 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %307 = arith.muli %306, %306 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<8xi16>
      %308 = mgmt.relinearize %307 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %309 = mgmt.modreduce %180 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %310 = arith.muli %309, %309 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<8xi16>
      %311 = mgmt.relinearize %310 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %312 = mgmt.modreduce %183 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %313 = arith.muli %312, %312 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<8xi16>
      %314 = mgmt.relinearize %313 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %315 = mgmt.modreduce %186 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %316 = arith.muli %315, %315 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<8xi16>
      %317 = mgmt.relinearize %316 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %318 = mgmt.modreduce %189 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %319 = arith.muli %318, %318 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<8xi16>
      %320 = mgmt.relinearize %319 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %321 = mgmt.modreduce %192 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %322 = arith.muli %321, %321 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<8xi16>
      %323 = mgmt.relinearize %322 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %324 = mgmt.modreduce %195 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %325 = arith.muli %324, %324 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<8xi16>
      %326 = mgmt.relinearize %325 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %327 = arith.addi %230, %233 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %328 = arith.addi %236, %239 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %329 = arith.addi %327, %328 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %330 = arith.addi %242, %245 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %331 = arith.addi %248, %251 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %332 = arith.addi %330, %331 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %333 = arith.addi %329, %332 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %334 = arith.addi %254, %257 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %335 = arith.addi %260, %263 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %336 = arith.addi %334, %335 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %337 = arith.addi %266, %269 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %338 = arith.addi %272, %275 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %339 = arith.addi %337, %338 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %340 = arith.addi %336, %339 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %341 = arith.addi %333, %340 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %342 = arith.addi %278, %281 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %343 = arith.addi %284, %287 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %344 = arith.addi %342, %343 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %345 = arith.addi %290, %293 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %346 = arith.addi %296, %299 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %347 = arith.addi %345, %346 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %348 = arith.addi %344, %347 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %349 = arith.addi %302, %305 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %350 = arith.addi %308, %311 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %351 = arith.addi %349, %350 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %352 = arith.addi %314, %317 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %353 = arith.addi %320, %323 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %354 = arith.addi %353, %326 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %355 = arith.addi %352, %354 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %356 = arith.addi %351, %355 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %357 = arith.addi %348, %356 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %358 = arith.addi %341, %357 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
      %359 = mgmt.modreduce %358 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %360 = arith.muli %359, %359 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xi16>
      %361 = mgmt.relinearize %360 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %362 = mgmt.modreduce %233 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %363 = arith.muli %362, %362 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xi16>
      %364 = mgmt.relinearize %363 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %365 = mgmt.modreduce %236 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %366 = arith.muli %365, %365 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xi16>
      %367 = mgmt.relinearize %366 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %368 = mgmt.modreduce %239 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %369 = arith.muli %368, %368 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xi16>
      %370 = mgmt.relinearize %369 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %371 = mgmt.modreduce %242 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %372 = arith.muli %371, %371 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xi16>
      %373 = mgmt.relinearize %372 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %374 = mgmt.modreduce %245 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %375 = arith.muli %374, %374 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xi16>
      %376 = mgmt.relinearize %375 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %377 = mgmt.modreduce %248 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %378 = arith.muli %377, %377 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xi16>
      %379 = mgmt.relinearize %378 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %380 = mgmt.modreduce %251 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %381 = arith.muli %380, %380 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xi16>
      %382 = mgmt.relinearize %381 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %383 = mgmt.modreduce %254 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %384 = arith.muli %383, %383 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xi16>
      %385 = mgmt.relinearize %384 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %386 = mgmt.modreduce %257 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %387 = arith.muli %386, %386 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xi16>
      %388 = mgmt.relinearize %387 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %389 = mgmt.modreduce %260 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %390 = arith.muli %389, %389 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xi16>
      %391 = mgmt.relinearize %390 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %392 = mgmt.modreduce %263 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %393 = arith.muli %392, %392 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xi16>
      %394 = mgmt.relinearize %393 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %395 = mgmt.modreduce %266 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %396 = arith.muli %395, %395 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xi16>
      %397 = mgmt.relinearize %396 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %398 = mgmt.modreduce %269 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %399 = arith.muli %398, %398 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xi16>
      %400 = mgmt.relinearize %399 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %401 = mgmt.modreduce %272 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %402 = arith.muli %401, %401 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xi16>
      %403 = mgmt.relinearize %402 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %404 = mgmt.modreduce %275 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %405 = arith.muli %404, %404 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xi16>
      %406 = mgmt.relinearize %405 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %407 = mgmt.modreduce %278 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %408 = arith.muli %407, %407 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xi16>
      %409 = mgmt.relinearize %408 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %410 = mgmt.modreduce %281 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %411 = arith.muli %410, %410 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xi16>
      %412 = mgmt.relinearize %411 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %413 = mgmt.modreduce %284 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %414 = arith.muli %413, %413 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xi16>
      %415 = mgmt.relinearize %414 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %416 = mgmt.modreduce %287 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %417 = arith.muli %416, %416 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xi16>
      %418 = mgmt.relinearize %417 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %419 = mgmt.modreduce %290 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %420 = arith.muli %419, %419 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xi16>
      %421 = mgmt.relinearize %420 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %422 = mgmt.modreduce %293 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %423 = arith.muli %422, %422 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xi16>
      %424 = mgmt.relinearize %423 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %425 = mgmt.modreduce %296 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %426 = arith.muli %425, %425 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xi16>
      %427 = mgmt.relinearize %426 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %428 = mgmt.modreduce %299 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %429 = arith.muli %428, %428 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xi16>
      %430 = mgmt.relinearize %429 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %431 = mgmt.modreduce %302 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %432 = arith.muli %431, %431 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xi16>
      %433 = mgmt.relinearize %432 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %434 = mgmt.modreduce %305 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %435 = arith.muli %434, %434 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xi16>
      %436 = mgmt.relinearize %435 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %437 = mgmt.modreduce %308 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %438 = arith.muli %437, %437 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xi16>
      %439 = mgmt.relinearize %438 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %440 = mgmt.modreduce %311 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %441 = arith.muli %440, %440 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xi16>
      %442 = mgmt.relinearize %441 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %443 = mgmt.modreduce %314 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %444 = arith.muli %443, %443 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xi16>
      %445 = mgmt.relinearize %444 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %446 = mgmt.modreduce %317 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %447 = arith.muli %446, %446 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xi16>
      %448 = mgmt.relinearize %447 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %449 = mgmt.modreduce %320 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %450 = arith.muli %449, %449 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xi16>
      %451 = mgmt.relinearize %450 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %452 = mgmt.modreduce %323 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %453 = arith.muli %452, %452 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xi16>
      %454 = mgmt.relinearize %453 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %455 = mgmt.modreduce %326 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %456 = arith.muli %455, %455 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xi16>
      %457 = mgmt.relinearize %456 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %458 = arith.addi %361, %364 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %459 = arith.addi %367, %370 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %460 = arith.addi %458, %459 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %461 = arith.addi %373, %376 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %462 = arith.addi %379, %382 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %463 = arith.addi %461, %462 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %464 = arith.addi %460, %463 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %465 = arith.addi %385, %388 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %466 = arith.addi %391, %394 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %467 = arith.addi %465, %466 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %468 = arith.addi %397, %400 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %469 = arith.addi %403, %406 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %470 = arith.addi %468, %469 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %471 = arith.addi %467, %470 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %472 = arith.addi %464, %471 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %473 = arith.addi %409, %412 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %474 = arith.addi %415, %418 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %475 = arith.addi %473, %474 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %476 = arith.addi %421, %424 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %477 = arith.addi %427, %430 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %478 = arith.addi %476, %477 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %479 = arith.addi %475, %478 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %480 = arith.addi %433, %436 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %481 = arith.addi %439, %442 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %482 = arith.addi %480, %481 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %483 = arith.addi %445, %448 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %484 = arith.addi %451, %454 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %485 = arith.addi %484, %457 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %486 = arith.addi %483, %485 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %487 = arith.addi %482, %486 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %488 = arith.addi %479, %487 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %489 = arith.addi %472, %488 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      %490 = mgmt.modreduce %489 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<8xi16>
      secret.yield %490 : tensor<8xi16>
    } -> !secret.secret<tensor<8xi16>>
    return %0 : !secret.secret<tensor<8xi16>>
  }
}

