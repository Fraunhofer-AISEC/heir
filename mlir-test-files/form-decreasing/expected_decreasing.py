import sys
sys.path.insert(0, "/Users/mar69689/Documents/Projekte/Google/llvm-project/build/tools/mlir/python_packages/mlir_core")

import mlir.execution_engine as exe
import mlir.passmanager as pm
import mlir.ir as ir
from mlir._mlir_libs._mlirRegisterEverything import register_dialects, register_llvm_translations
import ctypes
import numpy as np

class MemRefDescriptor(ctypes.Structure):
    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned",   ctypes.c_void_p),
        ("offset",    ctypes.c_int64),
        ("size",      ctypes.c_int64),
        ("stride",    ctypes.c_int64),
    ]

def make_memref(np_array):
    desc = MemRefDescriptor()
    ptr = np_array.ctypes.data
    desc.allocated = ptr
    desc.aligned   = ptr
    desc.offset    = 0
    desc.size      = np_array.shape[0]
    desc.stride    = 1
    return desc

with open("form-decreasing.mlir", "r") as f:
    src = f.read()

registry = ir.DialectRegistry()
register_dialects(registry)

with ir.Context() as ctx:
    ctx.append_dialect_registry(registry)
    ctx.load_all_available_dialects()
    ctx.enable_multithreading(False)
    register_llvm_translations(ctx)

    module = ir.Module.parse(src)

    pipeline = (
        "convert-elementwise-to-linalg,"
        "one-shot-bufferize{bufferize-function-boundaries=true},"
        "convert-linalg-to-loops,"
        "convert-scf-to-cf,"
        "convert-index-to-llvm,"
        "convert-arith-to-llvm,"
        "expand-strided-metadata,"
        "finalize-memref-to-llvm,"
        "convert-cf-to-llvm,"
        "func.func(llvm-request-c-wrappers),"
        "convert-func-to-llvm,"
        "reconcile-unrealized-casts"
    )
    pm.PassManager.parse(f"builtin.module({pipeline})").run(module.operation)

    # Write lowered IR for inspection
    with open("lowered.mlir", "w") as f:
        f.write(str(module))
    print("Lowered IR written to lowered.mlir")

    # Create execution engine
    print("Creating execution engine...")
    engine = exe.ExecutionEngine(module, opt_level=0)
    print("Engine created successfully")

    # --- Diagnostic: check symbol via ctypes in current process ---
    lib = ctypes.CDLL(None)
    for name in ["_mlir_ciface_func", "__mlir_ciface_func", "func", "_func"]:
        try:
            sym = getattr(lib, name)
            print(f"Found symbol via ctypes: {name} -> {sym}")
        except AttributeError:
            print(f"Symbol NOT found via ctypes: {name}")

    # --- Try engine.invoke (uses ciface internally) ---
    print("Trying engine.invoke...")
    inputs = [np.ones(8, dtype=np.int16)] + [np.zeros(8, dtype=np.int16) for _ in range(64)]
    output = np.zeros(8, dtype=np.int16)

    all_arrays = [output] + inputs
    descs = [make_memref(a) for a in all_arrays]
    args = [ctypes.pointer(d) for d in descs]

    try:
        engine.invoke("func", *args)
        print("invoke succeeded!")
        print("Output:", output)
    except Exception as e:
        print(f"invoke failed: {e}")

    # --- Try lookup with various name variants ---
    for name in ["_mlir_ciface_func", "__mlir_ciface_func", "func", "_func"]:
        try:
            fptr = engine.lookup(name)
            print(f"engine.lookup succeeded for: {name} -> {fptr}")
        except RuntimeError as e:
            print(f"engine.lookup failed for '{name}': {e}")