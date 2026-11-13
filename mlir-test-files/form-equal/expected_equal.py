import sys
import subprocess
import ctypes
import numpy as np

sys.path.insert(0, "/Users/mar69689/Documents/Projekte/Google/llvm-project/build/tools/mlir/python_packages/mlir_core")

import mlir.passmanager as pm
import mlir.ir as ir
from mlir._mlir_libs._mlirRegisterEverything import register_dialects, register_llvm_translations

BUILD_DIR  = "/Users/mar69689/Documents/Projekte/Google/llvm-project/build"
MLIR_TRANSLATE = f"{BUILD_DIR}/bin/mlir-translate"

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
    ptr  = np_array.ctypes.data
    desc.allocated = ptr
    desc.aligned   = ptr
    desc.offset    = 0
    desc.size      = np_array.shape[0]
    desc.stride    = 1
    return desc

# ── 1. Parse & lower ────────────────────────────────────────────────────────
with open("form-equal.mlir", "r") as f:
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

    with open("lowered.mlir", "w") as f:
        f.write(str(module))
    print("Lowered IR written to lowered.mlir")

# ── 2. MLIR (LLVM dialect) → LLVM IR ────────────────────────────────────────
print("Translating to LLVM IR...")
r = subprocess.run(
    [MLIR_TRANSLATE, "--mlir-to-llvmir", "lowered.mlir", "-o", "lowered.ll"],
    capture_output=True, text=True
)
if r.returncode != 0:
    print("mlir-translate failed:\n", r.stderr)
    sys.exit(1)
print("LLVM IR written to lowered.ll")

# ── 3. Compile to shared library ─────────────────────────────────────────────
print("Compiling to shared library...")
r = subprocess.run(
    ["clang", "-shared", "-fPIC", "-O0", "lowered.ll", "-o", "libfunc.dylib"],
    capture_output=True, text=True
)
if r.returncode != 0:
    print("clang failed:\n", r.stderr)
    sys.exit(1)
print("Shared library written to libfunc.dylib")

# ── 4. Load  ───────────────────────────────────────────────────────────
lib    = ctypes.CDLL("./libfunc.dylib")
cfunc  = getattr(lib, "_mlir_ciface_func")
cfunc.restype  = None
cfunc.argtypes = [ctypes.POINTER(MemRefDescriptor)] * 65

# ── Call the function ─────────────────────────────────────────────────────────
inputs = [np.ones(8, dtype=np.int16)] + [np.zeros(8, dtype=np.int16) for _ in range(63)]
output = np.zeros(8, dtype=np.int16)

all_arrays = [output] + inputs
descs = [make_memref(a) for a in all_arrays]
args  = [ctypes.pointer(d) for d in descs]

cfunc(*args)

# ── Read result from the descriptor the function wrote into ───────────────────
# The function fills descs[0].aligned with a pointer to the result data
result_ptr = ctypes.cast(descs[0].aligned, ctypes.POINTER(ctypes.c_int16))
result = np.ctypeslib.as_array(result_ptr, shape=(8,)).copy()
print("Output:", result)