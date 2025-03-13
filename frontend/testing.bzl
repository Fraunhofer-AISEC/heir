"""Macros for py_test rules that use the python frontend."""

load("@rules_python//python:defs.bzl", "py_test")

def frontend_test(name, srcs, deps = [], data = [], tags = []):
    """A py_test replacement with an env including all backend dependencies.
    """
    include_dirs = [
        "openfhe/",
        "openfhe/src/binfhe/include",
        "openfhe/src/core/include",
        "openfhe/src/pke/include",
        "cereal/include",
        "rapidjson/include",
    ]

    libs = [
        "binfhe",
        "core",
        "pke",
    ]

    py_test(
        name = name,
        srcs = srcs,
        python_version = "PY3",
        srcs_version = "PY3",
        deps = deps + [
            ":frontend",
            "@com_google_absl_py//absl/testing:absltest",
        ],
        imports = ["."],
        data = data,
        tags = tags,
        env = {
            # this dir is relative to $RUNFILES_DIR, which is set by bazel at runtime
            "OPENFHE_LIB_DIR": "openfhe",
            "OPENFHE_INCLUDE_TYPE": "source-relative",
            "OPENFHE_LINK_LIBS": ":".join(libs),
            "OPENFHE_INCLUDE_DIR": ":".join(include_dirs),
            "HEIR_REPO_ROOT_MARKER": "bazel-bin",
            "HEIR_OPT_PATH": "tools/heir-opt",
            "HEIR_TRANSLATE_PATH": "tools/heir-translate",
            "PYBIND11_INCLUDE_PATH": "pybind11/include",
            "NUMBA_USE_LEGACY_TYPE_SYSTEM": "1",
        },
    )
