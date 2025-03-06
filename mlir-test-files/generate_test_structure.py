#!/usr/bin/env python3

import os
import sys
import shutil
import re

def create_directory(directory_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

def replace_test_name(content, old_name, new_name):
    """Replace all occurrences of the test name"""
    
    # Handle function names
    content = content.replace(f"main_{old_name}_common", f"main_{new_name}_common")
    content = content.replace(f"{old_name}_common", f"{new_name}_common")
    content = content.replace(f"{old_name}_openfhe", f"{new_name}_openfhe")
    content = content.replace(f"{old_name}_direct", f"{new_name}_direct")
    content = content.replace(f"{old_name}_bisection", f"{new_name}_bisection")
    
    # Handle filenames in includes
    content = content.replace(f"main_{old_name}.h", f"main_{new_name}.h")
    content = content.replace(f"{old_name}.mlir", f"{new_name}.mlir")
    
    # Handle uppercase define guards for headers
    content = content.replace(f"{old_name.upper()}_COMMON_H", f"{new_name.upper().replace('-', '_')}_COMMON_H")

    # Replace the test name with proper case preservation
    content = content.replace(old_name, new_name)
    content = content.replace(old_name.upper(), new_name.upper())
    content = content.replace(old_name.capitalize(), new_name.capitalize())
    
    return content

def copy_and_modify_file(src_path, dest_path, old_name, new_name):
    """Copy a file and replace the test name references"""
    if not os.path.exists(src_path):
        print(f"Error: Source file {src_path} does not exist")
        return
    
    with open(src_path, 'r') as file:
        content = file.read()
    
    modified_content = replace_test_name(content, old_name, new_name)
    
    with open(dest_path, 'w') as file:
        file.write(modified_content)
    
    print(f"Created: {dest_path}")

def add_build_targets(test_name):
    """Add the required targets to the BUILD file"""
    build_file_path = "/home/ubuntu/heir-aisec/mlir-test-files/BUILD"
    
    with open(build_file_path, 'r') as file:
        content = file.read()
    
    # Generate new build targets for the test
    new_targets = f"""
cc_library(
    name = "main_{test_name}_common",
    srcs = ["{test_name}/main_{test_name}_common.cpp"],
    hdrs = ["{test_name}/main_{test_name}_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_{test_name}_bisection",
    srcs = ["{test_name}/main_{test_name}_bisection.cpp"],
    deps = [
        ":main_{test_name}_common",
        ":{test_name}_bisection",
    ],
)

cc_binary(
    name = "main_{test_name}_openfhe",
    srcs = ["{test_name}/main_{test_name}_openfhe.cpp"],
    deps = [
        ":main_{test_name}_common",
        ":{test_name}_openfhe",
    ],
)

cc_binary(
    name = "main_{test_name}_direct",
    srcs = ["{test_name}/main_{test_name}_direct.cpp"],
    deps = [
        ":main_{test_name}_common",
        ":{test_name}_direct",
    ],
)

cc_library(
    name = "{test_name}_bisection",
    srcs = ["{test_name}/{test_name}_bisection.cpp"],
    hdrs = ["{test_name}/{test_name}_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "{test_name}_openfhe",
    srcs = ["{test_name}/{test_name}_openfhe.cpp"],
    hdrs = ["{test_name}/{test_name}_openfhe.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "{test_name}_direct",
    srcs = ["{test_name}/{test_name}_direct.cpp"],
    hdrs = ["{test_name}/{test_name}_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)
"""
    
    # Append the new targets to the BUILD file
    with open(build_file_path, 'a') as file:
        file.write(new_targets)
    
    print(f"Added BUILD targets for {test_name}")

def generate_test_structure(test_name):
    """Generate the test structure for a new test name"""
    # Base directory for all tests
    base_dir = "/home/ubuntu/heir-aisec/mlir-test-files"
    
    # Template test name
    template_name = "dot8"
    template_dir = os.path.join(base_dir, template_name)
    
    # New test directory
    new_test_dir = os.path.join(base_dir, test_name)
    create_directory(new_test_dir)
    
    # Files to create (based on dot8 structure)
    files = [
        f"main_{template_name}_common.h",
        f"main_{template_name}_common.cpp",
        f"main_{template_name}_openfhe.cpp",
        f"main_{template_name}_direct.cpp",
        f"main_{template_name}_bisection.cpp",
        f"{template_name}.mlir"
    ]
    
    # Create each file with appropriate content modifications
    for file in files:
        src_path = os.path.join(template_dir, file)
        dest_file = file.replace(template_name, test_name)
        dest_path = os.path.join(new_test_dir, dest_file)
        copy_and_modify_file(src_path, dest_path, template_name, test_name)
    
    # Add the required BUILD targets
    add_build_targets(test_name)
    
    print(f"\nTest structure for '{test_name}' has been generated.")
    print(f"You may need to further modify the files to adapt them to your specific test.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_test_structure.py <test_name>")
        sys.exit(1)
    
    test_name = sys.argv[1]
    generate_test_structure(test_name)
