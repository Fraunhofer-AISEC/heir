#!/bin/bash

# Script to create a new test structure based on the dot8 template

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <test_name>"
    echo "Example: $0 matrix_mult"
    exit 1
fi

TEST_NAME=$1
TEST_DIR="/home/ubuntu/heir-aisec/mlir-test-files/$TEST_NAME"

# Check if directory already exists and is non-empty
if [ -d "$TEST_DIR" ] && [ "$(ls -A $TEST_DIR 2>/dev/null)" ]; then
    read -p "Directory $TEST_DIR already exists and is not empty. Overwrite? (y/n): " confirm
    if [[ $confirm != [Yy]* ]]; then
        echo "Operation cancelled."
        exit 0
    fi
fi

# Call the Python script
python3 /home/ubuntu/heir-aisec/mlir-test-files/generate_test_structure.py "$TEST_NAME"

# Make the new test directory executable
chmod +x "$TEST_DIR"