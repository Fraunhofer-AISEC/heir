#!/usr/bin/env python3

import argparse
import json
import re
import math
import time
import sys

def extract_bgv_params(input_file, test_name, approach):
    """Extract BGV parameters from MLIR file and format as JSON"""
    try:
        with open(input_file, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {input_file}: {e}", file=sys.stderr)
        return None
    
    # Extract the BGV parameters block
    bgv_pattern = r'{bgv\.schemeParam = #bgv\.scheme_param<([^>]*)>'
    match = re.search(bgv_pattern, content)
    if not match:
        print(f"Error: BGV scheme parameters not found in {input_file}", file=sys.stderr)
        return None
    
    params_text = match.group(1)
    
    # Extract individual parameters
    log_n_match = re.search(r'logN = (\d+)', params_text)
    q_values_match = re.search(r'Q = \[([\d, ]+)\]', params_text)
    p_values_match = re.search(r'P = \[([\d, ]+)\]', params_text)
    plaintext_mod_match = re.search(r'plaintextModulus = (\d+)', params_text)
    
    if not all([log_n_match, q_values_match, plaintext_mod_match]):
        print(f"Error: Could not extract all required parameters from {input_file}", file=sys.stderr)
        return None
    
    log_n = int(log_n_match.group(1))
    q_values = [int(q.strip()) for q in q_values_match.group(1).split(',')]
    plaintext_mod = int(plaintext_mod_match.group(1))
    
    # Handle P values, which might be empty
    p_values = []
    if p_values_match:
        p_values = [int(p.strip()) for p in p_values_match.group(1).split(',')]
    
    # Calculate bit sizes for Q values
    moduli_sizes = [math.floor(math.log2(q)) + 1 for q in q_values]
    total_size = sum(moduli_sizes)
    
    # Calculate bit sizes for P values
    extension_sizes = [math.floor(math.log2(p)) + 1 for p in p_values]
    extension_total = sum(extension_sizes)
    
    # Calculate ring dimension
    ring_dimension = 2 ** log_n
    
    # Create JSON output
    result = {
        "testname": test_name,
        "selectionApproach": approach,
        "timestamp": int(time.time()),
        "modulusSizes": moduli_sizes,
        "totalSize": total_size,
        "ringDimension": ring_dimension,
        "plaintextModulus": plaintext_mod,
        "extensionModuliSize": extension_total,
        "extensionModuliSizes": extension_sizes,
    }
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Extract BGV parameters from MLIR file')
    parser.add_argument('input_file', help='Input MLIR file')
    parser.add_argument('output_file', help='Output JSON file')
    parser.add_argument('test_name', help='Test name')
    parser.add_argument('approach', help='Parameter selection approach')
    
    args = parser.parse_args()
    
    result = extract_bgv_params(args.input_file, args.test_name, args.approach)
    if result:
        try:
            with open(args.output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"BGV parameters extracted and saved to {args.output_file}")
        except Exception as e:
            print(f"Error writing to file {args.output_file}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
