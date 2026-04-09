#!/usr/bin/env python3
"""Extract per-test OperationCount metrics from HEIR stderr logs."""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import time
from typing import Optional, Tuple

CIPHERTEXT_RE = re.compile(r"OperationCount\(ciphertextCount=(\d+),\s*keySwitchCount=(\d+)\)")
UNINITIALIZED_RE = re.compile(r"OperationCount\(uninitialized\)")
SUFFIX_RE = re.compile(r"^(.*)-(\d+)$")


def parse_test_name(test_name: str) -> Tuple[str, Optional[int]]:
    match = SUFFIX_RE.match(test_name)
    if not match:
        return test_name, None
    return match.group(1), int(match.group(2))


EXTRACT_HEADERS = [
    "testname",
    "family",
    "levelIndex",
    "source",
    "timestamp",
    "logFile",
    "operationLines",
    "sumCiphertextCount",
    "sumKeySwitchCount",
    "maxCiphertextCount",
    "uninitializedCount",
]


def extract_payload(input_path: pathlib.Path, output_path: pathlib.Path, test_name: str, source: str) -> None:
    text = input_path.read_text() if input_path.exists() else ""

    matches = CIPHERTEXT_RE.findall(text)
    uninitialized_count = len(UNINITIALIZED_RE.findall(text))

    ciphertext_values = [int(ciphertext) for ciphertext, _ in matches]
    keyswitch_values = [int(keyswitch) for _, keyswitch in matches]

    family, level_index = parse_test_name(test_name)

    row = {
        "testname": test_name,
        "family": family,
        "levelIndex": level_index,
        "source": source,
        "timestamp": int(time.time()),
        "logFile": str(input_path),
        "operationLines": len(matches),
        "sumCiphertextCount": sum(ciphertext_values),
        "sumKeySwitchCount": sum(keyswitch_values),
        "maxCiphertextCount": max(ciphertext_values, default=0),
        "uninitializedCount": uninitialized_count,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXTRACT_HEADERS)
        writer.writeheader()
        writer.writerow(row)
def main() -> int:
    parser = argparse.ArgumentParser(description="Extract per-test OperationCount metrics")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    extract_parser = subparsers.add_parser("extract", help="Extract counts from one log file")
    extract_parser.add_argument("--input", required=True, help="Input stderr log file")
    extract_parser.add_argument("--output", required=True, help="Output CSV path")
    extract_parser.add_argument("--test-name", required=True, help="Test name")
    extract_parser.add_argument("--source", required=True, help="Source tag")

    args = parser.parse_args()

    if args.cmd == "extract":
        extract_payload(
            pathlib.Path(args.input),
            pathlib.Path(args.output),
            args.test_name,
            args.source,
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
