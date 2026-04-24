#!/usr/bin/env python3
import argparse
import json
import re
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 16
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["legend.title_fontsize"] = 12

COMPARISON_APPROACH_ORDER = ["direct", "closed", "bisection", "greedy", "gap-mono"]
APPROACH_DISPLAY_NAMES = {
    "direct": "openfhe",
    "gap-mono": "heir",
}
APPROACH_COMPARISON_KIND = "execution"
APPROACH_EXECUTION_BACKEND = {
    "greedy": "lattigo",
    "gap-mono": "lattigo",
}
KIND_PLOT_PRIORITY = ["annotated", "computed", "execution", "openfhe-result", "lattigo-result"]
BAR_COLORS = ["#179c7d", "#005b7f", "#a6bbc8", "#008598", "#39c1cd", "#b2d235"]
BAR_HATCHES = ["///", "\\\\", "xx", "..", "++", "oo"]
KIND_HATCHES = {
    "execution": "",
    "computed": "//",
    "annotated": "xx",
    "openfhe-result": "..",
    "lattigo-result": "++",
}
FAMILY_TEST_LABELS: Dict[str, List[Tuple[str, str]]] = {
    "add-eq": [
        ("add-eq-2", "Depth 2"),
        ("add-eq-6", "Depth 6"),
        ("add-eq-10", "Depth 10"),
        ("add-eq-14", "Depth 14"),
        ("add-eq-18", "Depth 18"),
    ],
    "add-64": [
        ("add-64-0", "Level 4\n(before mult)"),
        ("add-64-1", "Level 4"),
        ("add-64-2", "Level 3"),
        ("add-64-3", "Level 2"),
        ("add-64-4", "Level 1"),
    ],
    "rotate": [
        ("rotate-0", "Level 4\n(before mult)"),
        ("rotate-1", "Level 4"),
        ("rotate-2", "Level 3"),
        ("rotate-3", "Level 2"),
        ("rotate-4", "Level 1"),
    ],
    "rotate-of": [
        ("rotate-of-1", "Level 4"),
        ("rotate-of-2", "Level 3"),
        ("rotate-of-3", "Level 2"),
        ("rotate-of-4", "Level 1"),
    ],
    "add-num": [
        ("add-num-1", "2"),
        ("add-num-32", "32"),
        ("add-num-64", "64"),
        ("add-num-128", "128"),
    ],
    "rotate-num": [
        ("rotate-num-1", "2"),
        ("rotate-num-32", "17"),
        ("rotate-num-64", "33"),
        ("rotate-num-128", "65"),
    ],
    "form": [
        ("form-equal", "Equal"),
        ("form-high-low", "High-Low$^*$"),
        ("form-low-high", "Low-High"),
        ("form-hill", "Hill"),
        ("form-valley", "Valley"),
        ("form-increasing", "Increasing$^*$"),
        ("form-decreasing", "Decreasing"),
    ],
}
JSON_FILE_RE = re.compile(
    r"^(?P<test>.+?)_(?P<approach>[a-z0-9\-]+?)_(?P<kind>[a-z0-9\-]+)_(?P<timestamp>\d{8}_\d{6}|\d+)\.json$"
)

# Build a lookup once at module level
_TEST_TO_FAMILY: Dict[str, str] = {
    test: family
    for family, entries in FAMILY_TEST_LABELS.items()
    for test, _ in entries
}

@dataclass
class ResultRecord:
    test_name: str
    approach: str
    timestamp: int
    kind: str
    total_size: int
    modulus_sizes: List[int]
    source_file: Path


def parse_timestamp(raw: str) -> int:
    # Supports both unix timestamps (e.g., 1773849681) and
    # file timestamps (e.g., 20260318_160047).
    if "_" in raw:
        return int(raw.replace("_", ""))
    return int(raw)


def parse_test_script_algorithms(test_script: Path) -> List[str]:
    # Parse process_algorithm calls to provide context in the final summary.
    approach_re = re.compile(r'process_algorithm\s+"(?P<algo>[a-z0-9\-]+)"')

    content = test_script.read_text()

    parsed_order: List[str] = []
    for m in approach_re.finditer(content):
        algo = m.group("algo").lower()
        if algo not in parsed_order:
            parsed_order.append(algo)
    return parsed_order


def family_name(test_name: str) -> str:
    if test_name in _TEST_TO_FAMILY:
        return _TEST_TO_FAMILY[test_name]
    # Fallback for tests not in FAMILY_TEST_LABELS
    if "-" not in test_name:
        return test_name
    return test_name.rsplit("-", 1)[0]


def parse_total_size(payload: dict) -> Tuple[int, List[int]]:
    modulus_sizes = payload.get("modulusSizes")
    if modulus_sizes is None:
        # Execution result payloads store ciphertext primes under qModulusSizes.
        modulus_sizes = payload.get("qModulusSizes")

    if not isinstance(modulus_sizes, list) or not all(isinstance(x, (int, float)) for x in modulus_sizes):
        raise ValueError("Invalid or missing modulusSizes/qModulusSizes")

    modulus_sizes = [int(x) for x in modulus_sizes]
    computed_total = int(sum(modulus_sizes))

    if "totalSize" in payload:
        try:
            reported_total = int(payload["totalSize"])
            if reported_total != computed_total:
                print(
                    f"[warn] totalSize mismatch for {payload.get('testname', '<unknown>')}: "
                    f"reported={reported_total}, computed={computed_total}. Using computed total."
                )
        except Exception:
            pass

    return computed_total, modulus_sizes


def normalize_test_name(payload_test_name: object, fallback: str) -> str:
    if not isinstance(payload_test_name, str):
        return fallback

    candidate = payload_test_name.strip()
    if not candidate:
        return fallback

    # Ignore template placeholders like "<testname>" and other invalid tokens.
    if (candidate.startswith("<") and candidate.endswith(">")) or ("<" in candidate or ">" in candidate):
        return fallback

    return candidate


def should_replace(current: Optional[ResultRecord], candidate: ResultRecord) -> bool:
    if current is None:
        return True

    # Prefer newer timestamp.
    if candidate.timestamp != current.timestamp:
        return candidate.timestamp > current.timestamp
    return False


def load_results(
    data_dir: Path,
    approach_order: List[str],
    selected_kind: str,
    preferred_execution_backend: Dict[str, str],
) -> Tuple[
    Dict[Tuple[str, str], ResultRecord],
    Dict[Tuple[str, str], Set[str]],
    Dict[Tuple[str, str, str], ResultRecord],
]:
    selected: Dict[Tuple[str, str], ResultRecord] = {}
    available_kinds: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    latest_by_kind: Dict[Tuple[str, str, str], ResultRecord] = {}
    allowed_approaches = set(approach_order)

    for path in sorted(data_dir.glob("*.json")):
        m = JSON_FILE_RE.match(path.name)
        if not m:
            continue

        approach = m.group("approach").lower()
        if approach not in allowed_approaches:
            continue

        kind = m.group("kind")
        if kind.endswith("-result"):
            continue
        timestamp = parse_timestamp(m.group("timestamp"))

        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            print(f"[warn] Failed to parse {path}: {exc}")
            continue

        backend_raw = payload.get("backend")
        backend = backend_raw.lower() if isinstance(backend_raw, str) else ""
        expected_backend = preferred_execution_backend.get(approach, "")
        if kind == "execution" and expected_backend and backend != expected_backend:
            continue

        test_name = normalize_test_name(payload.get("testname"), m.group("test"))

        try:
            total_size, modulus_sizes = parse_total_size(payload)
        except ValueError as exc:
            print(f"[warn] Skipping {path}: {exc}")
            continue

        record = ResultRecord(
            test_name=test_name,
            approach=approach,
            timestamp=timestamp,
            kind=kind,
            total_size=total_size,
            modulus_sizes=modulus_sizes,
            source_file=path,
        )

        key = (test_name, approach)
        available_kinds[key].add(kind)

        by_kind_key = (test_name, approach, kind)
        if should_replace(latest_by_kind.get(by_kind_key), record):
            latest_by_kind[by_kind_key] = record

        if kind != selected_kind:
            continue

        if should_replace(selected.get(key), record):
            selected[key] = record

    return selected, available_kinds, latest_by_kind


def group_by_family(selected: Dict[Tuple[str, str], ResultRecord]) -> Dict[str, Dict[str, Dict[str, ResultRecord]]]:
    grouped: Dict[str, Dict[str, Dict[str, ResultRecord]]] = defaultdict(lambda: defaultdict(dict))
    for (test_name, approach), record in selected.items():
        grouped[family_name(test_name)][test_name][approach] = record
    return grouped


def group_by_family_kind(
    latest_by_kind: Dict[Tuple[str, str, str], ResultRecord]
) -> Dict[str, Dict[str, Dict[str, Dict[str, ResultRecord]]]]:
    grouped: Dict[str, Dict[str, Dict[str, Dict[str, ResultRecord]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    for (test_name, approach, kind), record in latest_by_kind.items():
        grouped[family_name(test_name)][test_name][approach][kind] = record
    return grouped


def ordered_tests_and_labels(family: str, tests_present: Set[str]) -> Tuple[List[str], List[str]]:
    configured = FAMILY_TEST_LABELS.get(family, [])
    tests: List[str] = []
    labels: List[str] = []

    for test_name, display_label in configured:
        if test_name in tests_present:
            tests.append(test_name)
            labels.append(display_label)

    remaining = sorted(t for t in tests_present if t not in set(tests))
    tests.extend(remaining)
    labels.extend(remaining)

    return tests, labels


def approach_display_name(approach: str) -> str:
    return APPROACH_DISPLAY_NAMES.get(approach, approach)


def plot_family(
    family: str,
    family_data: Dict[str, Dict[str, ResultRecord]],
    approach_order: List[str],
    out_dir: Path,
    pdf_dir: Path,
) -> None:
    tests, display_labels = ordered_tests_and_labels(family, set(family_data.keys()))
    num_tests = len(tests)
    num_approaches = len(approach_order)

    x = np.arange(num_tests)
    width = min(0.18, 0.85 / max(1, num_approaches))

    fig_width = max(10, 1.4 * num_tests)
    fig, ax = plt.subplots(figsize=(fig_width, 6.5))

    max_y = 1
    missing_points: List[Tuple[float, float]] = []

    for idx, approach in enumerate(approach_order):
        offset = (idx - (num_approaches - 1) / 2) * width
        color = BAR_COLORS[idx % len(BAR_COLORS)]
        hatch = BAR_HATCHES[idx % len(BAR_HATCHES)]
        heights = []
        for test in tests:
            record = family_data[test].get(approach)
            if record is None:
                heights.append(np.nan)
            else:
                heights.append(record.total_size)
                max_y = max(max_y, record.total_size)

        heights_np = np.array(heights, dtype=float)
        bars = ax.bar(
            x + offset,
            np.nan_to_num(heights_np, nan=0.0),
            width=width,
            label=approach_display_name(approach),
            alpha=0.95,
            color=color,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.6,
        )

        for bar, h in zip(bars, heights_np):
            if np.isnan(h):
                bar.set_alpha(0.15)
                missing_points.append((bar.get_x() + bar.get_width() / 2, 0))
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + max(1.0, 0.01 * max_y),
                f"{int(h)}",
                ha="center",
                va="bottom",
                fontsize=10,
                rotation=90,
            )

    for px, _ in missing_points:
        ax.text(px, max(1.0, 0.02 * max_y), "NA", ha="center", va="bottom", fontsize=9, rotation=90)

    ax.set_ylabel(r"$\mathrm{Total\ Ciphertext\ Modulus\ (bit)}$")
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=35, ha="right")
    ax.set_ylim(0, max_y * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(title=r"$\mathrm{Approach}$")
    fig.tight_layout()

    out_path = out_dir / f"{family}_comparison.png"
    fig.savefig(out_path, dpi=180)
    pdf_path = pdf_dir / f"{family}_comparison.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)


def plot_family_kind_comparison(
    family: str,
    family_data_by_kind: Dict[str, Dict[str, Dict[str, ResultRecord]]],
    approach_order: List[str],
    out_dir: Path,
    pdf_dir: Path,
) -> None:
    tests, display_labels = ordered_tests_and_labels(family, set(family_data_by_kind.keys()))
    series: List[Tuple[str, str]] = []
    for approach in approach_order:
        kinds_present: Set[str] = set()
        for test in tests:
            kinds_present.update(family_data_by_kind[test].get(approach, {}).keys())

        ordered_kinds = sorted(
            kinds_present,
            key=lambda k: (
                KIND_PLOT_PRIORITY.index(k) if k in KIND_PLOT_PRIORITY else len(KIND_PLOT_PRIORITY),
                k,
            ),
        )
        for kind in ordered_kinds:
            series.append((approach, kind))

    if not series:
        return

    num_tests = len(tests)
    num_series = len(series)
    x = np.arange(num_tests)
    width = min(0.14, 0.88 / max(1, num_series))

    fig_width = max(12, 1.7 * num_tests)
    fig, ax = plt.subplots(figsize=(fig_width, 7.0))

    max_y = 1
    missing_points: List[Tuple[float, float]] = []

    for idx, (approach, kind) in enumerate(series):
        offset = (idx - (num_series - 1) / 2) * width
        color = BAR_COLORS[approach_order.index(approach) % len(BAR_COLORS)]
        hatch = KIND_HATCHES.get(kind, BAR_HATCHES[idx % len(BAR_HATCHES)])

        heights = []
        for test in tests:
            record = family_data_by_kind[test].get(approach, {}).get(kind)
            if record is None:
                heights.append(np.nan)
            else:
                heights.append(record.total_size)
                max_y = max(max_y, record.total_size)

        heights_np = np.array(heights, dtype=float)
        legend_label = f"{approach_display_name(approach)}-{kind}"
        bars = ax.bar(
            x + offset,
            np.nan_to_num(heights_np, nan=0.0),
            width=width,
            label=legend_label,
            alpha=0.95,
            color=color,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.6,
        )

        for bar, h in zip(bars, heights_np):
            if np.isnan(h):
                bar.set_alpha(0.15)
                missing_points.append((bar.get_x() + bar.get_width() / 2, 0))
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + max(1.0, 0.01 * max_y),
                f"{int(h)}",
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=90,
            )

    for px, _ in missing_points:
        ax.text(px, max(1.0, 0.02 * max_y), "NA", ha="center", va="bottom", fontsize=9, rotation=90)

    ax.set_ylabel(r"$\mathrm{Total\ Ciphertext\ Modulus\ (bit)}$")
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=35, ha="right")
    ax.set_ylim(0, max_y * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(title=r"$\mathrm{Approach\!\!\text{-}\!Kind}$", ncol=2, fontsize=10)
    fig.tight_layout()

    out_path = out_dir / f"{family}_kind_comparison.png"
    fig.savefig(out_path, dpi=180)
    pdf_path = pdf_dir / f"{family}_kind_comparison.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)


def write_summary_csv(
    grouped: Dict[str, Dict[str, Dict[str, ResultRecord]]],
    approach_order: List[str],
    out_path: Path,
) -> None:
    lines = ["family,test," + ",".join(approach_order)]
    for family in sorted(grouped.keys()):
        for test in sorted(grouped[family].keys()):
            values = []
            for approach in approach_order:
                record = grouped[family][test].get(approach)
                values.append(str(record.total_size) if record else "")
            lines.append(f"{family},{test}," + ",".join(values))

    out_path.write_text("\n".join(lines) + "\n")


def write_availability_csv(
    grouped: Dict[str, Dict[str, Dict[str, ResultRecord]]],
    available_kinds: Dict[Tuple[str, str], Set[str]],
    expected_kinds: Dict[str, Set[str]],
    approach_order: List[str],
    out_path: Path,
) -> None:
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "family",
                "test",
                "approach",
                "expected_kinds",
                "available_kinds",
                "used_kind",
                "total_size",
                "status",
            ]
        )

        for family in sorted(grouped.keys()):
            for test in sorted(grouped[family].keys()):
                for approach in approach_order:
                    expected = expected_kinds.get(approach, set())
                    available = available_kinds.get((test, approach), set())
                    record = grouped[family][test].get(approach)

                    used_kind = record.kind if record else ""
                    total_size = str(record.total_size) if record else ""

                    if not expected:
                        status = "not_expected"
                    elif expected.issubset(available):
                        status = "complete"
                    elif available:
                        status = "partial"
                    else:
                        status = "missing"

                    writer.writerow(
                        [
                            family,
                            test,
                            approach,
                            ";".join(sorted(expected)),
                            ";".join(sorted(available)),
                            used_kind,
                            total_size,
                            status,
                        ]
                    )


def print_summary(
    grouped: Dict[str, Dict[str, Dict[str, ResultRecord]]],
    available_kinds: Dict[Tuple[str, str], Set[str]],
    expected_kinds: Dict[str, Set[str]],
    approach_order: List[str],
) -> None:
    print("\n== Total ciphertext moduli size summary ==")
    for family in sorted(grouped.keys()):
        print(f"\n[{family}]")
        for test in sorted(grouped[family].keys()):
            parts = []
            for approach in approach_order:
                record = grouped[family][test].get(approach)
                value = str(record.total_size) if record else "NA"
                expected = expected_kinds.get(approach, set())
                available = available_kinds.get((test, approach), set())

                if not expected:
                    status = "(not-expected)"
                elif expected.issubset(available):
                    status = "(complete)"
                elif available:
                    status = "(partial)"
                else:
                    status = "(missing)"

                parts.append(f"{approach}={value}{status}")
            print(f"  {test}: " + ", ".join(parts))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize total ciphertext moduli sizes. Approach comparison uses execution, "
            "while kind comparison includes all available kinds per approach."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Directory with JSON result files (default: mlir-test-files/data)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "plots",
        help="Directory to write figures and summary CSV",
    )
    parser.add_argument(
        "--test-script",
        type=Path,
        default=Path(__file__).resolve().parent / "test.sh",
        help="Path to test.sh used to derive expected approach/kind availability",
    )

    args = parser.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {args.data_dir}")

    if not args.test_script.exists():
        raise FileNotFoundError(f"test.sh does not exist: {args.test_script}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = args.out_dir / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    approach_order = list(COMPARISON_APPROACH_ORDER)
    expected_kinds: Dict[str, Set[str]] = {
        approach: {APPROACH_COMPARISON_KIND} for approach in approach_order
    }

    selected, available_kinds, latest_by_kind = load_results(
        args.data_dir,
        approach_order,
        APPROACH_COMPARISON_KIND,
        APPROACH_EXECUTION_BACKEND,
    )
    if not selected:
        print("No matching approach result files found.")
        return 1

    grouped = group_by_family(selected)
    grouped_by_kind = group_by_family_kind(latest_by_kind)

    for family, family_data in sorted(grouped.items()):
        plot_family(family, family_data, approach_order, args.out_dir, pdf_dir)
        plot_family_kind_comparison(
            family,
            grouped_by_kind.get(family, {}),
            approach_order,
            args.out_dir,
            pdf_dir,
        )

    summary_csv = args.out_dir / "total_ciphertext_moduli_size_summary.csv"
    write_summary_csv(grouped, approach_order, summary_csv)

    availability_csv = args.out_dir / "availability_comparison.csv"
    write_availability_csv(
        grouped, available_kinds, expected_kinds, approach_order, availability_csv
    )

    print_summary(grouped, available_kinds, expected_kinds, approach_order)

    print(f"\nWrote {len(grouped)} figure(s) to: {args.out_dir}")
    print(f"Wrote PDF figures to: {pdf_dir}")
    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Wrote availability comparison CSV: {availability_csv}")

    script_algorithms = parse_test_script_algorithms(args.test_script)
    print(
        f"Compared approaches: {', '.join(approach_order)}; kind policy="
        f"all approaches->{APPROACH_COMPARISON_KIND}"
    )
    if script_algorithms:
        print(f"Algorithms found in {args.test_script.name}: {', '.join(script_algorithms)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
