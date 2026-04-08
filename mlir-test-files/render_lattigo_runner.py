#!/usr/bin/env python3
import pathlib
import re
import sys


def main() -> int:
  if len(sys.argv) != 9:
    raise SystemExit(
        "usage: render_lattigo_runner.py <template> <output> <test_name> <selection_approach> <arity> <input_go> <expected_value> <params_output_path>"
    )

  template_path = pathlib.Path(sys.argv[1])
  output_path = pathlib.Path(sys.argv[2])
  test_name = sys.argv[3]
  selection_approach = sys.argv[4]
  arity = int(sys.argv[5])
  input_go = pathlib.Path(sys.argv[6])
  expected_value = int(sys.argv[7])
  params_output_path = sys.argv[8]

  template = template_path.read_text()
  input_text = input_go.read_text()

  encrypt_calls = []
  out_args = []
  for idx in range(arity):
    helper = "func__encrypt__arg0"
    if re.search(rf"^func\s+func__encrypt__arg{idx}\(", input_text, re.M):
      helper = f"func__encrypt__arg{idx}"
    encrypt_calls.append(
        f"  ct{idx} := {helper}(evaluator, param, encoder, encryptor, values[{idx}])"
    )
    out_args.append(f", ct{idx}")

  rendered = template
  rendered = rendered.replace("__ARITY__", str(arity))
  rendered = rendered.replace("__TEST_NAME__", test_name)
  rendered = rendered.replace("__SELECTION_APPROACH__", selection_approach)
  rendered = rendered.replace("__PARAMS_OUTPUT_PATH__", params_output_path)
  rendered = rendered.replace("__EXPECTED_VALUE__", str(expected_value))
  rendered = rendered.replace("__ENCRYPT_CALLS__", "\n".join(encrypt_calls))
  rendered = rendered.replace("__OUT_ARGS__", "".join(out_args))

  output_path.write_text(rendered)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
