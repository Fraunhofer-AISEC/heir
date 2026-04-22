#!/usr/bin/env python3
import pathlib
import re
import sys

VALUES_INIT_ALL = """\
  values := make([][]int16, {arity})
  for i := 0; i < {arity}; i++ {{
    values[i] = make([]int16, 8)
    for j := 0; j < 8; j++ {{
      values[i][j] = 1
    }}
  }}"""

VALUES_INIT_FIRST_ONLY = """\
  values := make([][]int16, {arity})
  for i := 0; i < {arity}; i++ {{
    values[i] = make([]int16, 8)
    if i == 0 {{
      for j := 0; j < 8; j++ {{
        values[i][j] = 1
      }}
    }}
  }}"""

def main() -> int:
  if len(sys.argv) != 10:
    raise SystemExit(
        "usage: render_lattigo_runner.py <template> <output> <test_name> <selection_approach> <arity> <input_go> <expected_value> <params_output_path> <values_mode: all|first_only>"
    )

  template_path = pathlib.Path(sys.argv[1])
  output_path = pathlib.Path(sys.argv[2])
  test_name = sys.argv[3]
  selection_approach = sys.argv[4]
  arity = int(sys.argv[5])
  input_go = pathlib.Path(sys.argv[6])
  expected_value = int(sys.argv[7])
  params_output_path = sys.argv[8]
  values_mode = sys.argv[9]

  if values_mode not in ("all", "first_only"):
    raise SystemExit("values_mode must be 'all' or 'first_only'")

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

  values_template = VALUES_INIT_ALL if values_mode == "all" else VALUES_INIT_FIRST_ONLY
  values_init = values_template.format(arity=arity)

  rendered = template
  rendered = rendered.replace("__ARITY__", str(arity))
  rendered = rendered.replace("__TEST_NAME__", test_name)
  rendered = rendered.replace("__SELECTION_APPROACH__", selection_approach)
  rendered = rendered.replace("__PARAMS_OUTPUT_PATH__", params_output_path)
  rendered = rendered.replace("__EXPECTED_VALUE__", str(expected_value))
  rendered = rendered.replace("__ENCRYPT_CALLS__", "\n".join(encrypt_calls))
  rendered = rendered.replace("__OUT_ARGS__", "".join(out_args))
  rendered = rendered.replace("__VALUES_INIT__", values_init)

  output_path.write_text(rendered)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
