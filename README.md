# CorBenchX

**CorBenchX** is a large-scale synthetic chest X-Ray error dataset and vision–language benchmark for report error detection and correction. It contains paired “clean” (original) and “error-injected” reports, with detailed annotations of error types, and descriptions.

CorBenchX is derived via synthetic error injection from the de-identified MIMIC-CXR v2.0.0 dataset under the PhysioNet Credentialed Health Data License v1.5.0.

## Dataset Structure

```bash
CorBenchX/
└── multi_error/
│ ├── multi_error_report/
│ │ ├── p18/
│ │ │ ├── p18000291/
│ │ │ │ ├── s55388853.txt # error-injected report
│ │ │ │ └── …
│ │ │ └── …
│ │ └── …
│ ├── multi_error.json
├── single_error/
│ ├── single_error_report_part1/
│ │ ├── p10/
│ │ │ ├── p10000764/
│ │ │ │ ├── s57375967.txt # error-injected report
│ │ │ │ └── …
│ │ │ └── …
│ │ └── …
│ ├── single_error_report_part2/
│ ├── train.json
│ └── test.json
├── demo
│ ├── demo.py/


```

- **`single_error/single_error_report_part1/`** contains error-injected `.txt` files:
- **`multi_errors/multi_error_report/`** follows the same naming but each file may contain 2–3 injected errors.
- **`train.json`** and **`test.json`** partition each split.
- **`demo/demo.py`** provides an example of how to leverage this dataset to prompt models for error detection and correction.

## JSON Format

Each entry in `train.json` and `test.json` is a JSON object with the following fields:

```json
{
  "image_path": "physionet.org/files/mimic-cxr-jpg/2.0.0/files/p18/p18079244/s58587528/c6ee601e-5178e3ed-18fd0aee-92ffd231-940e5cad.jpg",
  "input_report": "FINDINGS: ... Consolodation at the left base is not excluded. IMPRESSION: ---",
  "output_report": "FINDINGS: ... Consolidation at the left base is not excluded. IMPRESSION: ---",
  "error_type": "Spelling Error",
  "error_description": "Misspelling \"Consolidation\" as \"Consolodation\" in the FINDINGS section."
}
```
- **`image_path`** — Path to the AP/PA chest X-ray image.
- **`input_report`** — The error-injected report text.
- **`output_report`** — The original, error-free report text.
- **`error_type`** — One of {Omission, Insertion, Spelling Error, Side Confusion, Other}.
- **`error_description`** — A concise explanation of the injected error.