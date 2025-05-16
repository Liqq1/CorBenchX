import json
import re
from collections import defaultdict
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model predictions with accuracy, precision, and recall.")
    parser.add_argument("--json", type=str, required=True, help="Path to the input JSON file.")
    return parser.parse_args()

args = parse_args()
data = json.load(open(args.json, "r"))

# Stats containers
tp = defaultdict(int)
fp = defaultdict(int)
fn = defaultdict(int)
acc_data = defaultdict(list)

all_gt = []
all_pred = []

for d in data:
    pred = d["model_output"]
    gt = d["ground_truth"]

    # Extract from <answer>
    match_answer = re.search(r"<answer>(.*?)</answer>", pred, re.DOTALL)
    pred = match_answer.group(0) if match_answer else pred

    # Extract [Error Type]
    pattern = r"\[Error Type\]:\s*(.*?)(?=\n)"
    pred_match = re.search(pattern, pred, re.DOTALL)
    gt_match = re.search(pattern, gt, re.DOTALL)

    if not gt_match:
        continue

    gt_label = gt_match.group(0).split(":")[-1].strip()

    if pred_match:
        pred_label = pred_match.group(0).split(":")[-1].strip()
    else:
        pred_label = None

    all_gt.append(gt_label)
    all_pred.append(pred_label)

    # For accuracy
    acc_data[gt_label].append(1.0 if pred_label == gt_label else 0.0)

    # For precision/recall
    if pred_label == gt_label:
        tp[gt_label] += 1
    else:
        if pred_label is not None:
            fp[pred_label] += 1
        fn[gt_label] += 1

# Overall accuracy
correct = sum(tp.values())
total = len(all_gt)
print("Total Accuracy:", correct / total)

# Per-class metrics
print("\nPer-class metrics:")
for label in sorted(set(all_gt + [p for p in all_pred if p])):
    acc_list = acc_data.get(label, [])
    acc = sum(acc_list) / len(acc_list) if acc_list else 0.0

    true_pos = tp[label]
    false_pos = fp[label]
    false_neg = fn[label]

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0

    print(f"{label}:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")

# Macro averages
all_labels = set(all_gt + [p for p in all_pred if p])
macro_precision = sum(
    tp[l] / (tp[l] + fp[l]) if (tp[l] + fp[l]) > 0 else 0.0 for l in all_labels
) / len(all_labels)
macro_recall = sum(
    tp[l] / (tp[l] + fn[l]) if (tp[l] + fn[l]) > 0 else 0.0 for l in all_labels
) / len(all_labels)

print(f"\nMacro Precision: {macro_precision:.4f}")
print(f"Macro Recall:    {macro_recall:.4f}")
