import json
import re
from collections import defaultdict
from functools import reduce
import argparse
import numpy as np
from green_score import GREEN
model_name = "StanfordAIMI/GREEN-radllama2-7b"
green_scorer = GREEN(model_name, output_dir=".")

def compute_green(refs, hyps):
    green_scorer = GREEN(model_name, output_dir=".")
    mean, std, green_score_list, summary, result_df = green_scorer(refs, hyps)
    return {
        'mean': mean,
        'std': std,
        'reward_list': green_score_list,
        'summary': summary,
        'result': result_df
    }

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--json", type=str)
    parser.add_argument("--save_file_path", type=str, default=None)
    args = parser.parse_args()
    return args

args = parse_args()
data = json.load(open(args.json, "r"))
# data = json.load(open("./Qwen2.5-VL-3B-GRPO-RER-20250414_2.json", "r"))
res = defaultdict(list)
refs, hyps = [], []
for d in data:
    pred = d["model_output"]
    gt = d["ground_truth"]
    pattern = r"<answer>(.*?)</answer>"
    matches = re.search(pattern, pred, re.DOTALL)
    pred = matches.group(0) if matches else pred
    pattern = r"\[Error Type\]:\s*(.*?)(?=\n)"
    matches = re.search(pattern, pred, re.DOTALL)
    gt_m = re.search(pattern, gt, re.DOTALL).group(0).split(":")[-1].strip()
    if matches is None:
        res[gt_m].append(0.0)
        continue
    res[gt_m].append(1.0 if matches == gt_m else 0.0)
    matches = matches.group(0).split(":")[-1].strip()
    if matches != gt_m:
        print(matches, "|", gt_m)
    pattern = r"Correct Report\s*(.*?)(?=\[Error|$)"
    matches = re.search(pattern, pred, re.DOTALL)
    gt_m = re.search(pattern, gt, re.DOTALL).group(0).split(":")[-1].strip()
    if matches is None:
        matches = 'None'
    refs.append(gt_m)
    hyps.append(matches.group(0).split(":")[-1].strip())
    if len(refs) > 10:
        break
green_score = compute_green(refs, hyps)
print("GREENScore:")
print(green_score['reward_list'])
print(green_score['summary'])
for index, row in green_score['result'].iterrows():
    print(f"Row {index}:\n")
    for col_name in green_score['result'].columns:
        print(f"{col_name}: {row[col_name]}\n")
    print('-' * 80)