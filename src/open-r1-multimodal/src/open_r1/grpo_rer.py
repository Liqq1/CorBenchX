# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import os
import csv
import re
import datetime
from dataclasses import dataclass, field
from typing import Optional
import pathlib
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration
import wandb

from math_verify import parse, verify
from open_r1.trainer import VLMGRPOTrainer, GRPOConfig, VLMGRPOTrainer
from open_r1.vlm_modules import *
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
import yaml
import json
import random
import math

from open_r1.qwen2_5vl_monkey_patch import monkey_patch_qwen2_5vl_flash_attn, monkey_patch_qwen2_5vl_forward
monkey_patch_qwen2_5vl_flash_attn()


# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "bleu"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args: GRPOScriptArguments, question_template: str):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []
        self.question_template = question_template

        if data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999

                for data in datasets:
                    json_path = data.get("json_path")
                    sampling_strategy = data.get("sampling_strategy", "all")
                    sampling_number = None

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]
                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

        # self.question_template = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
        self.question_template = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."

        # step1
        self.input_template = """Find the errors in the following report, and provide the error type. The report may inadvertently contain common errors in the following 5 categories:
1. Omission: The omission of relevant words or expressions, including deletions or missing words (e.g., "fracture" instead of "no fracture").
2. Insertion: The unintentional insertion of incorrect words or expressions, including inappropriate words, wrong word substitutions, or extra words (e.g., "abnormal" instead of "normal").
3. Spelling Error: Spelling mistakes or word truncations due to manual text processing errors (e.g., "pnuemothorax" instead of "pneumothorax").
4. Side Confusion: Errors involving laterality or orientation (e.g., "right" instead of "left," or "lateral" instead of "medial").
5. Other: Includes errors in units of measurement, punctuation and hyphen (e.g., Incorrect unit substitution from "cm" to "mm", the nodule size was erroneously reported as 1.5 mm instead of 1.5 cm; missing period at the end of the sentence; "post diuresis" instead of "post-diuresis").

Report:
{}
Output format:
[Error Type]: Fill in one of the following: "Omission, Insertion, Spelling Error, Side Confusion, Other".
"""

        # step2
        self.input_template2 = """Based on the identified error type{}, provide the error description.
Output format:
[Error Description]: Describe the identified error.
"""

        # step3
        self.input_template3 = """Based on the identified error type{} and error description{}, provide a correct report.
Output format:
[Correct Report]: Revise the original report and provide the corrected version.
"""

        self.output_template = """[Error Type]: {}
[Error Description]: {}
[Correct Report]: {}
"""

        self.output_template1 = """[Error Type]: {}"""

        self.output_template2 = """[Error Description]: {}"""

        self.output_template3 = """[Correct Report]: {}
"""

        self.image_path_prefix = "physionet.org/files/mimic-cxr-jpg/2.0.0/files/"

        mimic_metadata_path = "data/mimic-cxr-2.0.0-metadata_fixed.csv"
        self.dicom_view_dict = {}
        with open(mimic_metadata_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                dicom_id = row['dicom_id']
                view_position = row['ViewPosition']
                self.dicom_view_dict[dicom_id] = view_position

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        # Format into conversation
        def make_conversation(example):
            return {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["problem"]},
                ],
            }
        QUESTION_TEMPLATE = self.question_template
        def make_conversation_image(example):
            return {
                "prompt": [
                    # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                        ],
                    }
                ],
            }

        example = self.list_data_dict[i]
        image_root = self.script_args.image_root

        example["image"] = example["image_path"]

        if 'image' in example:
            image_path = os.path.join(image_root, example['image'])

            # In case the image is not found
            while not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, randomly selecting another image")
                new_index = random.randint(0, len(self.list_data_dict)-1)
                example = self.list_data_dict[new_index]
                image_path = os.path.join(image_root, example['image'])
            image = Image.open(image_path).convert("RGB")
            image = image.resize((336, 336))
        else:
            image = None


        example["problem"] = self.input_template.format(example['input_report'])
        example["solution"] = self.output_template.format(example['error_type'], example['error_description'], example['output_report'])

        return {
            'image': image,
            'problem': example["problem"],
            'solution': example["solution"],
            "input_report": example['input_report'],
            "question_template": self.question_template,
            "input_template_step2": self.input_template2,
            "input_template_step3": self.input_template3,
            "gt_step1": self.output_template1.format(example['error_type']),
            "gt_step2": self.output_template2.format(example['error_description']),
            "gt_step3": self.output_template3.format(example['output_report']),
            'prompt': make_conversation_image(example)['prompt'] if 'image' in example else make_conversation(example)['prompt'],
        }


def get_vlm_module(model_name_or_path):
    if "qwen" in model_name_or_path.lower():
        return Qwen2VLModule
    elif "internvl" in model_name_or_path.lower():
        return InvernVLModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")

def is_dist_avail_and_initialized():
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return torch.distributed.get_rank()

def main(script_args, training_args, model_args):
    # wandb
    if os.environ.get("WANDB_MODE") != "disabled" and get_rank() == 0:
        now = datetime.datetime.now()
        exp_name = now.strftime("%Y_%m_%d_%H_%M_%S")
        wandb.init(
            project="rer",
            config=training_args.to_dict(),
            name=f"{exp_name}",
            dir=training_args.output_dir,
        )

    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)

    # Load the reward functions
    reward_funcs_registry = {
        "accuracy": vlm_module_cls.accuracy_reward_step,
        "bleu": vlm_module_cls.bleu_reward_step,
        "format": vlm_module_cls.format_reward_rer,
    }
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    # Load the dataset
    dataset = LazySupervisedDataset(script_args.dataset_name, script_args, question_template=vlm_module_cls.get_question_template(task_type="rec"))

    trainer_cls = VLMGRPOTrainer
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
        torch_dtype=model_args.torch_dtype,
    )

    if list(pathlib.Path(training_args.output_dir).glob('checkpoint-*')):
        if get_rank() == 0:
            print('resume', '---' * 200)
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    if training_args.deepspeed and "zero3" in training_args.deepspeed:
        print("zero3 is used, qwen2_5vl forward monkey patch is applied")
        monkey_patch_qwen2_5vl_forward()
    main(script_args, training_args, model_args)
