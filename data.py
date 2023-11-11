import copy
import json
import math
import os
import random
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import FuyuImageProcessor, FuyuProcessor
from transformers.models.fuyu.processing_fuyu import BEGINNING_OF_ANSWER_STRING

import utils
from config import Config
from sampler import PackedDistributedBatchSampler

@dataclass
class FuyuCollator(object):
    processor: FuyuProcessor
    train_on_inputs: bool

    def __call__(self, instances):
        images = [instance["image"] for instance in instances]
        texts = [
            instance["text"]
            + BEGINNING_OF_ANSWER_STRING
            + instance["target"]
            + self.processor.tokenizer.eos_token
            for instance in instances
        ]
        collated = self.processor(images=images, text=texts)
        batch_size = collated["input_ids"].shape[0]

        labels = copy.deepcopy(collated["input_ids"])
        labels = torch.where(collated["attention_mask"].bool(), labels, -100)
        for i in range(batch_size):
            target_size = (
                len(
                    self.processor.tokenizer(
                        instances[i]["target"], add_special_tokens=False
                    )
                )
                + 2
            )
            if self.train_on_inputs:
                target_size += (
                    len(
                        self.processor.tokenizer(
                            instances[i]["text"], add_special_tokens=False
                        )
                    )
                    + 1
                )
            labels[i, :-target_size] = -100

        collated["labels"] = labels
        if "is_correct" in instances[0]:
            collated["is_correct"] = torch.tensor(
                [instance["is_correct"] for instance in instances]
            )
            collated["question_id"] = torch.tensor(
                [instance["question_id"] for instance in instances], dtype=torch.long
            )
        return collated

def get_padding_dims(image_height, image_width, image_processor):
    padding_bottom = image_processor.size["height"] - image_height
    padding_right = image_processor.size["width"] - image_width
    return image_height + padding_bottom, image_width + padding_right

def get_scale_dims(image_height, image_width, image_processor):
    target_width, target_height = image_processor.size["width"], image_processor.size[
        "height"
    ]
    if image_width <= target_width and image_height <= target_height:
        return image_height, image_width

    height_scale_factor = target_height / image_height
    width_scale_factor = target_height / image_width
    optimal_scale_factor = min(height_scale_factor, width_scale_factor)

    new_height = int(image_height * optimal_scale_factor)
    new_width = int(image_width * optimal_scale_factor)
    return new_height, new_width
