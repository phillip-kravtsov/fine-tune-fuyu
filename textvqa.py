import json
import os
from collections import Counter
from typing import Dict
import random

from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import FuyuImageProcessor, FuyuProcessor

from config import TrainingConfig
from data import FuyuCollator


class TextVQADataset(Dataset):
    def __init__(self, json_path: str, images_dir: str, max_ids=None):
        with open(json_path, "r") as f:
            instances = json.load(f)["data"]
        self.instances = instances
        if max_ids is not None:
            random.shuffle(self.instances)
            self.instances = self.instances[:max_ids]
        self.images_dir = images_dir

    def __len__(self):
        return len(self.instances)

    @staticmethod
    def get_input_text(question: Dict):
        input_text = f"{question['question']}\n"
        return input_text

    def __getitem__(self, idx):
        instance = self.instances[idx]
        image_path = os.path.join(self.images_dir, f"{instance['image_id']}.jpg")
        image = Image.open(image_path).convert("RGB")
        answer = Counter(instance["answers"]).most_common(1)[0][0]
        return {
            "image": image,
            "text": TextVQADataset.get_input_text(instance),
            "target": answer,
        }


def get_data(config: TrainingConfig, tokenizer, world_size: int, local_rank: int):
    processor = FuyuProcessor(
        image_processor=FuyuImageProcessor(),
        tokenizer=tokenizer,
        add_beginning_of_answer_token=False,
    )
    processor.max_tokens_to_generate = 0
    train_dataset = TextVQADataset(
        json_path="/workspace/textvqa/train.json",
        images_dir="/workspace/textvqa/train_val_images",
    )
    collator = FuyuCollator(processor, config.train_on_questions)
    if config.use_packed_sampler:
        raise NotImplementedError("Packed sampler not implemented for TextVQA.")
    else:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.per_device_batch_size,
            collate_fn=collator,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
        )
        max_train_steps = len(train_dataloader)
    validation_dataloader = get_validation_dataloader(
        config, tokenizer, local_rank, world_size
    )
    return (train_dataloader, max_train_steps, validation_dataloader)


def get_validation_dataloader(config, tokenizer, local_rank, world_size):
    processor = FuyuProcessor(
        image_processor=FuyuImageProcessor(),
        tokenizer=tokenizer,
        add_beginning_of_answer_token=False,
    )
    processor.max_tokens_to_generate = 0
    validation_dataset = TextVQADataset(
        json_path="/workspace/textvqa/val.json",
        images_dir="/workspace/textvqa/train_val_images",
        max_ids=config.max_eval_ids,
    )
    validation_sampler = DistributedSampler(
        validation_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=False,
    )
    collator = FuyuCollator(processor, False)
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=config.eval_batch_size,
        collate_fn=collator,
        sampler=validation_sampler,
        num_workers=4,
        pin_memory=True,
    )
    return validation_dataloader
