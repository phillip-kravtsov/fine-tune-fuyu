from typing import Dict

import datasets
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import FuyuImageProcessor, FuyuProcessor

from config import TrainingConfig
from .data import FuyuCollator, FuyuMultiAnswerCollator


class ScienceQADataset(Dataset):
    def __init__(self, dataset: datasets.Dataset, images_only: bool = True):
        self.images_only = images_only
        if self.images_only:
            self.dataset = [d for d in tqdm(dataset) if d["image"] is not None]
        else:
            self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def get_input_text(question: Dict):
        input_text = f"Answer the following multiple choice question: {question['question']}\nChoose from the following answers:\n"
        for answer in question["choices"]:
            input_text += f"{answer}\n"
        return input_text

    def __getitem__(self, idx):
        question = self.dataset[idx]
        image = question["image"].convert("RGB")
        return {
            "image": image,
            "text": ScienceQADataset.get_input_text(question),
            "target": question["choices"][question["answer"]],
        }


class MultiAnswerScienceQADataset(Dataset):
    def __init__(self, dataset: datasets.Dataset, images_only: bool = True):
        self.images_only = images_only
        if self.images_only:
            self.dataset = [d for d in tqdm(dataset) if d["image"] is not None]
        else:
            self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        question = self.dataset[idx]
        image = question["image"].convert("RGB")
        return {
            "image": image,
            "text": ScienceQADataset.get_input_text(question),
            "answers": question["choices"],
            "correctAnswer": question["answer"],
        }


def get_data(config: TrainingConfig, world_size: int, local_rank: int, tokenizer):
    processor = FuyuProcessor(
        image_processor=FuyuImageProcessor(),
        tokenizer=tokenizer,
        add_beginning_of_answer_token=False,
    )
    processor.max_tokens_to_generate = 0
    full_dataset = datasets.load_dataset("derek-thomas/ScienceQA")
    train_dataset = ScienceQADataset(full_dataset["train"], True)
    collator = FuyuCollator(processor, config.train_on_questions)
    if config.use_packed_sampler:
        raise NotImplementedError("Packed sampler not implemented for ScienceQA.")
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
            num_workers=2,
            pin_memory=True,
        )
        max_train_steps = len(train_dataloader)
    validation_dataloader = get_validation_dataloader(
        config, tokenizer, local_rank, world_size, False, full_dataset
    )
    multiple_choice_dataloader = get_validation_dataloader(
        config, tokenizer, local_rank, world_size, True, full_dataset
    )
    return (
        train_dataloader,
        max_train_steps,
        validation_dataloader,
        multiple_choice_dataloader,
    )


def get_validation_dataloader(
    config, tokenizer, local_rank, world_size, multi_answer=False, full_dataset=None
):
    processor = FuyuProcessor(
        image_processor=FuyuImageProcessor(),
        tokenizer=tokenizer,
        add_beginning_of_answer_token=False,
    )
    processor.max_tokens_to_generate = 0
    if full_dataset is None:
        full_dataset = datasets.load_dataset("derek-thomas/ScienceQA")
    if multi_answer:
        validation_dataset = MultiAnswerScienceQADataset(
            full_dataset["validation"], True
        )
    else:
        validation_dataset = ScienceQADataset(full_dataset["validation"], True)
    validation_sampler = DistributedSampler(
        validation_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=False,
    )
    collator = (
        FuyuCollator(processor, False)
        if not multi_answer
        else FuyuMultiAnswerCollator(processor)
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=config.eval_batch_size if not multi_answer else 1,
        collate_fn=collator,
        sampler=validation_sampler,
        # num_workers=2,
        pin_memory=True,
    )
    return validation_dataloader
