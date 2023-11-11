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
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import FuyuImageProcessor, FuyuProcessor

import data as fuyu_data
import utils
from config import Config
from sampler import PackedDistributedBatchSampler

AI2D_DATA_DIR = "/workspace/ai2d"
FONT_PATH = "/workspace/Arial.ttf"


@dataclass
class MultipleChoiceQuestion(object):
    image_path: str
    question: str
    answers: List[str]
    correct_answer: int
    question_id: int


def get_int_question_id(question_id: str) -> int:
    image_id, suffix = question_id.split(".")
    return int(image_id) * 100 + int(suffix.split("-")[-1])


def get_image_id(question: MultipleChoiceQuestion):
    return question.image_path.split("/")[-1].split(".")[0]


def get_ai2d_questions(root_dir: str, skip_abc=False) -> List[MultipleChoiceQuestion]:
    questions_dir = os.path.join(root_dir, "questions")
    per_image_data = []
    for path in sorted(os.listdir(questions_dir)):
        with open(os.path.join(questions_dir, path), "r") as f:
            per_image_data.append(json.load(f))

    questions = []
    images_dir = os.path.join(root_dir, "images")
    for image_data in per_image_data:
        image_name = image_data["imageName"]
        image_path = os.path.join(images_dir, image_name)
        image_abc_path = os.path.join(
            images_dir, image_name.replace(".png", "-abc.png")
        )
        for question_text, question_data in image_data["questions"].items():
            is_abc = question_data["abcLabel"]
            if skip_abc and is_abc:
                continue
            question_id = get_int_question_id(question_data["questionId"])
            questions.append(
                MultipleChoiceQuestion(
                    image_path=image_abc_path if is_abc else image_path,
                    question=question_text,
                    answers=question_data["answerTexts"],
                    correct_answer=question_data["correctAnswer"],
                    question_id=question_id,
                )
            )
    return questions


def get_input_text(question: MultipleChoiceQuestion):
    input_text = f"Answer the following multiple choice question: {question.question}\nPossible answers are:\n"
    for answer in question.answers:
        input_text += f"{answer}\n"
    return input_text


def get_ai2d_test_ids():
    test_ids_csv_path = os.path.join(AI2D_DATA_DIR, "ai2d_test_ids.csv")
    with open(test_ids_csv_path, "r") as f:
        test_ids = [line.strip() for line in f.readlines()]
    return sorted(test_ids)


def add_labels_to_model_inputs(model_inputs, tokenizer, target):
    input_ids = model_inputs["input_ids"].squeeze()
    target_ids = tokenizer.encode(
        target + tokenizer.eos_token,
        add_special_tokens=False,
        return_tensors="pt",
    ).squeeze()
    all_ids = torch.concat([input_ids, target_ids])
    labels = copy.deepcopy(all_ids)
    labels[: input_ids.shape[0]] = -100
    model_inputs["input_ids"] = all_ids
    model_inputs["labels"] = labels


class AI2DMultipleChoiceDataset(Dataset):
    def __init__(
        self,
        questions: List[MultipleChoiceQuestion],
        processor: FuyuProcessor,
        include_labels=True,
    ):
        self.processor = processor
        self.include_labels = include_labels
        self.questions = questions

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        q = self.questions[idx]
        image = Image.open(q.image_path).convert("RGB")
        return {
            "image": image,
            "text": get_input_text(q),
            "target": q.answers[q.correct_answer],
            "is_correct": True,
            "question_id": q.question_id,
        }

    # not really ai2d specific... to move
    def estimate_input_size(self, idx):
        q = self.questions[idx]
        w, h = Image.open(q.image_path).size
        patch_h, patch_w = 30, 30
        input = self.processor.tokenizer.encode(get_input_text(q))
        target = self.processor.tokenizer.encode(
            q.answers[q.correct_answer], add_special_tokens=False
        )
        padded_h, padded_w = fuyu_data.get_padding_dims(
            *fuyu_data.get_scale_dims(h, w, self.processor.image_processor),
            self.processor.image_processor
        )
        new_h = min(padded_h, math.ceil(h / patch_h) * patch_h)
        new_w = min(padded_w, math.ceil(w / patch_w) * patch_w)
        num_patches = self.processor.image_processor.get_num_patches(
            new_h, new_w, {"height": patch_h, "width": patch_w}
        )
        return len(input) + len(target) + 2 + new_h // patch_h + num_patches


class AI2DMultipleAnswerDataset(Dataset):
    def __init__(
        self,
        questions: List[MultipleChoiceQuestion],
        processor: FuyuProcessor,
        include_labels=True,
    ):
        self.include_labels = include_labels
        self.processor = processor
        self.questions: List[MultipleChoiceQuestion] = questions
        self.length = sum([len(q.answers) for q in self.questions])
        self.answer_idx_to_question_idx = [
            (i, j) for i, q in enumerate(self.questions) for j in range(len(q.answers))
        ]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        question_idx, answer_idx = self.answer_idx_to_question_idx[idx]
        q: MultipleChoiceQuestion = self.questions[question_idx]
        image = Image.open(q.image_path).convert("RGB")
        return {
            "image": image,
            "text": get_input_text(q),
            "target": q.answers[answer_idx],
            "is_correct": answer_idx == q.correct_answer,
            "question_id": q.question_id,
        }


def get_data(config: Config, world_size: int, local_rank: int, tokenizer):
    # Cache vocab to fix performance issues.
    vocab = tokenizer.get_vocab()
    tokenizer.get_vocab = lambda: vocab

    processor = FuyuProcessor(
        image_processor=FuyuImageProcessor(),
        tokenizer=tokenizer,
        add_beginning_of_answer_token=False,
    )
    processor.max_tokens_to_generate = 0
    test_ids = get_ai2d_test_ids()
    random.seed(config.seed)
    random.shuffle(test_ids)
    if config.max_eval_ids is not None:
        test_ids = test_ids[: config.max_eval_ids]
    test_ids = set(test_ids)

    all_questions = get_ai2d_questions(AI2D_DATA_DIR, skip_abc=config.skip_abc)
    test_questions = [q for q in all_questions if get_image_id(q) in test_ids]
    train_questions = [q for q in all_questions if get_image_id(q) not in test_ids]
    if local_rank == 0:
        print(
            f"There are {len(train_questions)} training questions and {len(test_questions)} test questions."
        )
    train_dataset = AI2DMultipleChoiceDataset(train_questions, processor)
    val_dataset = AI2DMultipleChoiceDataset(test_questions, processor)
    multiple_answer_dataset = AI2DMultipleAnswerDataset(test_questions, processor)
    data_collator = fuyu_data.FuyuCollator(
        processor, train_on_inputs=config.train_on_questions
    )
    if config.use_packed_sampler:
        print("loading packed sampler--getting lengths")
        lengths = np.array(
            [train_dataset.estimate_input_size(i) for i in range(len(train_dataset))]
        )
        batch_sampler = PackedDistributedBatchSampler(
            batch_max_length=(max(lengths) + 1) * config.per_device_batch_size,
            lengths=lengths,
            num_replicas=world_size,
            rank=local_rank,
            seed=config.seed,
        )
        train_dataloader = DataLoader(
            train_dataset,
            collate_fn=data_collator,
            pin_memory=True,
            batch_sampler=batch_sampler,
            num_workers=4,
        )
        max_train_steps = torch.tensor(len(train_dataloader)).long().to(local_rank)
        dist.all_reduce(max_train_steps, op=dist.ReduceOp.MIN)
        max_train_steps = max_train_steps.cpu().item()
    else:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True,
            seed=config.seed,
        )
        train_dataloader = DataLoader(
            train_dataset,
            collate_fn=data_collator,
            batch_size=config.per_device_batch_size,
            pin_memory=True,
            num_workers=4,
            sampler=train_sampler,
        )
        max_train_steps = len(train_dataloader)

    multiple_choice_sampler = DistributedSampler(
        multiple_answer_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=False,
        seed=config.seed,
    )

    multiple_choice_dataloader = DataLoader(
        multiple_answer_dataset,
        batch_size=config.eval_batch_size,
        collate_fn=fuyu_data.FuyuCollator(processor, False),
        pin_memory=True,
        sampler=multiple_choice_sampler,
        worker_init_fn=utils.seed_worker,
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=False,
        seed=config.seed,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        collate_fn=fuyu_data.FuyuCollator(processor, False),
        pin_memory=True,
        sampler=val_sampler,
        worker_init_fn=utils.seed_worker,
    )

    return (
        train_dataloader,
        max_train_steps,
        val_dataloader,
        multiple_choice_dataloader,
    )


def replace_text_and_save(base_path, question: MultipleChoiceQuestion):
    j = int(question.image_path.replace("-abc", "").split(".")[0].split("/")[-1])
    impath = os.path.join(base_path, "images", f"{j}.png")
    impath_abc = os.path.join(base_path, "images", f"{j}-abc.png")
    if os.path.exists(impath_abc):
        return
    anpath = os.path.join(base_path, "annotations", f"{j}.png.json")
    qpath = os.path.join(base_path, "questions", f"{j}.png.json")
    with open(anpath, "r") as f:
        annotation = json.load(f)
    with open(qpath, "r") as f:
        question = json.load(f)
    img = Image.open(impath).convert("RGB")
    text_annotation = annotation["text"]
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    for value in text_annotation.values():
        top_left = tuple(value["rectangle"][0])
        bottom_right = tuple(value["rectangle"][1])
        rectangle_height = bottom_right[1] - top_left[1]
        if rectangle_height == 0:
            print(text_annotation)
            continue
        font = ImageFont.truetype(FONT_PATH, rectangle_height)
        draw.rectangle([top_left, bottom_right], fill=(255, 255, 255))
        replacement_text = value["replacementText"]
        draw.text(top_left, replacement_text, font=font, fill=(0, 0, 0))
    img.save(impath_abc)


def create_overlay_images():
    questions = get_ai2d_questions(AI2D_DATA_DIR, False)
    for question in tqdm(questions):
        replace_text_and_save(AI2D_DATA_DIR, question)


if __name__ == "__main__":
    create_overlay_images()
