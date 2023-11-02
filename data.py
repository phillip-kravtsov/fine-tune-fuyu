import copy
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Set

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
from tqdm import tqdm
from transformers import FuyuImageProcessor, FuyuProcessor

import utils
from config import Config

AI2D_DATA_DIR = "/workspace/ai2d"
FONT_PATH = "/workspace/Arial.ttf"


@dataclass
class MultipleChoiceQuestion(object):
    image_path: str
    question: str
    answers: List[str]
    correct_answer: int
    question_id: int


def get_question_id(question_id_string):
    image_id, suffix = question_id_string.split(".")
    question_id = int(image_id) * 100 + int(suffix.split("-")[-1])
    return question_id


def get_ai2d_questions(
    root_dir: str, question_ids: Optional[Set[str]], skip_abc=False
) -> List[MultipleChoiceQuestion]:
    questions = []
    questions_dir = os.path.join(root_dir, "questions")
    images_dir = os.path.join(root_dir, "images")
    image_jsons = []
    for path in sorted(os.listdir(questions_dir)):
        with open(os.path.join(questions_dir, path), "r") as f:
            image_jsons.append(json.load(f))
    for data in image_jsons:
        image_name = data["imageName"]
        image_path = os.path.join(images_dir, image_name)
        image_abc_path = os.path.join(
            images_dir, image_name.replace(".png", "-abc.png")
        )
        for question_text, question_data in data["questions"].items():
            is_abc = question_data["abcLabel"]
            if is_abc and skip_abc:
                continue
            # hack to get an integral id
            question_id = get_question_id(question_data["questionId"])
            if question_ids is not None and question_id not in question_ids:
                continue
            questions.append(
                MultipleChoiceQuestion(
                    image_path=image_path if not is_abc else image_abc_path,
                    question=question_text,
                    answers=question_data["answerTexts"],
                    correct_answer=question_data["correctAnswer"],
                    question_id=question_id,
                )
            )
    return questions


def get_input_text(question: MultipleChoiceQuestion):
    input_text = f"Answer the following multiple choice question: {question.question}\nPossible answers are:"
    for answer in question.answers:
        input_text += f"{answer}\n"
    return input_text


def get_ai2d_test_ids():
    test_ids_csv_path = os.path.join(AI2D_DATA_DIR, "ai2d_test_ids.csv")
    with open(test_ids_csv_path, "r") as f:
        test_ids = [line.strip() for line in f.readlines()]
    return test_ids


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
        root_dir: str,
        processor: FuyuProcessor,
        skip_abc=False,
        include_labels=True,
    ):
        self.questions: List[MultipleChoiceQuestion] = []
        self.processor = processor
        self.include_labels = include_labels
        self.skip_abc = skip_abc
        self._init_questions(root_dir)

    def _init_questions(self, root_dir):
        self.questions = get_ai2d_questions(root_dir, None, skip_abc=self.skip_abc)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx: int):
        q = self.questions[idx]
        image = Image.open(q.image_path).convert("RGB")
        input_text = get_input_text(q)
        model_inputs = self.processor(images=image, text=input_text)
        if self.include_labels:
            target = q.answers[q.correct_answer]
            add_labels_to_model_inputs(model_inputs, self.processor.tokenizer, target)
            model_inputs["is_correct"] = True
            model_inputs["question_id"] = q.question_id
        return model_inputs

    def split(self, test_ids: List[str]):
        image_to_question_indices: OrderedDict[str, List[int]] = OrderedDict()
        for i, question in enumerate(self.questions):
            image_id = question.image_path.split("/")[-1].split(".")[0]
            image_to_question_indices.setdefault(image_id, []).append(i)
        images = list(image_to_question_indices.keys())
        train_ids = [im for im in images if im not in test_ids]
        train_idxs: List[int] = []
        test_idxs: List[int] = []
        for im in train_ids:
            train_idxs.extend(image_to_question_indices[im])
        for im in test_ids:
            if im not in image_to_question_indices:
                continue
            test_idxs.extend(image_to_question_indices[im])
        test_question_ids = [self.questions[i].question_id for i in test_idxs]
        return (
            Subset(self, train_idxs),
            Subset(self, test_idxs),
            test_question_ids,
        )


class AI2DDatasetForAutoEval(Dataset):
    def __init__(
        self,
        root_dir: str,
        processor: FuyuProcessor,
        question_ids: List[str],
        include_labels=True,
        skip_abc=False,
    ):
        self.question_ids = set(question_ids)
        self.include_labels = include_labels
        self.questions: List[MultipleChoiceQuestion] = []
        self.processor = processor
        self.skip_abc = skip_abc
        self._init_questions(root_dir)
        self.length = sum([len(q.answers) for q in self.questions])
        self.answer_idx_to_question_idx = [
            (i, j) for i, q in enumerate(self.questions) for j in range(len(q.answers))
        ]

    def _init_questions(self, root_dir):
        self.questions = get_ai2d_questions(
            root_dir, self.question_ids, skip_abc=self.skip_abc
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        question_idx, answer_idx = self.answer_idx_to_question_idx[idx]
        q = self.questions[question_idx]
        image = Image.open(q.image_path).convert("RGB")
        input_text = get_input_text(q)
        model_inputs = self.processor(images=image, text=input_text)
        if self.include_labels:
            target = q.answers[answer_idx]
            add_labels_to_model_inputs(model_inputs, self.processor.tokenizer, target)
            model_inputs["is_correct"] = q.correct_answer == answer_idx
            model_inputs["question_id"] = q.question_id
        return model_inputs


@dataclass
class DataCollatorForMultimodal(object):
    pad_token_id: int

    def __call__(self, instances):
        collated = {}
        pad_values = {
            "input_ids": -1,
            "image_patches": 0,
            "image_patches_indices": -1,
            "labels": -100,
        }
        for key in ["input_ids", "labels", "image_patches", "image_patches_indices"]:
            if key in instances[0]:
                values = [instance[key].squeeze() for instance in instances]
                collated[key] = pad_sequence(
                    values, batch_first=True, padding_value=pad_values[key]
                )
        attention_mask = collated["input_ids"].ne(pad_values["input_ids"])
        # Fuyu does not have a pad token id, so we don't want to overwrite
        # the zero token.
        collated["input_ids"][~attention_mask] = self.pad_token_id
        collated["attention_mask"] = attention_mask
        if "is_correct" in instances[0]:
            collated["is_correct"] = torch.tensor(
                [instance["is_correct"] for instance in instances]
            )
            collated["question_id"] = torch.tensor(
                [instance["question_id"] for instance in instances]
            ).long()
        return collated


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
        font = ImageFont.truetype(FONT_PATH, rectangle_height)
        draw.rectangle([top_left, bottom_right], fill=(255, 255, 255))
        replacement_text = value["replacementText"]
        draw.text(top_left, replacement_text, font=font, fill=(0, 0, 0))
    img.save(impath_abc)


def get_data(config: Config, world_size, local_rank, tokenizer):
    # Cache vocab for performance
    vocab = tokenizer.get_vocab()
    tokenizer.get_vocab = lambda: vocab
    processor = FuyuProcessor(
        image_processor=FuyuImageProcessor(),
        tokenizer=tokenizer,
    )
    # This is only for training.
    processor.max_tokens_to_generate = 0

    test_ids = get_ai2d_test_ids()
    if config.max_eval_ids is not None:
        test_ids = test_ids[: config.max_eval_ids]
    full_ds = AI2DMultipleChoiceDataset(
        AI2D_DATA_DIR, processor, skip_abc=config.skip_abc
    )
    train_dataset, _, test_question_ids = full_ds.split(test_ids)
    dataset_for_auto_eval = AI2DDatasetForAutoEval(
        AI2D_DATA_DIR, processor, test_question_ids, skip_abc=config.skip_abc
    )
    data_collator = DataCollatorForMultimodal(pad_token_id=0)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
        seed=102,
    )
    auto_eval_sampler = DistributedSampler(
        dataset_for_auto_eval,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=False,
        seed=102,
    )

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=data_collator,
        batch_size=config.per_device_batch_size,
        pin_memory=True,
        num_workers=4,
        sampler=train_sampler,
    )
    auto_eval_dataloader = DataLoader(
        dataset_for_auto_eval,
        batch_size=config.eval_batch_size,
        collate_fn=data_collator,
        pin_memory=True,
        sampler=auto_eval_sampler,
        worker_init_fn=utils.seed_worker,
    )
    return train_dataloader, auto_eval_dataloader


def create_overlay_images():
    questions = get_ai2d_questions(AI2D_DATA_DIR, None, False)
    for question in tqdm(questions):
        replace_text_and_save(AI2D_DATA_DIR, question)


if __name__ == "__main__":
    create_overlay_images()
