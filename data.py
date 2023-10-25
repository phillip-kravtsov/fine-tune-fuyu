import json
import random
from collections import OrderedDict
from typing import Dict, Tuple, List, Set, Optional
import os
import copy
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset, Subset
from torch.nn.utils.rnn import pad_sequence
from transformers import FuyuProcessor
from tqdm import tqdm


@dataclass
class Question(object):
    image_path: str
    question: str
    answer: str
    question_id: str
    is_correct: bool


def get_questions(
    root_dir: str, question_ids: Optional[Set[str]], correct_only: bool
) -> Tuple[List[Question], Dict[str, List[int]]]:
    image_to_question_indices = {}
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
            # only care for images with questions.
            assert os.path.exists(image_path), f"Could not find {image_path}"
            assert os.path.exists(image_abc_path), f"Could not find {image_abc_path}"
            question_id = question_data["questionId"]
            is_abc = question_data["abcLabel"]
            if question_ids is not None and question_id not in question_ids:
                continue
            for i, answer_text in enumerate(question_data["answerTexts"]):
                if correct_only and question_data["correctAnswer"] != i:
                    continue
                questions.append(
                    Question(
                        image_path=image_path if not is_abc else image_abc_path,
                        question=question_text,
                        answer=answer_text,
                        question_id=question_id,
                        is_correct=question_data["correctAnswer"] == i,
                    )
                )
                if image_name in image_to_question_indices:
                    image_to_question_indices[image_name].append(len(questions) - 1)
                else:
                    image_to_question_indices[image_name] = [len(questions) - 1]
    return questions, image_to_question_indices


class AI2DDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        processor: FuyuProcessor,
        instruction: Optional[str] = None,
        include_labels: bool = True,
    ):
        self.questions: List[Question] = []
        self.image_to_question_indices = OrderedDict()
        self.processor = processor
        self.include_labels = include_labels
        self.instruction = instruction
        self._init_questions(root_dir)

    def _init_questions(self, root_dir):
        self.questions, self.image_to_question_indices = get_questions(
            root_dir, None, True
        )

    def __getitem__(self, idx):
        q = self.questions[idx]
        image = Image.open(q.image_path).convert("RGB")
        input_text = q.question + "\n"
        if self.instruction:
            input_text = self.instruction + " " + input_text
        model_inputs = self.processor(images=image, text=input_text)
        if model_inputs is None:
            raise ValueError(f"ModelInputs is none on {idx}")
        if self.include_labels:
            input_ids = model_inputs["input_ids"].squeeze()
            target = q.answer
            target_ids = self.processor.tokenizer.encode(
                target + self.processor.tokenizer.eos_token,
                add_special_tokens=False,
                return_tensors="pt",
            ).squeeze()
            # input_ids should have boa token.
            all_ids = torch.concat([input_ids, target_ids])
            labels = copy.deepcopy(all_ids)
            labels[: input_ids.shape[0]] = -100
            model_inputs["input_ids"] = all_ids
            model_inputs["labels"] = labels
            model_inputs["is_correct"] = q.is_correct
            model_inputs["question_id"] = q.question_id
        return model_inputs

    def __len__(self):
        return len(self.questions)

    def split(
        self, prop: float
    ) -> Tuple["AI2DDataset", "AI2DDataset", List[str], List[str]]:
        images = list(self.image_to_question_indices.keys())
        random.shuffle(images)
        idx = int(prop * len(images))
        first_images, second_images = images[:idx], images[idx:]
        first_indices, second_indices = [], []
        for im in first_images:
            first_indices.extend(self.image_to_question_indices[im])
        for im in second_images:
            second_indices.extend(self.image_to_question_indices[im])

        first_questions = [self.questions[i].question_id for i in first_indices]
        second_questions = [self.questions[i].question_id for i in second_indices]
        return (
            Subset(self, first_indices),
            Subset(self, second_indices),
            first_questions,
            second_questions,
        )


class AI2DDatasetForAutoEval(AI2DDataset):
    def __init__(
        self,
        root_dir: str,
        processor: FuyuProcessor,
        question_ids: List[str],
        instruction: Optional[str] = None,
        include_labels=True,
    ):
        self.question_ids = set(question_ids)
        self.include_labels = include_labels
        self.questions: List[Question] = []
        self.image_to_question_indices = OrderedDict()
        self.processor = processor
        self.instruction = instruction
        self._init_questions(root_dir)

    def _init_questions(self, root_dir):
        # each question should correspond to each of the
        self.questions, self.image_to_question_indices = get_questions(
            root_dir, self.question_ids, False
        )


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
            collated["question_id"] = [
                instance["question_id"] for instance in instances
            ]
        return collated


def replace_text_and_save(question: Question):
    base_path = "/home/ubuntu/ai2d"
    j = int(question.image_path.split(".")[0].split("/")[-1])
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
        font = ImageFont.truetype("/home/ubuntu/fuyu/Arial.ttf", rectangle_height)
        draw.rectangle([top_left, bottom_right], fill=(255, 255, 255))
        replacement_text = value["replacementText"]
        draw.text(top_left, replacement_text, font=font, fill=(0, 0, 0))
    img.save(impath_abc)


if __name__ == "__main__":
    questions, _ = get_questions("/home/ubuntu/ai2d", None, True)
    for question in tqdm(questions):
        replace_text_and_save(question)
