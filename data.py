import json
from collections import OrderedDict
from typing import List, Set, Optional
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
class MultipleChoiceQuestion(object):
    image_path: str
    question: str
    answers: List[str]
    correct_answer: int
    question_id: Optional[str]

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
            question_id = question_data["questionId"]
            if question_ids is not None and question_id not in question_ids:
                continue
            questions.append(
                MultipleChoiceQuestion(
                    image_path=image_path if not is_abc else image_abc_path,
                    question=question_text,
                    answers=question_data["answerTexts"],
                    correct_answer=question_data["correctAnswer"],
                    question_id=question_id
                ))
    return questions

def get_input_text(question: MultipleChoiceQuestion):
    input_text = f"Answer the following multiple choice question: {question.question}\nPossible answers are:"
    for answer in question.answers:
        input_text += f"{answer}\n"
    return input_text

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
        self.questions = get_ai2d_questions(
            root_dir, None, skip_abc=self.skip_abc
        )

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx: int):
        q = self.questions[idx]
        image = Image.open(q.image_path).convert("RGB")
        input_text = get_input_text(q)
        model_inputs = self.processor(images=image, text=input_text)
        if model_inputs is None:
            raise ValueError(f"ModelInputs is none on {idx}")
        if self.include_labels:
            input_ids = model_inputs["input_ids"].squeeze()
            target = q.answers[q.correct_answer]
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
            model_inputs["is_correct"] = True
            model_inputs["question_id"] = q.question_id
        return model_inputs

    def split(
        self, test_ids: List[str]
    ):
        image_to_question_indices: OrderedDict[str, List[int]] = OrderedDict()
        for i, question in enumerate(self.questions):
            image_id = question.image_path.split('/')[-1].split('.')[0]
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


def get_ai2d_test_ids():
    test_ids_csv = "/home/ubuntu/ai2d/ai2d_test_ids.csv"
    with open(test_ids_csv, "r") as f:
        test_ids = [line.strip() for line in f.readlines()]
    return test_ids

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
        # each question should correspond to each of the
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
            input_ids = model_inputs["input_ids"].squeeze()
            target = q.answers[answer_idx]
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
            collated["question_id"] = [
                instance["question_id"] for instance in instances
            ]
        return collated


def replace_text_and_save(base_path, question: MultipleChoiceQuestion):
    j = int(question.image_path.replace('-abc','').split(".")[0].split("/")[-1])
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
    questions = get_ai2d_questions("/home/ubuntu/ai2d", None, True)
    for question in tqdm(questions):
        replace_text_and_save("/home/ubuntu/ai2d", question)
