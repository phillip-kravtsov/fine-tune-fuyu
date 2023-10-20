import json
import random
from collections import OrderedDict
from typing import Dict, Optional, Tuple
import os
import copy
from dataclasses import dataclass
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import FuyuProcessor, FuyuImageProcessor, AutoTokenizer

""" 
images
  0.png
  1.png ...
questions
  1.png.json...
"""

IMG_PLACEHOLDER_TOKEN = 71011
IMG_PLACEHOLDER_TOKEN = 71019


class AI2DDataset(Dataset):
    def __init__(self, root_dir: Optional[str]):
        self.questions = []
        self.image_to_question = OrderedDict()
        if root_dir:
            self._init_questions(root_dir)

    def _init_questions(self, root_dir):
        questions_dir = os.path.join(root_dir, "questions")
        images_dir = os.path.join(root_dir, "images")
        for path in os.listdir(questions_dir):
            with open(os.path.join(questions_dir, path), "r") as f:
                data = json.load(f)
            image_name = data["imageName"]
            image_path = os.path.join(images_dir, image_name)
            for question, question_data in data["questions"].items():
                answer = question_data["answerTexts"][question_data["correctAnswer"]]
                question = {
                    "image_path": image_path,
                    "question": question,
                    "answer": answer,
                }
                self.questions.append(question)
                if image_name in self.image_to_question:
                    self.image_to_question[image_name].append(question)
                else:
                    self.image_to_question[image_name] = [question]

    # Splits on `proportion` images.
    def split(self, proportion) -> Tuple["AI2DDataset", "AI2DDataset"]:
        images = list(self.image_to_question.keys())
        idx = int(len(images) * proportion)
        assert idx > 0
        first, second = AI2DDataset(None), AI2DDataset(None)
        for image_name in images[:idx]:
            first.questions.extend(self.image_to_question[image_name])
            first.image_to_question[image_name] = self.image_to_question[image_name]
        for image_name in images[idx:]:
            second.questions.extend(self.image_to_question[image_name])
            second.image_to_question[image_name] = self.image_to_question[image_name]
        return first, second

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        qdict = self.questions[idx]
        image_pil = Image.open(qdict["image_path"])
        return {**qdict, "image": image_pil}


@dataclass
class DataCollatorForMultiModal(object):
    processor: FuyuProcessor
    include_labels: bool

    def __call__(self, instances):
        images = [instance["image"] for instance in instances]

        def text(q):
            if self.include_labels:
                return "\n".join([q["question"], q["answer"]])
            else:
                return q["question"] + "\n"

        texts = [text(instance) for instance in instances]
        try:
            result: Optional[Dict] = self.processor(text=texts, images=images)
        except Exception as e:
            for image in images:
                print(image.size)
            return None
        if result is None:
            raise ValueError
        out = {
            **result,
        }
        if self.include_labels:
            attention_mask = result["attention_mask"]
            labels = copy.deepcopy(result["input_ids"])
            # set ignore indices for labels, considering padding
            for i, instance in enumerate(instances):
                aenc = self.processor.tokenizer.encode(
                    instance["answer"], add_special_tokens=False
                )
                # get the amount of padding received in this batch.
                offset = torch.sum(attention_mask[i] == 0).item()
                until = len(aenc) + 1 + offset
                labels[i, :-until] = -100  # ignore index for loss
                labels[i][attention_mask[i] == 0] = -100
            out["labels"] = labels
        return out


if __name__ == "__main__":
    pretrained_path = "adept/fuyu-8b"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    image_processor = FuyuImageProcessor()
    processor = FuyuProcessor(image_processor=image_processor, tokenizer=tokenizer)
    ds = AI2DDataset("/home/ubuntu/fuyu/ai2d")
    model_outputs = print(ds[1])
