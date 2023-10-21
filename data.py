import json
import random
from collections import OrderedDict
from typing import Dict, Tuple, List, Any
import os
import copy
from dataclasses import dataclass
from PIL import Image
import torch
from torch.utils.data import Dataset, Subset
from torch.nn.utils.rnn import pad_sequence
from transformers import FuyuProcessor, FuyuImageProcessor, AutoTokenizer

random.seed(102123)


class AI2DDataset(Dataset):
    def __init__(self, root_dir: str, processor: FuyuProcessor):
        self.questions: List[Dict[str, Any]] = []
        self.image_to_question_indices = OrderedDict()
        self.processor = processor
        self.include_labels = True
        self._init_questions(root_dir)

    def _init_questions(self, root_dir):
        questions_dir = os.path.join(root_dir, "questions")
        images_dir = os.path.join(root_dir, "images")
        for path in sorted(os.listdir(questions_dir)):
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
                if image_name in self.image_to_question_indices:
                    self.image_to_question_indices[image_name].append(
                        len(self.questions) - 1
                    )
                else:
                    self.image_to_question_indices[image_name] = [
                        len(self.questions) - 1
                    ]

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        q = self.questions[idx]
        image = Image.open(q["image_path"]).convert("RGB")
        model_inputs = self.processor(images=image, text=q["question"])
        if model_inputs is None:
            raise ValueError(f"ModelInputs is none on {idx}")
        if self.include_labels:
            input_ids = model_inputs["input_ids"].squeeze()
            target = q["answer"]
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
        return model_inputs

    def split(self, prop: float) -> Tuple["AI2DDataset", "AI2DDataset"]:
        images = list(self.image_to_question_indices.keys())
        random.shuffle(images)
        idx = int(prop * len(images))
        first_images, second_images = images[:idx], images[idx:]
        first_indices, second_indices = [], []
        for im in first_images:
            first_indices.extend(self.image_to_question_indices[im])
        for im in second_images:
            second_indices.extend(self.image_to_question_indices[im])
        return Subset(self, first_indices), Subset(self, second_indices)


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
            values = [instance[key].squeeze() for instance in instances]
            collated[key] = pad_sequence(
                values, batch_first=True, padding_value=pad_values[key]
            )
        attention_mask = collated["input_ids"].ne(pad_values["input_ids"])
        # Fuyu does not have a pad token id, so we don't want to overwrite
        # the zero token.
        collated["input_ids"][~attention_mask] = self.pad_token_id
        collated["attention_mask"] = attention_mask
        return collated


if __name__ == "__main__":
    pretrained_path = "adept/fuyu-8b"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    image_processor = FuyuImageProcessor()
    processor = FuyuProcessor(image_processor=image_processor, tokenizer=tokenizer)
    ds = AI2DDataset("/home/ubuntu/fuyu/ai2d", processor)
    model_outputs = print(ds[1])
