import copy
from dataclasses import dataclass

import torch
from transformers import FuyuProcessor
from transformers.models.fuyu.processing_fuyu import BEGINNING_OF_ANSWER_STRING


@dataclass
class FuyuMultiAnswerCollator(object):
    processor: FuyuProcessor

    def __call__(self, instances):
        tokenizer = self.processor.tokenizer
        images = [instance["image"] for instance in instances]
        texts = [
            instance["text"] + BEGINNING_OF_ANSWER_STRING for instance in instances
        ]
        collated = self.processor(images=images, text=texts)
        batch_answers = []
        for instance in instances:
            answers = []
            for answer in instance["answers"]:
                answers.append(
                    tokenizer.encode(
                        answer, add_special_tokens=False, return_tensors="pt"
                    )
                )
            batch_answers.append(answers)
        collated["batch_answers"] = batch_answers
        collated["correct_answers"] = torch.tensor(
            [instance["correctAnswer"] for instance in instances]
        )
        return collated


@dataclass
class FuyuCollator(object):
    processor: FuyuProcessor
    train_on_inputs: bool

    def __call__(self, instances):
        tokenizer = self.processor.tokenizer
        images = [instance["image"] for instance in instances]
        texts = [
            instance["text"]
            + BEGINNING_OF_ANSWER_STRING
            + instance["target"]
            + tokenizer.eos_token
            for instance in instances
        ]
        collated = self.processor(
            images=images, text=texts
        )  # .to(self.device, torch.bfloat16)
        batch_size = collated["input_ids"].shape[0]

        labels = copy.deepcopy(collated["input_ids"])
        labels = torch.where(collated["attention_mask"].bool(), labels, -100)
        for i in range(batch_size):
            text, target = instances[i]["text"], instances[i]["target"]
            tokenized_target = tokenizer(target, add_special_tokens=False)
            target_size = len(tokenized_target) + 2  # EOS + boa token
            if self.train_on_inputs:
                tokenized_text = tokenizer(text, add_special_tokens=False)
                target_size += len(tokenized_text) + 1
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
    target_width, target_height = (
        image_processor.size["width"],
        image_processor.size["height"],
    )
    if image_width <= target_width and image_height <= target_height:
        return image_height, image_width

    height_scale_factor = target_height / image_height
    width_scale_factor = target_height / image_width
    optimal_scale_factor = min(height_scale_factor, width_scale_factor)

    new_height = int(image_height * optimal_scale_factor)
    new_width = int(image_width * optimal_scale_factor)
    return new_height, new_width
