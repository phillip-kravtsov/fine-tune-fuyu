import copy

import numpy as np
import torch
import transformers as tr
from PIL import Image
from tqdm import tqdm

import utils
from config import Config
from train import load_model


def main():
    input_image_path = "download.jpeg"
    output_image_path = "adv.png"
    config = Config(use_flash_attn=True)
    model, tokenizer = load_model(config, device_map="cuda:0")

    image = Image.open(input_image_path)
    processor = tr.FuyuProcessor(
        tokenizer=tokenizer,
        image_processor=tr.FuyuImageProcessor(),
        add_beginning_of_answer_token=False,
    )
    target_text = "AdversarialOutputException()"
    input_text = "what is in the main image at the center of the screen, under 'this is the title'?\n"
    text = input_text + target_text

    inputs = processor(images=[image], text=text).to(0)

    target_length = len(tokenizer.encode(target_text, add_special_tokens=False))
    labels = copy.deepcopy(inputs["input_ids"])
    labels[:, :-target_length] = -100
    for parameter in model.parameters():
        parameter.requires_grad = False
    criterion = torch.nn.CrossEntropyLoss()

    def fgsm_image_patches(param, eps=0.04):
        to_optimize = torch.clamp(param - (param.grad.data.sign() * eps), -1, 1)
        return to_optimize

    _, dims = utils.get_image_from_inputs(inputs)

    h0, h1 = 9, 19
    w0, w1 = 8, 24

    patches = inputs["image_patches"][0].clone().detach().view(dims[1], dims[0], -1)
    patches[h0:h1, w0:w1, :] = 0
    inputs["image_patches"] = [patches.view(1, -1, 2700)]
    inputs_to_optimize = torch.zeros(
        [h1 - h0, w1 - w0, 2700], requires_grad=True
    ).cuda()
    inputs_to_optimize.retain_grad()

    for _ in tqdm(range(10)):
        model.zero_grad()

        patches = inputs["image_patches"][0].clone().detach()
        viewed = patches.reshape(dims[1], dims[0], -1)
        viewed[h0:h1, w0:w1, :] = inputs_to_optimize
        inputs["image_patches"] = [viewed.reshape(1, -1, 2700)]

        with torch.autocast("cuda"):
            logits = model(**inputs).logits
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[..., 1:].contiguous().cuda()
        loss = criterion(
            shifted_logits.view(-1, tokenizer.vocab_size), shifted_labels.view(-1)
        )
        loss.backward()
        inputs_to_optimize = fgsm_image_patches(inputs_to_optimize)

        # Zero out the gradients for inputs_to_optimize before detaching
        if inputs_to_optimize.grad is not None:
            inputs_to_optimize.grad = None  # Clear the gradients

        inputs_to_optimize = inputs_to_optimize.detach()
        inputs_to_optimize.requires_grad_()

    patch = inputs_to_optimize.clone().detach().view(h1 - h0, w1 - w0, 30, 30, 3)
    outarr = np.zeros(((h1 - h0) * 30, (w1 - w0) * 30, 3))
    for h in range(patch.shape[0]):
        for w in range(patch.shape[1]):
            outarr[h * 30 : (h + 1) * 30, w * 30 : (w + 1) * 30, :] = (
                patch[h, w, :].cpu().numpy()
            )
    outim = Image.fromarray(((outarr + 1) * 255 / 2).astype(np.uint8))
    outim.save(output_image_path)
