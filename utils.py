import cProfile
import gc
import json
import random

import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer
from transformers.models.fuyu import FuyuConfig, FuyuForCausalLM

ADEPT_VOCAB_SIZE = 262144


def prepare_inputs(model_inputs, device, fdtype=torch.bfloat16):
    result = {}
    for k, v in model_inputs.items():
        if k in ("is_correct", "question_id"):
            continue
        tensor = v.to(device)
        if tensor.dtype in (torch.float32, torch.float16, torch.bfloat16):
            tensor = tensor.to(fdtype)
        result[k] = tensor
    return result


def clear_mem():
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()


def print_parameters(model):
    for name, _ in model.named_parameters():
        if any(str(i) in name for i in range(1, 10)):
            continue
        if "0" in name:
            print(name.replace("0", "%d"))
        else:
            print(name)


def python_profile_function(func):
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        profiler.dump_stats(f"{func.__name__}.prof")
        return result

    return wrapper


# from Stas Bekman https://github.com/stas00/ml-engineering/tree/master/reproducibility
def enforce_reproducibility(use_seed=None):
    seed = use_seed if use_seed is not None else random.randint(1, 1000000)

    random.seed(seed)  # python RNG
    np.random.seed(seed)  # numpy RNG

    # pytorch RNGs
    torch.manual_seed(seed)  # cpu + cuda
    torch.cuda.manual_seed_all(seed)  # multi-gpu - can be called without gpus
    return seed


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def vocab_surgery(fuyu_model: FuyuForCausalLM, tokenizer):
    print("Doing model surgery.")
    start, end = (3, 70_003)
    assert (
        tokenizer.vocab_size == ADEPT_VOCAB_SIZE
    ), "Not doing model surgery on a model with a different vocab size."
    tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
    vocab = tokenizer_json["model"]["vocab"]
    new_vocab = []
    for i, tok in enumerate(vocab):
        if i < start or i >= end:
            new_vocab.append(tok)
    tokenizer_json["model"]["vocab"] = new_vocab
    tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))

    embed = fuyu_model.language_model.model.embed_tokens.weight.detach()
    hidden_size = embed.shape[1]
    new_embed = torch.concat([embed[:start, :], embed[end:, :]])
    new_vocab_size = new_embed.shape[0]
    new_embed = torch.nn.Embedding(new_vocab_size, hidden_size, _weight=new_embed)
    fuyu_model.language_model.model.embed_tokens = new_embed.to(fuyu_model.device)

    head = fuyu_model.language_model.lm_head.weight.detach()
    new_linear = torch.nn.Linear(hidden_size, new_vocab_size, bias=False)
    new_linear_weight = torch.concat([head[:start, :], head[end:, :]])
    new_linear.weight.data = new_linear_weight
    fuyu_model.language_model.lm_head = new_linear.to(fuyu_model.device)

    fuyu_model.config.update(dict(vocab_size=new_vocab_size))
    fuyu_model.language_model.config.update(dict(vocab_size=new_vocab_size))
    return fuyu_model, tokenizer


def estimate_activation_memory(config: FuyuConfig, s, b):
    h = config.hidden_size
    L = config.num_hidden_layers
    a = config.num_attention_heads
    return (34 * h + 5 * a * s) * s * b * L


# thanks to artidoro/qlora
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )


# use 1011 and 1019 for slim tokenizer.
def get_image_from_inputs(
    inputs, idx=0, image_token=71011, image_nl_token=71019, patch_h=30, patch_w=30
):
    ids = inputs["input_ids"][idx].detach().cpu()
    patches = inputs["image_patches"][idx].detach().cpu()
    assert patches.shape[1] == patch_h * patch_w * 3
    w, h = 0, 0
    # assumes variable size images (so includes raster order newlines)
    for token in ids[idx]:
        if token == image_token and h == 0:
            w += 1
        if token == image_nl_token:
            h += 1
    assert w * h == patches.shape[0]
    imarr = np.zeros((patch_h * h, patch_w * w, 3))
    for i in range(h):
        for j in range(w):
            patch_idx = w * i + j
            imarr[
                i * patch_h : (i + 1) * patch_h, j * patch_w : (j + 1) * patch_w, :
            ] = patches[patch_idx].view(patch_h, patch_w, 3)
    scaled = (((imarr + 1) / 2) * 255).astype(np.uint8)
    return Image.fromarray(scaled)


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor
