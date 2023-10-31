import cProfile
import gc
import json
import random

import numpy as np
import torch
from tokenizers import Tokenizer
from transformers.models.fuyu import FuyuConfig, FuyuForCausalLM

ADEPT_VOCAB_SIZE = 262144

def prepare_inputs(model_inputs, fdtype=torch.bfloat16):
    result = {}
    for k, v in model_inputs.items():
        if k in ("is_correct", "question_id"):
            continue
        tensor = v.to("cuda:0")
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


def profile_function(func):
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        profiler.dump_stats(f"{func.__name__}.prof")
        return result

    return wrapper


def enforce_reproducibility(use_seed=None):
    seed = use_seed if use_seed is not None else random.randint(1, 1000000)
    print(f"Using seed: {seed}")

    random.seed(seed)  # python RNG
    np.random.seed(seed)  # numpy RNG

    # pytorch RNGs
    torch.manual_seed(seed)  # cpu + cuda
    torch.cuda.manual_seed_all(seed)  # multi-gpu - can be called without gpus
    if use_seed:  # slower speed! https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
