import cProfile
import gc
import json
import os
import random
from dataclasses import asdict

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

import wandb
from config import Config

OUTPUT_DIR = "/workspace/fuyu/output"


def prepare_inputs(model_inputs, device, fdtype=torch.bfloat16):
    result = {}
    for k, v in model_inputs.items():
        if k in ("is_correct", "question_id"):
            continue
        if isinstance(v, torch.Tensor):
            tensor = v.to(device)
            if tensor.dtype in (torch.float32, torch.float16, torch.bfloat16):
                tensor = tensor.to(fdtype)
            result[k] = tensor
        elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
            result[k] = [v_.to(device) for v_ in v]
        else:
            result[k] = v
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


def get_all_reduce_mean(tensor):
    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    # tensor = tensor / dist.get_world_size()
    return tensor


# copied from artidoro/qlora
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
    inputs,
    idx=0,
    image_token=71011,
    image_nl_token=71019,
    patch_h=30,
    patch_w=30,
    return_dims=False,
):
    if image_token not in inputs["input_ids"][idx]:
        raise ValueError(
            f"Image token {image_token} not found in input. Try f{1011 if image_token == 71011 else 71011}"
        )
    ids = inputs["input_ids"][idx].detach().cpu()
    patches = inputs["image_patches"][idx].detach().cpu()[0]
    assert patches.shape[1] == patch_h * patch_w * 3, patches.shape
    w, h = 0, 0
    # assumes variable_size images (so includes raster order newlines)
    for token in ids:
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
    if return_dims:
        return Image.fromarray(scaled), (w, h)
    else:
        return Image.fromarray(scaled)


def get_checkpoint_dir(step, run_name=None):
    if run_name is None:
        assert wandb.run is not None
        run_name = wandb.run.name
    return os.path.join(OUTPUT_DIR, f"{run_name}/step-{step}")


def get_run_dir(run_name=None):
    if run_name is None:
        assert wandb.run is not None
        run_name = wandb.run.name
    return os.path.join(OUTPUT_DIR, run_name)


def load_config(run_name: str) -> Config:
    with open(f"{OUTPUT_DIR}/{run_name}/config.json", "r") as f:
        config = json.loads(f.read())
    return Config(**config)


def save_config(config: Config, run_name: str):
    run_dir = get_run_dir(run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            f.write(json.dumps(asdict(config)))


def export_profile(prof, local_rank):
    trace_path = f"traces/trace_{local_rank}.json"
    timeline_path = f"timelines/memory_timeline_{local_rank}.html"
    print(f"Exporting chrome trace to {trace_path}")
    if not os.path.exists("traces"):
        os.makedirs("traces")
    prof.export_chrome_trace(f"traces/trace_{local_rank}.json")

    print(f"Exporting memory timeline to {timeline_path}")
    if not os.path.exists("timelines"):
        os.makedirs("timelines")
    prof.export_memory_timeline(
        f"timelines/memory_timeline_{local_rank}.html", f"cuda:{local_rank}"
    )
