import argparse
import functools
import gc
import glob
import json
import os
import pprint
import time
from dataclasses import asdict
from typing import Union

import torch
import torch.distributed as dist
import transformers
from peft import PeftModel, get_peft_model
from peft.tuners.lora import LoraConfig, LoraLayer
from torch.distributed.fsdp.api import (
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.profiler import profile
from tqdm import tqdm
from transformers import FuyuForCausalLM, get_scheduler
from transformers.models.fuyu.modeling_fuyu import FuyuVisionEmbedTokens
from transformers.models.persimmon.modeling_persimmon import (
    PersimmonDecoderLayer,
    PersimmonEmbedTokens,
    PersimmonOutputEmbedding,
)

import data as data_module
import eval
import utils
import wandb
from config import Config, parse_args

OUTPUT_DIR = "/workspace/fuyu/output"


def get_lora_model(model, checkpoint_dir: str, config: Config):
    if os.path.exists(os.path.join(checkpoint_dir, "adapter_model")):
        model = PeftModel.from_pretrained(
            model, os.path.join(checkpoint_dir, "adapter_model")
        )
    else:
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                names = name.split(".")
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        lora_module_names.remove("lm_head")
        if not config.lora_vision:
            lora_module_names.remove("vision_embed_tokens")
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=list(lora_module_names),
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    for name, module in model.named_modules():
        if hasattr(module, "weight") and module.weight.dtype != torch.bfloat16:
            print(name)
            module.to(torch.bfloat16)
        if isinstance(module, LoraLayer):
            module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model


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


def save_lora_model(step, model, tokenizer):
    model_path = os.path.join(get_checkpoint_dir(step), "adapter_model")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(get_checkpoint_dir(step))


def save_fsdp_model(step, model, tokenizer):
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()
    if local_rank == 0:
        print("Saving model.")
        model.save_pretrained(get_checkpoint_dir(step), state_dict=cpu_state)
        tokenizer.save_pretrained(get_checkpoint_dir(step))


def save_model(step, model, tokenizer, is_lora):
    return (
        save_lora_model(step, model, tokenizer)
        if is_lora
        else save_fsdp_model(step, model, tokenizer)
    )


def get_latest_checkpoint_dir(run_name: str) -> str:
    run_dir = get_run_dir(run_name)
    paths = glob.glob(os.path.join(run_dir, "step-*"))
    steps = [p.split("-")[-1] for p in paths]
    if "final" in steps:
        checkpoint_dir = os.path.join(run_dir, "step-final")
    else:
        step = max([int(s) for s in steps])
        checkpoint_dir = os.path.join(run_dir, f"step-{step}")
    return checkpoint_dir


def load_model(config: Config):
    model_path = config.model_name_or_path
    if config.run_name is not None:
        model_path = get_latest_checkpoint_dir(config.run_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    fuyu_config = transformers.FuyuConfig.from_pretrained(model_path)
    fuyu_config.update(dict(use_flash_attn=config.use_flash_attn))
    model: transformers.FuyuForCausalLM = transformers.FuyuForCausalLM.from_pretrained(
        model_path,
        config=fuyu_config,
        torch_dtype=torch.bfloat16,
    )
    assert model.language_model.config.use_flash_attn == config.use_flash_attn
    # Avoid conflicts with gradient checkpointing.
    model.language_model.config.use_cache = False
    model.config.use_cache = False
    model.language_model.model.gradient_checkpointing_enable()
    model.gradient_checkpointing_enable()

    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            assert module.weight.dtype == torch.bfloat16, f"{name} is not bfloat16"

    if config.lora:
        model = get_lora_model(model, model_path, config)
    elif config.run_name is not None:
        raise NotImplementedError("Resuming non-finetune runs not yet implemented.")
    return model, tokenizer


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


def get_optimizer(model: Union[FuyuForCausalLM, FSDP, PeftModel], config: Config):
    if config.lora:
        opt_params = [p for n, p in model.named_parameters() if "lora" in n]
    else:
        opt_params = model.parameters()
    utils.print_trainable_parameters(model)
    optimizer = torch.optim.AdamW(opt_params, foreach=False, lr=config.learning_rate)
    return optimizer


def export_profile(prof, local_rank):
    print("Exporting chrome trace.")
    prof.export_chrome_trace(f"traces/trace_{local_rank}.json")
    print("Exporting memory timeline.")
    prof.export_memory_timeline(
        f"timelines/memory_timeline_{local_rank}.html", f"cuda:{local_rank}"
    )


def train(
    config: Config,
    local_rank,
    world_size,
):
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        # wrap very large embedding layers to reduce OOM.
        transformer_layer_cls={
            FuyuVisionEmbedTokens,
            PersimmonEmbedTokens,
            PersimmonDecoderLayer,
            PersimmonOutputEmbedding,
        },
    )

    model, tokenizer = load_model(config)
    model = FSDP(
        model,
        limit_all_gathers=True,
        cpu_offload=None,
        backward_prefetch=None,
        param_init_fn=None,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
    )
    optimizer = get_optimizer(model, config)

    train_dataloader, auto_eval_dataloader = data_module.get_data(
        config, world_size=world_size, local_rank=local_rank, tokenizer=tokenizer
    )
    max_train_steps = torch.tensor(len(train_dataloader)).long().to(local_rank)
    # allreduce
    print(max_train_steps)
    dist.all_reduce(max_train_steps, op=dist.ReduceOp.MIN)
    print("postreduce", max_train_steps)

    if local_rank == 0:
        print(model)

    lr_scheduler = get_scheduler(
        name=config.scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=max_train_steps.cpu().item(),
    )

    if local_rank == 0:
        wandb.init(project="fuyu", config=config.__dict__)
        if wandb.run is None:
            raise Exception
        save_config(config, wandb.run.name)

    completed_steps = 0
    model.train()
    dist.barrier()
    throughput_counter = 0

    def step_fn(batch):
        nonlocal model
        nonlocal completed_steps
        nonlocal throughput_counter
        model.train()
        batch = utils.prepare_inputs(batch, model.device, fdtype=torch.bfloat16)
        ct = torch.tensor(batch["input_ids"].shape[1]).to(local_rank)
        dist.all_reduce(ct, op=dist.ReduceOp.SUM)
        throughput_counter += ct.cpu().item()
        try:
            loss = model(**batch).loss
            loss.backward()
            model.clip_grad_norm_(1.0)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
        except torch.cuda.OutOfMemoryError as e:
            print("OOM on inputs with shape", batch["input_ids"].shape)
            raise e
        loss = loss.detach()
        loss = utils.get_all_reduce_mean(loss).item()
        completed_steps += 1
        if local_rank == 0:
            if wandb.run is not None:
                wandb.log({"step": completed_steps, "loss/train": loss})

        if completed_steps % config.save_every_steps == 0:
            save_model(completed_steps, model, tokenizer, config.lora)

        if completed_steps % config.eval_every_steps == 0:
            accuracy = eval.auto_eval_dist(
                model, auto_eval_dataloader, local_rank, world_size
            )
            if local_rank == 0 and accuracy is not None:
                wandb.log({"step": completed_steps, "accuracy/val": accuracy})

    start_time = time.time()

    def log_throughput():
        nonlocal start_time
        nonlocal completed_steps
        nonlocal throughput_counter
        if completed_steps % 10 == 0 and local_rank == 0:
            end_time = time.time()
            throughput = throughput_counter / (end_time - start_time)
            print(f"Throughput: {throughput} elts/sec")
            wandb.log({"step": completed_steps, "throughput": throughput})
            throughput_counter = 0
            start_time = time.time()

    if config.profile:
        with profile(profile_memory=True, record_shapes=True, with_stack=True) as prof:
            for batch in tqdm(train_dataloader, disable=(local_rank != 0)):
                gc.collect()
                torch.cuda.memory.empty_cache()
                step_fn(batch)
                if completed_steps >= 10:
                    log_throughput()
                    break
        export_profile(prof, local_rank)
    else:
        for batch in tqdm(train_dataloader, disable=(local_rank != 0)):
            gc.collect()
            torch.cuda.memory.empty_cache()
            step_fn(batch)
            log_throughput()
            if completed_steps >= max_train_steps:
                break
        accuracy = eval.auto_eval_dist(
            model, auto_eval_dataloader, local_rank, world_size
        )
        if local_rank == 0:
            wandb.log({"accuracy/final": accuracy})
        save_model("final", model, tokenizer, config.lora)


def main(config, local_rank, world_size):
    if config.run_name is not None:
        print(f"Loading config from {config.run_name}")
        config = load_config(config.run_name)
    if local_rank == 0:
        pprint.pprint(config)
    seed = utils.enforce_reproducibility(config.seed)
    config.seed = seed
    train(config, local_rank, world_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Configuration")
    config = parse_args(parser)
    if config.do_vocab_surgery:
        # Todo move to another file
        config.lora = False
        model, tokenizer = load_model(config)
        print("doing vocab surgery")
        model, tokenizer = utils.vocab_surgery(model, tokenizer)
        print(model)
        print("saving surgery model")
        tokenizer.save_pretrained("fuyu-8b-slim-vocab")
        model.save_pretrained("fuyu-8b-slim-vocab")
        exit(0)
    else:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
        print(torch.distributed.get_world_size())
        main(config, local_rank, world_size)
