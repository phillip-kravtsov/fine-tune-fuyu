import argparse
import functools
import glob
import json
import os
import pprint
import shutil
import time
from contextlib import nullcontext
from dataclasses import asdict
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import transformers
from torch.distributed.fsdp.api import (
    CPUOffload,
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
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler
from transformers.models.fuyu.modeling_fuyu import FuyuVisionEmbedTokens
from transformers.models.persimmon.modeling_persimmon import (
    PersimmonDecoderLayer,
    PersimmonEmbedTokens,
    PersimmonLMHead,
)
import fuyu

import ai2d
import eval
import lora
import scienceqa
import utils
import wandb
from config import TrainingConfig, parse_training_args
from utils import get_checkpoint_dir, get_run_dir

OUTPUT_DIR = "/workspace/fuyu/output"
SAVE_SIGNAL_FILE = "/root/should_save"


def save_fsdp_model(step, model, tokenizer, local_rank):
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()
    if local_rank == 0:
        print("Saving model.")
        run_dir = get_run_dir()
        checkpoints = glob.glob(os.path.join(run_dir, "step-*"))
        if len(checkpoints) >= 2:
            steps = sorted([int(c.split("-")[-1]) for c in checkpoints])
            for step_to_delete in steps[:-1]:
                print(f"Deleting checkpoint {step_to_delete}")
                shutil.rmtree(get_checkpoint_dir(step_to_delete))

        model.save_pretrained(get_checkpoint_dir(step), state_dict=cpu_state)
        tokenizer.save_pretrained(get_checkpoint_dir(step))
        if os.path.exists(SAVE_SIGNAL_FILE):
            os.remove(SAVE_SIGNAL_FILE)


def save_model(step, model, tokenizer, is_lora, local_rank):
    return (
        lora.save_lora_model(step, model, tokenizer)
        if is_lora
        else save_fsdp_model(step, model, tokenizer, local_rank)
    )


def load_model(config: TrainingConfig, device_map=None):
    model_path = config.model_name_or_path
    if config.run_name is not None:
        model_path = utils.get_latest_checkpoint_dir(config.run_name)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    vocab = tokenizer.get_vocab()
    tokenizer.get_vocab = lambda: vocab

    if device_map is None:
        device_map = f"cuda:{0}" if config.lora else None
    if config.patch_prediction:
        print("Loading patch prediction model")
        model = fuyu.FuyuWithPatchPrediction.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            use_flash_attention_2=config.use_flash_attn,
            device_map=device_map,
        )
    else:
        model = transformers.FuyuForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            use_flash_attention_2=config.use_flash_attn,
            device_map=device_map,
        )

    # Avoid conflicts with gradient checkpointing.
    if config.gradient_checkpointing:
        model.language_model.config.use_cache = False
        model.config.use_cache = False

        model.language_model.model.gradient_checkpointing_enable()
        model.gradient_checkpointing_enable()

    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            assert module.weight.dtype == torch.bfloat16, f"{name} is not bfloat16"
    if config.lora:
        model = lora.get_lora_model(model, model_path, config)
    elif config.run_name is not None:
        raise NotImplementedError("Resuming full runs not yet implemented.")

    if config.fsdp:
        print("Wrapping model in FSDP.")
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                FuyuVisionEmbedTokens,
                PersimmonLMHead,
                PersimmonEmbedTokens,
                PersimmonDecoderLayer,
            },
        )
        model = FSDP(
            model,
            limit_all_gathers=True,
            cpu_offload=CPUOffload(False),
            backward_prefetch=None,
            param_init_fn=None,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=torch.cuda.current_device(),
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
        )
    return model, tokenizer


def load_config(run_name: str) -> TrainingConfig:
    with open(f"{OUTPUT_DIR}/{run_name}/config.json", "r") as f:
        config = json.loads(f.read())
    return TrainingConfig(**config)


def save_config(config: TrainingConfig, run_name: str):
    run_dir = get_run_dir(run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            f.write(json.dumps(asdict(config)))


def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]

    result += list(model._parameters.keys())
    return result


def get_optimizer(model, max_train_steps: int, config: TrainingConfig):
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    if config.lora:
        opt_params = {
            n: p
            for n, p in model.named_parameters()
            if "lora" in n or "next_patch_predictor" in n
        }
        print(opt_params.keys())
        opt_params = opt_params.values()
    else:
        opt_params = optimizer_grouped_parameters
    utils.print_trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        opt_params,
        foreach=False,
        weight_decay=config.weight_decay,
        lr=config.learning_rate,
    )
    lr_scheduler = get_scheduler(
        name=config.scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=max_train_steps,
    )
    return optimizer, lr_scheduler


class Trainer:
    def __init__(
        self,
        model,
        tokenizer,
        optimizer,
        lr_scheduler,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        multiple_choice_dataloader: DataLoader,
        config: TrainingConfig,
        local_rank: int,
        world_size: int,
        max_train_steps: Optional[int] = None,
    ):
        print("Initializing trainer.")
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.multiple_choice_dataloader = multiple_choice_dataloader
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        if max_train_steps is None:
            max_train_steps = len(train_dataloader)
        self.max_train_steps = max_train_steps

    def _init_tracking(self):
        if self.local_rank == 0:
            wandb.init(
                project=f"fuyu-{self.config.dataset}", config=self.config.__dict__
            )
            if wandb.run is None:
                raise Exception
            save_config(self.config, wandb.run.name)

    def throughput(self, batch):
        end_time = time.time()
        tokens = torch.tensor(
            batch["input_ids"].shape[0] * batch["input_ids"].shape[1]
        ).to(self.local_rank)
        dist.all_reduce(tokens, op=dist.ReduceOp.SUM)
        self.throughput_counter += tokens.cpu().item()

        if self.completed_steps % 10 == 0 and self.local_rank == 0:
            throughput = self.throughput_counter / (
                end_time - self.throughput_start_time
            )
            # doesn't really apply to LoRA.
            achieved_teraflops = (throughput * 6 * 8e9) / 1e12
            if not self.config.lora:
                print(
                    f"Throughput: {round(throughput, 4)} tok/sec -> {round(achieved_teraflops, 3)} tflops"
                )
            wandb.log({"throughput": throughput}, step=self.completed_steps)
            self.throughput_counter = 0
            self.throughput_start_time = time.time()

    def _profile_train_loop(self):
        with profile(profile_memory=True, record_shapes=True, with_stack=True) as prof:
            for batch in tqdm(self.train_dataloader, disable=(self.local_rank != 0)):
                self.step(batch)
                if self.completed_steps >= 2:
                    self.throughput(batch)
                    break
        utils.export_profile(prof, self.local_rank)

    def step(self, batch):
        model, optimizer, lr_scheduler = self.model, self.optimizer, self.lr_scheduler
        model.train()
        forward_context = (
            torch.autocast("cuda") if not isinstance(model, FSDP) else nullcontext()
        )
        batch = batch.to(self.local_rank, torch.bfloat16)
        try:
            with forward_context:
                outputs = model(**batch)
            nll_loss = outputs.loss
            if self.config.patch_prediction:
                patch_loss = model.get_patch_prediction_loss(batch, outputs)
                loss = nll_loss + self.config.alpha * patch_loss
            else:
                loss = nll_loss
            loss.backward()

            if hasattr(model, "clip_grad_norm_"):
                model.clip_grad_norm_(1.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
        except torch.cuda.OutOfMemoryError as e:
            print("OOM on inputs with shape", batch["input_ids"].shape)
            raise e
        loss = loss.detach()
        loss = utils.get_all_reduce_mean(loss).item()
        to_log = {"loss/train": nll_loss}
        if self.config.patch_prediction:
            to_log["patch_loss/train"] = utils.get_all_reduce_mean(
                patch_loss.detach()
            ).item()
            to_log["nll_loss/train"] = utils.get_all_reduce_mean(
                nll_loss.detach()
            ).item()
        self.completed_steps += 1
        if self.local_rank == 0:
            if wandb.run is not None:
                wandb.log(to_log, step=self.completed_steps)

    def save_model(self, step=None):
        if step is None:
            step = self.completed_steps
        save_model(step, self.model, self.tokenizer, self.config.lora, self.local_rank)

    def train(self):
        config = self.config
        self._init_tracking()
        self.completed_steps: int = 0
        self.throughput_counter = 0
        self.throughput_start_time = time.time()
        dist.barrier()

        if config.profile:
            self._profile_train_loop()
            return

        if self.local_rank == 0:
            print("Beginning train loop.")
        for batch in tqdm(self.train_dataloader, disable=(self.local_rank != 0)):
            self.step(batch)
            self.throughput(batch)
            if self.completed_steps % config.save_every_steps == 0 or os.path.exists(
                SAVE_SIGNAL_FILE
            ):
                self.save_model()

            if self.completed_steps % config.eval_every_steps == 0:
                eval_results = self.eval("val")
                if self.local_rank == 0:
                    wandb.log(
                        eval_results,
                        step=self.completed_steps,
                    )
            if self.completed_steps >= self.max_train_steps:
                break

        eval_log = self.eval("final")

        if self.local_rank == 0:
            wandb.log(eval_log)
        self.save_model("final")

    def eval(self, suffix: str) -> Dict[str, Any]:
        results = {}
        if self.val_dataloader is not None:
            eval_results = eval.eval(
                self.model,
                self.val_dataloader,
                self.local_rank,
                self.world_size,
            )
            for k, v in eval_results.items():
                results[f"{k}/{suffix}"] = v
        if self.multiple_choice_dataloader is not None:
            strict_accuracy = eval.likelihood_eval_v2(
                self.model,
                self.multiple_choice_dataloader,
                self.local_rank,
                self.world_size,
            )
            results[f"likelihood_accuracy/{suffix}"] = strict_accuracy
        return results


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    parser = argparse.ArgumentParser(description="Fine-tune Fuyu models.")
    config = parse_training_args(parser)
    if config.run_name is not None:
        print(f"Loading config from {config.run_name}")
        config = load_config(config.run_name)
    utils.enforce_reproducibility(config.seed)
    if local_rank == 0:
        pprint.pprint(config)
        print(f"Using seed {config.seed}")

    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        local_rank = 0
        world_size = 1
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(local_rank)
    print("Initializing process group")
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    model, tokenizer = load_model(config)
    if config.dataset == "ai2d":
        (
            train_dataloader,
            max_train_steps,
            val_dataloader,
            multiple_choice_dataloader,
        ) = ai2d.get_data(
            config,
            tokenizer=tokenizer,
            local_rank=local_rank,
            world_size=world_size,
        )
    elif config.dataset == "scienceqa":
        (
            train_dataloader,
            max_train_steps,
            val_dataloader,
            multiple_choice_dataloader,
        ) = scienceqa.get_data(
            config,
            tokenizer=tokenizer,
            local_rank=local_rank,
            world_size=world_size,
        )
    else:
        raise ValueError(f"Unknown dataset {config.dataset}")

    optimizer, lr_scheduler = get_optimizer(model, max_train_steps, config)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        local_rank=local_rank,
        world_size=world_size,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        multiple_choice_dataloader=multiple_choice_dataloader,
        config=config,
    )
    trainer.train()


if __name__ == "__main__":
    main()
