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
from typing import Optional, Union, Dict, Any

import torch
import torch.distributed as dist
import transformers
import wandb
from peft import PeftModel, get_peft_model
from peft.tuners.lora import LoraConfig, LoraLayer
from torch.distributed.fsdp.api import (CPUOffload, FullStateDictConfig,
                                        MixedPrecision, ShardingStrategy,
                                        StateDictType)
from torch.distributed.fsdp.fully_sharded_data_parallel import \
    FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.profiler import profile
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import FuyuForCausalLM, get_scheduler
from transformers.models.persimmon.modeling_persimmon import (
    PersimmonDecoderLayer, PersimmonEmbedTokens)

import ai2d
import eval
import scienceqa
import utils
from config import Config, parse_args

OUTPUT_DIR = "/workspace/fuyu/output"
SAVE_SIGNAL_FILE = "/root/should_save"


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
        save_lora_model(step, model, tokenizer)
        if is_lora
        else save_fsdp_model(step, model, tokenizer, local_rank)
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


def load_model(config: Config, device_map=None):
    model_path = config.model_name_or_path
    if config.run_name is not None:
        model_path = get_latest_checkpoint_dir(config.run_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    if device_map is None:
        device_map = f"cuda:{0}" if config.lora else None
    model = transformers.FuyuForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=config.use_flash_attn,
        device_map=device_map,
    )

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
        raise NotImplementedError("Resuming full runs not yet implemented.")
    return model, tokenizer


def init_model(config):
    model, tokenizer = load_model(config)
    if not config.lora:
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                # FuyuVisionEmbedTokens,
                PersimmonEmbedTokens,
                PersimmonDecoderLayer,
                # PersimmonLMHead,
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
            # use_orig_params=True,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
        )
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


def get_optimizer(
    model: Union[FuyuForCausalLM, FSDP, PeftModel], max_train_steps: int, config: Config
):
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
        opt_params = [p for n, p in model.named_parameters() if "lora" in n]
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


def export_profile(prof, local_rank):
    print("Exporting chrome trace.")
    prof.export_chrome_trace(f"traces/trace_{local_rank}.json")
    print("Exporting memory timeline.")
    prof.export_memory_timeline(
        f"timelines/memory_timeline_{local_rank}.html", f"cuda:{local_rank}"
    )


class Trainer:
    def __init__(
        self,
        model,
        tokenizer,
        optimizer,
        lr_scheduler,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader],
        greedy_eval_dataloader: DataLoader,
        config: Config,
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
        self.eval_dataloader = eval_dataloader
        self.greedy_eval_dataloader = greedy_eval_dataloader
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        if max_train_steps is None:
            max_train_steps = len(train_dataloader)
        self.max_train_steps = max_train_steps

    def _init_tracking(self):
        if self.local_rank == 0:
            wandb.init(project="fuyu", config=self.config.__dict__)
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
        export_profile(prof, self.local_rank)

    def step(self, batch):
        model, optimizer, lr_scheduler = self.model, self.optimizer, self.lr_scheduler
        model.train()
        batch = utils.prepare_inputs(
            batch, f"cuda:{torch.cuda.current_device()}", fdtype=torch.bfloat16
        )
        forward_context = (
            torch.autocast("cuda") if not isinstance(model, FSDP) else nullcontext()
        )
        assert "image_patches" in batch and len(batch["image_patches"]) > 0, batch
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(k, v.shape)
            if isinstance(v, list):
                for i, tensor in enumerate(v):
                    print(f"{k}.{i}", tensor.shape)
        """
        try:
            with forward_context:
                loss = model(**batch).loss
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
        self.completed_steps += 1
        if self.local_rank == 0:
            if wandb.run is not None:
                wandb.log({"loss/train": loss}, step=self.completed_steps)

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
            print("Beginning train loop proper.")
        for batch in tqdm(self.train_dataloader, disable=(self.local_rank != 0)):
            self.step(batch)

            self.throughput(batch)

            if self.completed_steps % config.save_every_steps == 0 or os.path.exists(
                SAVE_SIGNAL_FILE
            ):
                self.save_model()

            if self.completed_steps % config.eval_every_steps == 0:
                to_log = self.eval("val")
                if self.local_rank == 0:
                    wandb.log(
                        to_log,
                        step=self.completed_steps,
                    )
            if self.completed_steps >= self.max_train_steps:
                break

        eval_log = self.eval("final")

        if self.local_rank == 0:
            wandb.log(eval_log)
        self.save_model("final")

    def eval(self, suffix: str) -> Dict[str, Any]:
        to_log = {}
        if self.greedy_eval_dataloader is not None:
            strict_accuracy, loss = eval.greedy_eval(
                self.model,
                self.greedy_eval_dataloader,
                self.local_rank,
                self.world_size,
            )
            to_log[f"strict_accuracy/{suffix}"] = strict_accuracy
            to_log[f"loss/{suffix}"] = loss

        if self.eval_dataloader is not None:
            accuracy, loss = eval.auto_eval_dist(
                self.model,
                self.eval_dataloader,
                self.local_rank,
                self.world_size,
            )
            to_log[f"accuracy/{suffix}"] = accuracy
            to_log[f"loss-auto/{suffix}"] = loss
        return to_log

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    parser = argparse.ArgumentParser(description="Fine-tune Fuyu models.")
    config = parse_args(parser)
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
    model, tokenizer = init_model(config)
    if config.dataset == 'ai2d':
        (
            train_dataloader,
            eval_dataloader,
            max_train_steps,
            greedy_eval_dataloader,
        ) = ai2d.get_data(
            config,
            world_size=world_size,
            local_rank=local_rank,
            tokenizer=tokenizer,
        )
    elif config.dataset == 'scienceqa':
        (
            train_dataloader,
            eval_dataloader,
            max_train_steps,
            greedy_eval_dataloader,
        ) = scienceqa.get_data(
            config,
            world_size=world_size,
            local_rank=local_rank,
            tokenizer=tokenizer,
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
        eval_dataloader=eval_dataloader,
        greedy_eval_dataloader=greedy_eval_dataloader,
        config=config,
    )
    trainer.train()


if __name__ == "__main__":
    main()
