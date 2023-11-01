import argparse
import functools
import glob
import json
import os
import pprint
from dataclasses import asdict, dataclass, field, fields
from typing import Optional, Union, get_origin, get_args
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)

import bitsandbytes as bnb
import torch
import transformers
from peft import PeftModel, get_peft_model
from peft.tuners.lora import LoraConfig, LoraLayer
from tqdm import tqdm
from transformers import FuyuForCausalLM, get_scheduler

import data as data_module
import eval
import utils
import wandb

OUTPUT_DIR = "/root/fuyu/output"


@dataclass
class Config:
    model_name_or_path: str = field(default="adept/fuyu-8b-slim-vocab")
    max_eval_ids: Optional[int] = field(default=500)
    train_on_questions: bool = field(default=False)
    eval_batch_size: int = field(default=4)
    save_every_steps: int = field(default=1000)
    eval_every_steps: int = field(default=1000)
    per_device_batch_size: int = field(default=2)
    learning_rate: float = field(default=3e-4)
    scheduler_type: str = field(default="constant")
    warmup_steps: int = field(default=200)
    lora: bool = field(default=False)
    lora_r: int = field(default=32)
    lora_alpha: int = field(default=32)
    lora_vision: bool = field(default=False)
    use_8bit_optimizer: bool = field(default=False)
    gradient_accumulation_steps: int = field(default=1)
    run_name: Optional[str] = field(default=None)
    weight_decay: float = field(default=0.01)
    do_vocab_surgery: bool = field(default=False)
    seed: Optional[int] = field(default=None)
    skip_abc: bool = field(default=False)


def get_lora_model(model, checkpoint_dir: Optional[str], config: Config):
    assert config.lora
    if checkpoint_dir is not None:
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
        if hasattr(module, 'weight') and module.weight.dtype != torch.bfloat16:
            print(name)
            module.to(torch.bfloat16)
        """
        if isinstance(module, LoraLayer):
            module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
        """
    return model


def save_model(step, model, is_lora):
    assert wandb.run is not None
    checkpoint_folder = os.path.join(OUTPUT_DIR, f"{wandb.run.name}/step-{step}")
    if is_lora:
        model_path = os.path.join(checkpoint_folder, "adapter_model")
    else:
        model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
    model.save_pretrained(model_path)


def get_latest_checkpoint_dir(run_name: str) -> str:
    run_dir = f"{OUTPUT_DIR}/{run_name}/"
    paths = glob.glob(os.path.join(run_dir, "step-*"))
    steps = [p.split("-")[-1] for p in paths]
    if "final" in steps:
        checkpoint_dir = os.path.join(run_dir, "step-final")
    else:
        step = max([int(s) for s in steps])
        checkpoint_dir = os.path.join(run_dir, f"step-{step}")
    return checkpoint_dir


def load_model(config: Config):
    checkpoint_dir = None
    if config.run_name is not None:
        checkpoint_dir = get_latest_checkpoint_dir(config.run_name)
    model = transformers.FuyuForCausalLM.from_pretrained(
        config.model_name_or_path, torch_dtype=torch.bfloat16
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name_or_path)

    model.gradient_checkpointing_enable()
    model.language_model.model.gradient_checkpointing_enable()
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight.dtype != torch.bfloat16:
            print(name)
            module.to(torch.bfloat16)

    if config.lora:
        model = get_lora_model(model, checkpoint_dir, config)
    elif config.run_name is not None:
        raise NotImplementedError("Resuming non-finetune runs not yet implemented.")
    return model, tokenizer


def load_config(run_name: str) -> Config:
    with open(f"/{OUTPUT_DIR}/{run_name}/config.json", "r") as f:
        config = json.loads(f.read())
    return Config(**config)


def save_config(config: Config, run_name: str):
    run_dir = f"/{OUTPUT_DIR}/{run_name}"
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            f.write(json.dumps(asdict(config)))


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


def get_optimizer(model: FuyuForCausalLM, config: Config):
    if config.lora:
        opt_params = [p for n, p in model.named_parameters() if "lora" in n]
        opt_group_params = [
            {
                "params": opt_params,
                "weight_decay": 0.0,
            },
        ]
    else:
        # todo consider (variable) weight decay
        opt_group_params = [
            {
                "params": model.parameters(),
                "weight_decay": 0.0,
            }
        ]
    print_trainable_parameters(model)
    if config.use_8bit_optimizer:
        optimizer = bnb.optim.PagedAdamW8bit(opt_group_params, lr=config.learning_rate)
    else:
        optimizer = torch.optim.SGD(
            opt_group_params, momentum=0.95, lr=config.learning_rate
        )
    return optimizer

def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor

def train(
    model: FuyuForCausalLM,
    tokenizer,
    config: Config,
    local_rank,
    world_size,
):
    train_dataloader, auto_eval_dataloader = data_module.get_data(config, world_size=world_size, local_rank=local_rank, tokenizer=tokenizer)
    max_train_steps = len(train_dataloader)
    optimizer = get_optimizer(model, config)
    from transformers.models.persimmon.modeling_persimmon import PersimmonDecoderLayer
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            PersimmonDecoderLayer,
        },
    )

    fsdp_config = dict(
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        backward_prefetch=None,
        param_init_fn=None,
        cpu_offload=None,
    )
    model = FSDP(model, **fsdp_config)

    lr_scheduler = get_scheduler(
        name=config.scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps * config.gradient_accumulation_steps,
        num_training_steps=max_train_steps * config.gradient_accumulation_steps,
    )
    """
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision="bf16",
    )
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    """
    if local_rank == 0:
        wandb.init(project="fuyu", config=config.__dict__)
        if wandb.run is None:
            raise Exception
        save_config(config, wandb.run.name)

    completed_steps = 0
    losses = []
    model.train()
    dist.barrier()
    for step, batch in enumerate(tqdm(train_dataloader, disable=(local_rank != 0))):
        model = model.train()
        print(local_rank, torch.cuda.memory_allocated() / 1e9, 'GB')
        batch = utils.prepare_inputs(batch, model.device, fdtype=torch.bfloat16)
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        loss = loss.detach()
        loss  = get_all_reduce_mean(loss).item()
        if local_rank == 0:
            losses.append(loss)

        completed_steps += 1
        if local_rank == 0:
            loss = sum(losses) / len(losses)
            wandb.log(
                {"step": completed_steps, "loss/train": loss }
            )
            losses = []
        if completed_steps % config.save_every_steps == 0:
            save_model(completed_steps, model, config.lora)
        if completed_steps % config.eval_every_steps == 0:
            model.eval()
            accuracy, eval_loss = eval.do_auto_eval(
                model, auto_eval_dataloader
            )
            wandb.log({
                    "step": completed_steps,
                    "accuracy/val": accuracy,
                    "loss/val": eval_loss,
                })
    accuracy, eval_loss = eval.do_auto_eval(model, auto_eval_dataloader)
    wandb.log({"accuracy/final": accuracy, "loss/final": eval_loss})
    save_model("final", model, config.lora)


def main(config, local_rank, world_size):
    if config.run_name is not None:
        print(f"Loading config from {config.run_name}")
        config = load_config(config.run_name)
    pprint.pprint(config)
    seed = utils.enforce_reproducibility(config.seed)
    config.seed = seed
    model, tokenizer = load_model(config)
    print("Loaded model, beginning training.")
    train(model, tokenizer, config, local_rank, world_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Configuration")
    for field in fields(Config):
        name = field.name
        default = field.default
        field_type = field.type
        is_optional = False
        if get_origin(field_type) is Union:
            arg_types = get_args(field_type)
            is_optional = any(t == type(None) for t in arg_types)
            actual_type = [t for t in arg_types if t != type(None)][0] if is_optional else field_type

        arg_type = field_type if not is_optional else field_type.__args__[0]
        if arg_type == bool:
            parser.add_argument(f"--{name}", action="store_true")
        else:
            parser.add_argument(f"--{name}", type=arg_type, default=default)
    args = parser.parse_args()
    config = Config(**vars(args))
    if config.do_vocab_surgery:
        model, tokenizer = load_model(config)
        model, tokenizer = utils.vocab_surgery(model, tokenizer)
        tokenizer.save_pretrained("adept/fuyu-8b-slim-vocab")
        model.save_pretrained("adept/fuyu-8b-slim-vocab")
    else:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
        main(config, local_rank, world_size)
