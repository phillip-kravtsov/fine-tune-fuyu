import argparse
import glob
import json
import os
import pprint
from dataclasses import asdict, dataclass, field, fields
from typing import Optional, Union, get_origin, get_args

import bitsandbytes as bnb
import torch
import transformers
from accelerate import Accelerator
from peft import PeftModel, get_peft_model
from peft.tuners.lora import LoraConfig, LoraLayer
from tqdm import tqdm
from transformers import FuyuForCausalLM, get_scheduler

import data as data_module
import eval
import fuyu.utils as utils
import wandb

OUTPUT_DIR = "/home/ubuntu/fuyu/output"
ADEPT_VOCAB_SIZE = 262144


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
        if isinstance(module, LoraLayer):
            module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
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


def load_model(config: Config, device="cuda:0"):
    checkpoint_dir = None
    if config.run_name is not None:
        checkpoint_dir = get_latest_checkpoint_dir(config.run_name)
    model = transformers.FuyuForCausalLM.from_pretrained(
        config.model_name_or_path, device_map=device, torch_dtype=torch.bfloat16
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name_or_path)

    if config.do_vocab_surgery and tokenizer.vocab_size == ADEPT_VOCAB_SIZE:
        model, tokenizer = utils.vocab_surgery(model, tokenizer)
        tokenizer.save_pretrained("adept/fuyu-8b-slim-vocab")
        model.save_pretrained("adept/fuyu-8b-slim-vocab")

    model.gradient_checkpointing_enable()
    model.language_model.model.gradient_checkpointing_enable()
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
        optimizer = torch.optim.AdamW(
            opt_group_params, betas=(0.9, 0.95), lr=config.learning_rate
        )
    return optimizer


def train(
    model: FuyuForCausalLM,
    tokenizer,
    config: Config,
):
    train_dataloader, auto_eval_dataloader = data_module.get_data(config, tokenizer)
    max_train_steps = len(train_dataloader)
    optimizer = get_optimizer(model, config)

    lr_scheduler = get_scheduler(
        name=config.scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps * config.gradient_accumulation_steps,
        num_training_steps=max_train_steps * config.gradient_accumulation_steps,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision="bf16",
    )
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    wandb.init(project="fuyu", config=config.__dict__)
    if wandb.run is None:
        raise Exception
    save_config(config, wandb.run.name)
    model = model.train()
    completed_steps = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        cleaned_batch = utils.clean(batch, fdtype=torch.bfloat16)
        with accelerator.accumulate(model):
            try:
                loss = model(**cleaned_batch).loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            except Exception as e:
                print(cleaned_batch["input_ids"].shape)
                raise e
        wandb.log(
            {"step": step, "loss/train": loss, "completed_steps": completed_steps}
        )
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            completed_steps += 1
            if completed_steps % config.save_every_steps == 0:
                save_model(completed_steps, model, config.lora)
            if completed_steps % config.eval_every_steps == 0 or step == 0:
                model.eval()
                accuracy, eval_loss = eval.do_auto_eval(
                    model, config.max_eval_ids, auto_eval_dataloader
                )
                wandb.log(
                    {
                        "step": completed_steps,
                        "accuracy/val": accuracy,
                        "loss/val": eval_loss,
                    }
                )
    accuracy, eval_loss = eval.do_auto_eval(model, None, auto_eval_dataloader)
    wandb.log({"accuracy/final": accuracy, "loss/final": eval_loss})
    save_model("final", model, config.lora)


def main(config):
    pprint.pprint(config)
    if config.run_name is not None:
        config = load_config(config.run_name)
    seed = utils.enforce_reproducibility(config.seed)
    config.seed = seed
    model, tokenizer = load_model(config)
    print("Loaded model, beginning training.")
    train(model, tokenizer, config)


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
    main(config)
