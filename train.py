import transformers
import glob
from collections import namedtuple
import os
import fuyu.data as data, helpers
import torch
import wandb
from peft import get_peft_model, PeftModel
from peft.tuners.lora import LoraLayer, LoraConfig
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass, field
import bitsandbytes as bnb


@dataclass
class Config:
    per_device_batch_size: int = field(default=2)
    learning_rate: float = field(default=3e-5)
    scheduler_type: str = field(default="constant")
    warmup_steps: int = field(default=0)
    lora: bool = field(default=True)
    lora_r: int = field(default=32)
    lora_alpha: int = field(default=32)
    use_8bit_optimizer: bool = field(default=False)
    gradient_accumulation_steps: int = field(default=1)
    max_eval_steps: Optional[int] = field(default=None)
    train_on_questions: bool = field(default=False)
    resume_run_name: Optional[str] = field(default=None)
    save_every_steps: int = field(default=500)
    eval_every_steps: int = field(default=500)


Data = namedtuple("Data", ["train_dataloader", "eval_dataloader", "data_collator"])


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


def get_data(config: Config):
    tokenizer = transformers.AutoTokenizer.from_pretrained("adept/fuyu-8b")
    processor = transformers.FuyuProcessor(
        image_processor=transformers.FuyuImageProcessor(debug=False),
        tokenizer=tokenizer,
    )
    # processor.max_tokens_to_generate = 0
    full_ds = data.AI2DDataset("/home/ubuntu/ai2d", processor)
    train_dataset, eval_dataset = full_ds.split(0.97)
    data_collator = data.DataCollatorForMultimodal(pad_token_id=0)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=config.per_device_batch_size,
        pin_memory=True,
        num_workers=min(config.per_device_batch_size, 8),
    )
    eval_batch_size = 2
    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=eval_batch_size,
        pin_memory=True,
        num_workers=min(eval_batch_size, 8),
    )
    return Data(train_dataloader, eval_dataloader, data_collator)


def save_model(step, model, is_lora):
    assert wandb.run is not None
    checkpoint_folder = f"/home/ubuntu/fuyu/output/{wandb.run.name}/step-{step}"
    if is_lora:
        model_path = os.path.join(checkpoint_folder, "adapter_model")
    else:
        model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
    model.save_pretrained(model_path)


def load_model(config: Config):
    model: transformers.FuyuForCausalLM = transformers.FuyuForCausalLM.from_pretrained(
        "adept/fuyu-8b", device_map="cuda:0", torch_dtype=torch.bfloat16
    )
    checkpoint_dir = None
    if config.resume_run_name is not None:
        run_dir = f"/home/ubuntu/fuyu/output/{config.resume_run_name}/"
        paths = glob.glob(os.path.join(run_dir, "step-*"))
        steps = [p.split("-")[-1] for p in paths]
        if "final" in steps:
            raise NotImplementedError(
                f"Resuming a finished run f{config.resume_run_name} is not supported."
            )
        step = max([int(s) for s in steps])
        checkpoint_dir = os.path.join(run_dir, f"step-{step}")

    model.gradient_checkpointing_enable()
    model.language_model.model.gradient_checkpointing_enable()
    if config.lora:
        return get_lora_model(model, checkpoint_dir, config)
    elif config.resume_run_name is not None:
        raise NotImplementedError("Resuming non-finetune runs not yet implemented.")
    return model


def do_eval(model, step, max_steps, eval_dataloader):
    losses = []
    print("Doing evaluation.")
    model.eval()
    for i, batch in enumerate(tqdm(eval_dataloader)):
        if max_steps is not None and i > max_steps:
            break
        if batch is None:
            print(f"Step {i} threw an error.")
            continue
        cleaned_batch = helpers.clean(batch, fdtype=torch.bfloat16)
        with torch.inference_mode(), torch.autocast("cuda"):
            loss = model(**cleaned_batch).loss
        losses.append(loss.item())
    avg_loss = sum(losses) / len(losses)
    wandb.log({"step": step, "loss/val": avg_loss})
    model.train()


def train(
    model,
    config: Config,
):
    data = get_data(config)
    max_train_steps = len(data.train_dataloader)
    if config.lora:
        opt_group_params = [
            {
                "params": [p for n, p in model.named_parameters() if "lora" in n],
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
    if config.use_8bit_optimizer:
        optimizer = bnb.optim.AdamW8bit(
            opt_group_params, betas=(0.9, 0.95), lr=config.learning_rate
        )
    else:
        optimizer = torch.optim.AdamW(
            opt_group_params, betas=(0.9, 0.95), lr=config.learning_rate
        )

    lr_scheduler = get_scheduler(
        name=config.scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps * config.gradient_accumulation_steps,
        num_training_steps=max_train_steps * config.gradient_accumulation_steps,
    )

    wandb.init(project="fuyu", config=config.__dict__)
    model = model.train()
    losses = []
    for step, batch in enumerate(tqdm(data.train_dataloader)):
        cleaned_batch = helpers.clean(batch, fdtype=torch.bfloat16)
        with torch.autocast("cuda"):
            loss = model(**cleaned_batch).loss
            losses.append(loss.item())
        if (step + 1) % 100 == 0:
            avg_loss = sum(losses) / len(losses)
            print("step", step, "loss", avg_loss)
            losses = []
        wandb.log({"step": step, "loss/train": loss})
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        if (step + 1) % config.save_every_steps == 0:
            save_model(step + 1, model, config.lora)
        if (step + 1) % config.eval_every_steps == 0 or step == 0:
            do_eval(model, step + 1, config.max_eval_steps, data.eval_dataloader)
    save_model("final", model, config.lora)


def main():
    config = Config(
        per_device_batch_size=2,
        learning_rate=3e-5,
        scheduler_type="constant",
        warmup_steps=0,
        lora=True,
        lora_r=32,
        lora_alpha=32,
        use_8bit_optimizer=False,
        gradient_accumulation_steps=1,
        max_eval_steps=None,
        train_on_questions=False,
        resume_run_name=None,
        save_every_steps=500,
        eval_every_steps=500,
    )
    model = load_model(config)
    print("Loaded model.")
    train(model, config)


if __name__ == "__main__":
    main()
