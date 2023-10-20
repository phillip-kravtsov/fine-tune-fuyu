import transformers
import importlib
import ai2d_data, helpers
import torch
import wandb
from PIL import Image
from transformers import Seq2SeqTrainer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm import tqdm
import bitsandbytes as bnb


def load_model(quantize=False, gradient_checkpointing=True):
    quantize = False
    if quantize:
        model = transformers.FuyuForCausalLM.from_pretrained(
            "adept/fuyu-8b",
            device_map="auto",
            load_in_8bit=True,
            load_in_4bit=False,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=False,
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
            torch_dtype=torch.bfloat16,
        )
        prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=use_gradient_checkpointing
        )
    else:
        model = transformers.FuyuForCausalLM.from_pretrained(
            "adept/fuyu-8b", device_map="cuda:0", torch_dtype=torch.bfloat16
        )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.language_model.model.gradient_checkpointing_enable()

    return model


def get_lora_model(model, config):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    lora_module_names.remove("lm_head")
    lora_module_names.remove("vision_embed_tokens")
    config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        target_modules=list(lora_module_names),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)

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


def train(
    model,
    train_dataset,
    eval_dataset,
    data_collator,
    config,
):
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=config["per_device_batch_size"],
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=config["per_device_batch_size"],
    )
    max_train_steps = len(train_dataloader)

    opt_group_params = [
        {
            "params": [p for n, p in model.named_parameters() if "lora" in n],
            "weight_decay": 0.0,
        },
    ]
    if config['use_8bit_optimizer']:
        optimizer = bnb.optim.AdamW8bit(opt_group_params, lr=config["learning_rate"])
    else:
        optimizer = torch.optim.AdamW(opt_group_params, lr=config["learning_rate"])
    lr_scheduler = get_scheduler(
        name=config["scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=config["warmup_steps"] * config["gradient_accumulation_steps"],
        num_training_steps=max_train_steps * config["gradient_accumulation_steps"],
    )
    model = model.train()
    losses = []

    wandb.init(project="fuyu", config=config)
    for step, batch in enumerate(train_dataloader):
        if batch is None:
            print(f"Step {step} threw an error.")
            continue
        cleaned_batch = helpers.clean(batch, fdtype=torch.bfloat16)
        with torch.autocast("cuda"):
            loss = model(**cleaned_batch).loss
            losses.append(loss.item())
        if step > 0 and step % 100 == 1:
            avg_loss = sum(losses) / len(losses)
            print("step", step, "loss", avg_loss)
            losses = []
        wandb.log({"loss": loss})
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()


def main():
    config = dict(
        per_device_batch_size=8,
        learning_rate=1e-4,
        scheduler_type="constant",
        warmup_steps=0,
        lora=True,
        lora_r=16,
        lora_alpha=16,
        use_8bit_optimizer=False,
        gradient_accumulation_steps=1,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained("adept/fuyu-8b")
    processor = transformers.FuyuProcessor(
        image_processor=transformers.FuyuImageProcessor(debug=False), tokenizer=tokenizer
    )
    full_ds = ai2d_data.AI2DDataset("/home/ubuntu/ai2d")
    train_dataset, eval_dataset = full_ds.split(0.9)
    data_collator = ai2d_data.DataCollatorForMultiModal(
        processor=processor, include_labels=True
    )
    model = load_model()
    if config['lora']:
        model = get_lora_model(model, config)
    train(model, train_dataset, eval_dataset, data_collator, config)


if __name__ == "__main__":
    main()
