import transformers
import glob
from collections import namedtuple, defaultdict
import os
import data, helpers
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

grouped_question_metadata = None
grouped_model_inputs = None
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
    run_name: Optional[str] = field(default=None)
    save_every_steps: int = field(default=500)
    eval_every_steps: int = field(default=500)


Data = namedtuple("Data", ["train_dataloader", "eval_dataloader", "data_collator", "dataset_for_auto_eval"])


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
    train_dataset, eval_dataset, _, eval_question_ids = full_ds.split(0.97)
    dataset_for_auto_eval = data.AI2DDatasetForEval("/home/ubuntu/ai2d", processor, eval_question_ids)

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
    return Data(train_dataloader, eval_dataloader, data_collator, dataset_for_auto_eval)


def save_model(step, model, is_lora):
    assert wandb.run is not None
    checkpoint_folder = f"/home/ubuntu/fuyu/output/{wandb.run.name}/step-{step}"
    if is_lora:
        model_path = os.path.join(checkpoint_folder, "adapter_model")
    else:
        model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
    model.save_pretrained(model_path)


def load_model(config: Config, device="cuda:0"):
    model: transformers.FuyuForCausalLM = transformers.FuyuForCausalLM.from_pretrained(
        "adept/fuyu-8b", device_map=device, torch_dtype=torch.bfloat16
    )
    checkpoint_dir = None
    if config.run_name is not None:
        run_dir = f"/home/ubuntu/fuyu/output/{config.run_name}/"
        paths = glob.glob(os.path.join(run_dir, "step-*"))
        steps = [p.split("-")[-1] for p in paths]
        if "final" in steps:
            checkpoint_dir = os.path.join(run_dir, f"step-final")
        else:
            step = max([int(s) for s in steps])
            checkpoint_dir = os.path.join(run_dir, f"step-{step}")

    model.gradient_checkpointing_enable()
    model.language_model.model.gradient_checkpointing_enable()
    if config.lora:
        return get_lora_model(model, checkpoint_dir, config)
    elif config.run_name is not None:
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

def do_auto_eval(model, step, max_questions, data_collator, dataset_for_auto_eval):
    print("Doing auto eval.")
    global grouped_model_inputs
    global grouped_question_metadata
    if grouped_model_inputs is None or grouped_question_metadata is None:
        grouped_model_inputs = defaultdict(list)
        grouped_question_metadata = dataset_for_auto_eval.by_question_id
    correct_by_question_id = {}
    for i, question_id in enumerate(tqdm(grouped_question_metadata.keys(), total=max_questions)):
        if max_questions is not None and i > max_questions:
            break
        meta = grouped_question_metadata[question_id]
        model_inputs_list = [
            dataset_for_auto_eval.get_model_inputs_for_question(meta[i])[0]
            for i in range(len(meta))
        ]
        grouped_model_inputs[question_id] = model_inputs_list
        probs = []
        # Ensure all tensors are on the same device as the model
        def chunk(l, n):
            for i in range(0, len(l), n):
                yield l[i:i + n]
        correctness = [meta[i]['isCorrect'] for i in range(len(meta))]
        probs = []
        if any([single_model_inputs['input_ids'].shape[0] > 600 for single_model_inputs in model_inputs_list]):
            chunk_size = 1
        else:
            chunk_size = 2
        for model_inputs in chunk(model_inputs_list, chunk_size): 
            collated = data_collator(model_inputs)
            with torch.inference_mode(), torch.autocast('cuda'):
                logits = model(**helpers.clean(collated)).logits
                shifted_logits = logits[:, :-1, :].contiguous().cuda()
                shifted_labels = collated['labels'][..., 1:].contiguous().cuda()

                log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
                non_ignore_indices = shifted_labels.ne(-100)
                gather_indices = torch.where(non_ignore_indices, shifted_labels, 0)
                selected_log_probs = torch.gather(log_probs, 2, gather_indices.unsqueeze(-1)).squeeze(-1)
                selected_log_probs = torch.where(non_ignore_indices, selected_log_probs, 0.0)
                sequence_log_prob = torch.sum(selected_log_probs, dim=1)
                result = torch.exp(sequence_log_prob)
                probs.append(result.cpu().numpy())
        probs = [p for sublist in probs for p in sublist]
        if all([p == 0 for p in probs]):
            print(f'Question {question_id} has no valid answers.')
        else:
            is_correct = correctness[probs.index(max(probs))]
            correct_by_question_id[question_id] = is_correct
    accuracy = sum(correct_by_question_id.values()) / len(correct_by_question_id)
    print(accuracy)
    wandb.log({"step": step, "eval_accuracy": accuracy})

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
            do_auto_eval(model, step+1, None, data.data_collator, data.dataset_for_auto_eval)
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
        run_name=None,
        save_every_steps=500,
        eval_every_steps=500,
    )
    model = load_model(config)
    print("Loaded model.")
    train(model, config)


if __name__ == "__main__":
    main()
