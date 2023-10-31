from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import fuyu.utils as utils
import wandb


def do_eval(model, step, max_steps, eval_dataloader):
    losses = []
    print("Doing evaluation.")
    model.eval()
    for i, batch in enumerate(tqdm(eval_dataloader, total=max_steps)):
        if max_steps is not None and i > max_steps:
            break
        cleaned_batch = utils.prepare_inputs(batch, fdtype=torch.bfloat16)
        with torch.inference_mode(), torch.autocast("cuda"):
            loss = model(**cleaned_batch).loss
        losses.append(loss.item())
    avg_loss = sum(losses) / len(losses)
    wandb.log({"step": step, "loss/val": avg_loss})
    model.train()


def get_subbatches(batch, batch_size):
    if batch_size > 1 and batch["input_ids"].shape[1] > 600:
        subbatches = [{} for _ in range(batch["input_ids"].shape[0])]
        for k, v in batch.items():
            for j in range(len(subbatches)):
                if isinstance(v, torch.Tensor):
                    subbatches[j][k] = v[j : j + 1, ...]
                else:
                    subbatches[j][k] = [v[j]]
    elif batch_size > 2 and batch["input_ids"].shape[1] > 400:
        subbatches = [{} for _ in range(batch["input_ids"].shape[0] // 2)]
        for k, v in batch.items():
            for j in range(len(subbatches)):
                if isinstance(v, torch.Tensor):
                    subbatches[j][k] = v[j * 2 : (j + 1) * 2, ...]
                else:
                    subbatches[j][k] = [v[j * 2], v[j * 2 + 1]]
    else:
        subbatches = [batch]
    return subbatches

def get_label_probs(logits: torch.Tensor, labels: torch.Tensor):
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous().cuda()
    log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
    non_ignore_indices = shifted_labels.ne(-100)
    gather_indices = torch.where(non_ignore_indices, shifted_labels, 0)
    selected_log_probs = torch.gather(
        log_probs, 2, gather_indices.unsqueeze(-1)
    ).squeeze(-1)
    selected_log_probs = torch.where(
        non_ignore_indices, selected_log_probs, 0.0
    )
    sequence_log_prob = torch.sum(selected_log_probs, dim=1)
    probs = torch.exp(sequence_log_prob).cpu().numpy()
    return probs


def do_auto_eval(model, dataloader: DataLoader):
    results_by_question_id = defaultdict(list)
    losses = []
    model.eval()
    def compute_accuracy():
        correct_by_question_id = {}
        for question_id, results in results_by_question_id.items():
            if len(results) < 4:
                continue
            results.sort(key=lambda x: x[0])
            correct_by_question_id[question_id] = results[-1][1]
        accuracy = sum(correct_by_question_id.values()) / len(correct_by_question_id)
        return accuracy

    batch_size = dataloader.batch_size
    total = len(dataloader)
    for batch in tqdm(dataloader, total=total):
        subbatches = get_subbatches(batch, batch_size)
        for subbatch in subbatches:
            with torch.inference_mode(), torch.autocast("cuda"):
                try:
                    outputs = model(**utils.prepare_inputs(subbatch))
                except Exception as e:
                    print(subbatch["input_ids"].shape)
                    raise e
                probs = get_label_probs(outputs.logits, subbatch['labels'])
            is_correct = subbatch["is_correct"].cpu().numpy()
            loss = outputs.loss.cpu().numpy()
            question_ids = subbatch["question_id"]
            for j, question_id in enumerate(question_ids):
                results_by_question_id[question_id].append((probs[j], is_correct[j]))
                if is_correct[j]:
                    losses.append(loss[j] if len(loss.shape) else loss.item())
    model.train()
    return compute_accuracy(), sum(losses) / len(losses)
