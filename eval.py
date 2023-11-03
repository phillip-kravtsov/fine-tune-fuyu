from collections import defaultdict

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
import wandb


def do_eval(model, step, max_steps, eval_dataloader):
    losses = []
    print("Doing evaluation.")
    model.eval()
    for i, batch in enumerate(tqdm(eval_dataloader, total=max_steps)):
        if max_steps is not None and i > max_steps:
            break
        cleaned_batch = utils.prepare_inputs(batch, model.device, fdtype=torch.bfloat16)
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


def get_label_log_probs(logits: torch.Tensor, labels: torch.Tensor):
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous().cuda()
    log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
    non_ignore_indices = shifted_labels.ne(-100)
    gather_indices = torch.where(non_ignore_indices, shifted_labels, 0)
    selected_log_probs = torch.gather(
        log_probs, 2, gather_indices.unsqueeze(-1)
    ).squeeze(-1)
    selected_log_probs = torch.where(non_ignore_indices, selected_log_probs, 0.0)
    return torch.sum(selected_log_probs, dim=1)


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
            with torch.inference_mode():
                try:
                    outputs = model(**utils.prepare_inputs(subbatch, model.device))
                except Exception as e:
                    print(subbatch["input_ids"].shape)
                    raise e
                probs = get_label_log_probs(outputs.logits, subbatch["labels"])
            is_correct = subbatch["is_correct"].cpu().numpy()
            loss = outputs.loss.cpu().numpy()
            question_ids = subbatch["question_id"]
            for j, question_id in enumerate(question_ids):
                results_by_question_id[question_id].append((probs[j], is_correct[j]))
                if is_correct[j]:
                    losses.append(loss[j] if len(loss.shape) else loss.item())
    model.train()
    return compute_accuracy(), sum(losses) / len(losses)


def auto_eval_dist(model, dataloader, rank, world_size):
    model.eval()
    id_tensors = []
    probs_tensors = []
    is_correct_tensors = []

    batch_size = dataloader.batch_size
    total = len(dataloader)
    for i, batch in enumerate(tqdm(dataloader, total=total, disable=(rank != 0))):
        subbatches = get_subbatches(batch, batch_size)
        for subbatch in subbatches:
            with torch.inference_mode():
                try:
                    outputs = model(**utils.prepare_inputs(subbatch, model.device))
                except torch.cuda.OutOfMemoryError as e:
                    print("Cuda OOM on input with shape", subbatch["input_ids"].shape)
                    raise e
                probs = get_label_log_probs(outputs.logits.float(), subbatch["labels"])
            is_correct = subbatch["is_correct"]
            question_ids = subbatch["question_id"]
            id_tensors.append(question_ids)
            is_correct_tensors.append(is_correct)
            probs_tensors.append(probs)
    flat_probs = torch.cat(probs_tensors).to(rank)  # Should not be necessary.
    flat_ids = torch.cat(id_tensors).to(rank)
    flat_correct = torch.cat(is_correct_tensors).to(rank)

    our_length = torch.tensor(flat_ids.shape[0]).to(rank)
    all_lens = [our_length for _ in range(world_size)]
    dist.all_gather(all_lens, our_length)

    if rank == 0:
        gather_probs_list = [
            torch.zeros(length.item()).to("cuda:0") for length in all_lens
        ]
        gather_ids_list = [
            torch.zeros(length.item()).long().to("cuda:0") for length in all_lens
        ]
        gather_correct_list = [
            torch.zeros(length.item()).bool().to("cuda:0") for length in all_lens
        ]

        dist.gather(flat_probs, gather_probs_list, dst=0)
        dist.gather(flat_ids, gather_ids_list, dst=0)
        dist.gather(flat_correct, gather_correct_list, dst=0)

        all_probs = torch.cat(gather_probs_list)
        all_ids = torch.cat(gather_ids_list)
        all_correct = torch.cat(gather_correct_list)
        by_question_id = defaultdict(list)
        for prob, id, correct in zip(all_probs, all_ids, all_correct):
            by_question_id[id.item()].append((prob.item(), correct.item()))
        correct_by_question_id = {}
        for question_id, results in by_question_id.items():
            results.sort(key=lambda x: x[0])
            if len(results):
                correct_by_question_id[question_id] = float(results[-1][1])
        acc = sum(correct_by_question_id.values()) / len(correct_by_question_id)
        return acc
    else:
        dist.gather(flat_probs, None, dst=0)
        dist.gather(flat_ids, None, dst=0)
        dist.gather(flat_correct, None, dst=0)
