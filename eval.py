from collections import defaultdict
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)
from tqdm import tqdm

import utils


def get_subbatches(batch, batch_size, gpu_poor=False):
    if not gpu_poor:
        return [batch]
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


def likelihood_eval(model, dataloader, rank, world_size):
    model.eval()
    id_tensors = []
    probs_tensors = []
    is_correct_tensors = []

    batch_size = dataloader.batch_size
    total = len(dataloader)

    with torch.inference_mode():
        for batch in tqdm(dataloader, total=total, disable=(rank != 0)):
            subbatches = get_subbatches(batch, batch_size)
            for subbatch in subbatches:
                autocast_context = (
                    torch.autocast("cuda")
                    if not isinstance(model, FSDP)
                    else nullcontext()
                )
                with autocast_context:
                    try:
                        outputs = model(**utils.prepare_inputs(subbatch, model.device))
                        probs = get_label_log_probs(
                            outputs.logits.float(), subbatch["labels"]
                        )
                    except torch.cuda.OutOfMemoryError as e:
                        print(
                            "Cuda OOM on input with shape", subbatch["input_ids"].shape
                        )
                        raise e
                is_correct = subbatch["is_correct"]
                question_ids = subbatch["question_id"]
                id_tensors.append(question_ids)
                is_correct_tensors.append(is_correct)
                probs_tensors.append(probs)

    flat_probs = torch.cat(probs_tensors).to(rank)  # Should not be necessary.
    flat_ids = torch.cat(id_tensors).to(rank)
    flat_correct = torch.cat(is_correct_tensors).to(rank)

    # TODO: see if we can remove this length check, presumably if the dataloaders
    # were of different lengths we'd see hangs?
    our_length = torch.tensor(flat_ids.shape[0]).to(rank)
    all_lens = [our_length for _ in range(world_size)]
    dist.all_gather(all_lens, our_length)

    gather_probs_list = (
        [torch.zeros(length.item()).to("cuda:0") for length in all_lens]
        if rank == 0
        else None
    )
    gather_ids_list = (
        [torch.zeros(length.item()).long().to("cuda:0") for length in all_lens]
        if rank == 0
        else None
    )
    gather_correct_list = (
        [torch.zeros(length.item()).bool().to("cuda:0") for length in all_lens]
        if rank == 0
        else None
    )

    dist.gather(flat_probs, gather_probs_list, dst=0)
    dist.gather(flat_ids, gather_ids_list, dst=0)
    dist.gather(flat_correct, gather_correct_list, dst=0)
    if rank == 0:
        all_probs = torch.cat(gather_probs_list)
        all_ids = torch.cat(gather_ids_list)
        all_correct = torch.cat(gather_correct_list)
        by_question_id = defaultdict(list)
        log_probs = []
        for prob, id, correct in zip(all_probs, all_ids, all_correct):
            by_question_id[id.item()].append((prob.item(), correct.item()))
            if correct:
                log_probs.append(prob.item())

        correct_by_question_id = {}
        for question_id, results in by_question_id.items():
            results.sort(key=lambda x: x[0])
            if len(results):
                correct_by_question_id[question_id] = float(results[-1][1])

        nll_loss = -sum(log_probs) / len(log_probs)
        acc = sum(correct_by_question_id.values()) / len(correct_by_question_id)
        return acc, nll_loss
    else:
        return None, None


def greedy_eval(model, dataloader, rank, world_size):
    model.eval()
    batch_size = dataloader.batch_size
    total = len(dataloader)
    correct_tensors = []
    loss_tensors = []

    autocast_context = (
        torch.autocast("cuda") if not isinstance(model, FSDP) else nullcontext()
    )
    with torch.inference_mode():
        for i, batch in enumerate(tqdm(dataloader, total=total, disable=(rank != 0))):
            subbatches = get_subbatches(batch, batch_size)
            for subbatch in subbatches:
                with autocast_context:
                    try:
                        outputs = model(**utils.prepare_inputs(subbatch, model.device))
                    except torch.cuda.OutOfMemoryError as e:
                        print(
                            "Cuda OOM on input with shape", subbatch["input_ids"].shape
                        )
                        raise e
                labels = subbatch["labels"]
                b = labels.shape[0]
                logits = outputs.logits.float()
                loss = outputs.loss.item()
                loss_tensors.append(torch.tensor([loss] * b))

                shifted_logits = logits[:, :-1, :].contiguous()
                shifted_labels = labels[..., 1:].contiguous().to(rank)
                argmax = torch.argmax(shifted_logits, dim=-1)
                for j in range(b):
                    non_ignore_indices = shifted_labels[j] != -100
                    targets = shifted_labels[j][non_ignore_indices]
                    greedy = argmax[j][non_ignore_indices]
                    is_correct = torch.all(targets == greedy).view(1)
                    correct_tensors.append(is_correct)

    flat_correct = torch.cat(correct_tensors).to(rank)
    flat_loss = torch.cat(loss_tensors).to(rank)
    rank_length = torch.tensor(flat_correct.shape[0]).to(rank)
    lengths = [rank_length for _ in range(world_size)]
    dist.all_gather(lengths, rank_length)

    if rank == 0:
        gather_correct_list = [
            torch.zeros(length.item()).bool().to("cuda:0") for length in lengths
        ]
        gather_loss_list = [
            torch.zeros(length.item()).to("cuda:0") for length in lengths
        ]
        dist.gather(flat_correct, gather_correct_list, dst=0)
        dist.gather(flat_loss, gather_loss_list, dst=0)
        all_correct = torch.cat(gather_correct_list)
        accuracy = all_correct.float().mean()
        avg_loss = torch.cat(gather_loss_list).mean()
        return accuracy, avg_loss
    else:
        dist.gather(flat_correct, None, dst=0)
        dist.gather(flat_loss, None, dst=0)
        return None, None
