from collections import defaultdict, OrderedDict
from contextlib import nullcontext
from typing import Tuple
import numpy as np
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from config import ModelConfig
import transformers
from transformers import FuyuForCausalLM, PreTrainedTokenizerFast

import utils


def load_model(
    config: ModelConfig, device_map=None
) -> Tuple[FuyuForCausalLM, PreTrainedTokenizerFast]:
    model_path = config.model_name_or_path
    if config.run_name is not None:
        model_path = utils.get_latest_checkpoint_dir(config.run_name)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    vocab = tokenizer.get_vocab()
    tokenizer.get_vocab = lambda: vocab

    if device_map is None:
        device_map = f"cuda:{0}" if config.lora else None
    model = FuyuForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=config.use_flash_attn,
        device_map=device_map,
    )
    return model, tokenizer


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
    shifted_labels = labels[..., 1:].contiguous()
    log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
    non_ignore_indices = shifted_labels.ne(-100)
    gather_indices = torch.where(non_ignore_indices, shifted_labels, 0)
    selected_log_probs = torch.gather(
        log_probs, 2, gather_indices.unsqueeze(-1)
    ).squeeze(-1)
    selected_log_probs = torch.where(non_ignore_indices, selected_log_probs, 0.0)
    return torch.sum(selected_log_probs, dim=1)


def are_labels_most_likely(logits: torch.Tensor, labels: torch.Tensor):
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()
    argmax = torch.argmax(shifted_logits, dim=-1)
    b = shifted_labels.shape[0]
    correct = torch.zeros(b).bool()
    for i in range(b):
        non_ignore_indices = shifted_labels[i] != -100
        targets = shifted_labels[i][non_ignore_indices]
        greedy = argmax[i][non_ignore_indices]
        is_correct = torch.all(targets == greedy).view(1)
        correct[i] = is_correct
    return correct


def expand_past_key_values(past_key_values, b):
    pkv = []
    for layer_key_values in past_key_values:
        k, v = layer_key_values
        k, v = k.repeat(b, 1, 1, 1), v.repeat(b, 1, 1, 1)
        pkv.append((k, v))
    return tuple(pkv)


def likelihood_eval_v2(
    model, dataloader, local_rank, world_size, gradient_checkpointing: bool = True
):
    """
    Measures the proportion of inputs where the correct answer has the highest probability under the model.
    The dataloader needs to return instances which have the standard multimodal inputs for the shared prefix
    (image + question) and a list of tokenized answers for each batch item.
    For now this only works with batch size=1.
    It computes the prefix kv cache in one forward pass, then computes likelihoods of all the answers in a batch.
    """

    corrects = []
    baseline = []
    if gradient_checkpointing:
        model.language_model.config.use_cache = True
        model.config.use_cache = True
        model.language_model.model.gradient_checkpointing_disable()
        model.gradient_checkpointing_disable()

    for i, batch in enumerate(tqdm(dataloader)):
        batch_answers = batch.pop("batch_answers")
        correct = batch.pop("correct_answers")
        batch = batch.to(local_rank, torch.bfloat16)
        if batch["input_ids"].shape[0] != 1:
            raise NotImplementedError("Only supports batch size 1")

        autocast_context = torch.autocast("cuda")
        with torch.inference_mode(), autocast_context:
            outputs = model(**batch)
        first_answer_token_probs = torch.softmax(outputs.logits[:, -1, :], -1)

        b = len(batch_answers[0])
        pkv = expand_past_key_values(outputs.past_key_values, b)
        answer_tokens = [t.view(-1) for t in batch_answers[0]]
        padded = pad_sequence(answer_tokens, batch_first=True).to(local_rank)
        with torch.inference_mode(), autocast_context:
            token_probs = torch.softmax(
                model(input_ids=padded, past_key_values=pkv).logits, -1
            )
        answer_probs = []
        answer_lengths = [a.view(-1).shape[0] for a in batch_answers[0]]

        for j, answer in enumerate(batch_answers[0]):
            flat_answer = answer.view(-1)
            answer_token_probs = token_probs[j, : len(flat_answer) - 1, :]
            full_answer_probs = torch.cat(
                [first_answer_token_probs, answer_token_probs]
            )
            full_answer_probs = full_answer_probs[
                torch.arange(len(flat_answer)), flat_answer
            ]
            answer_probs.append(torch.prod(full_answer_probs).item())

        is_model_correct = np.argmax(np.array(answer_probs)) == correct.item()

        corrects.append(is_model_correct)
        baseline_prediction = min(
            range(len(answer_lengths)), key=lambda i: answer_lengths[i]
        )
        baseline.append(baseline_prediction == correct.item())
        accuracy = sum(corrects) / len(corrects)
        if i % 25 == 0:
            print(accuracy)
            print(sum(baseline) / len(baseline))
    accuracy = sum(corrects) / len(corrects)
    if gradient_checkpointing:
        model.language_model.config.use_cache = False
        model.config.use_cache = False

        model.language_model.model.gradient_checkpointing_enable()
        model.gradient_checkpointing_enable()
    return accuracy


def likelihood_eval(model, dataloader, rank, world_size):
    model.eval()
    id_tensors = []
    probs_tensors = []
    is_correct_tensors = []

    batch_size = dataloader.batch_size
    total = len(dataloader)

    gpu_poor = False
    for batch in tqdm(dataloader, total=total, disable=(rank != 0)):
        subbatches = get_subbatches(batch, batch_size) if gpu_poor else [batch]
        batch.to(rank, torch.bfloat16)
        for subbatch in subbatches:
            autocast_context = (
                torch.autocast("cuda") if not isinstance(model, FSDP) else nullcontext()
            )
            with autocast_context, torch.inference_mode():
                try:
                    outputs = model(**subbatch)
                    probs = get_label_log_probs(
                        outputs.logits.float(), subbatch["labels"]
                    )
                except torch.cuda.OutOfMemoryError as e:
                    print("Cuda OOM on input with shape", subbatch["input_ids"].shape)
                    raise e
            id_tensors.append(subbatch["question_id"])
            is_correct_tensors.append(subbatch["is_correct"])
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

def eval(model, dataloader: DataLoader, rank: int, world_size: int):
    """
    Measures the proportion of data points where the labels are the most likely sequence, as well 
    as the loss.
    """
    model.eval()
    correct_tensors = []
    loss_lists = defaultdict(list)
    autocast_context = (
        torch.autocast("cuda") if not isinstance(model, FSDP) else nullcontext()
    )
    with torch.inference_mode():
        for batch in tqdm(dataloader, disable=(rank != 0)):
            batch = batch.to(rank, torch.bfloat16)
            with autocast_context:
                try:
                    outputs = model(**batch)
                except torch.cuda.OutOfMemoryError as e:
                    print("Cuda OOM on input with shape", batch["input_ids"].shape)
                    raise e
            labels = batch["labels"].to(rank)
            if isinstance(outputs, tuple):
                outputs, patch_predictions = outputs
                patch_loss = model.get_patch_prediction_loss(batch, patch_predictions)
                loss_lists['patch_loss'].append(patch_loss.view(1))
            logits = outputs.logits.float()
            correct_tensors.append(are_labels_most_likely(logits, labels))
            nll_loss = outputs.loss
            loss_lists['nll_loss'].append(nll_loss.view(1))

    loss_tensors = OrderedDict()
    for k in sorted(loss_lists.keys()):
        loss_tensors[k] = torch.cat(loss_lists[k]).to(rank) #type: ignore

    flat_correct = torch.cat(correct_tensors).to(rank) # type: ignore
    rank_length = torch.tensor(flat_correct.shape[0]).to(rank) #type: ignore

    lengths = [rank_length for _ in range(world_size)]
    dist.all_gather(lengths, rank_length)

    if rank == 0:
        gather_correct_list = [
            torch.zeros(length.item()).bool().to("cuda:0") for length in lengths
        ]
        dist.gather(flat_correct, gather_correct_list, dst=0)

        gather_loss_lists = {
            k: [torch.zeros_like(v) for _ in lengths]
            for k, v in loss_tensors.items()
        }
        for k, v in loss_tensors.items():
            dist.gather(v, gather_loss_lists[k], dst=0)
            loss_tensors[k] = torch.cat(gather_loss_lists[k])

        greedy_accuracy = torch.cat(gather_correct_list).float().mean()
        losses = {k: torch.cat(v).mean() for k, v in gather_loss_lists.items()}
        return {'greedy_accuracy': greedy_accuracy, **losses}
    else:
        dist.gather(flat_correct, None, dst=0)
        for k, v in loss_tensors.items():
            v = loss_tensors[k]
            dist.gather(v, None, dst=0)
        return {}


if __name__ == "__main__":
    import config
    import argparse
    import scienceqa

    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        local_rank = 0
        world_size = 1
    parser = argparse.ArgumentParser()
    model_config, eval_config = config.parse_args(
        parser, [config.ModelConfig, config.EvalConfig]
    )
    model, tokenizer = load_model(model_config, "cuda:0")
    model.eval()
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    validation_dataloader = scienceqa.get_validation_dataloader(
        eval_config, tokenizer, local_rank, world_size
    )
    results = eval(model, validation_dataloader, local_rank, world_size)
    print(results)
