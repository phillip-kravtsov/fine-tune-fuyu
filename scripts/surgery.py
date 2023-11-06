import argparse
import json

import torch
import transformers
from tokenizers import Tokenizer

ADEPT_VOCAB_SIZE = 262144
ADEPT_SURGERY_RANGE = (3, 70_003)


def vocab_surgery(fuyu_model, tokenizer):
    # Remove a range of presumably unused tokens from a model and a tokenizer,
    # then save the model
    print("Doing model surgery.")
    start, end = ADEPT_SURGERY_RANGE
    assert (
        tokenizer.vocab_size == ADEPT_VOCAB_SIZE
    ), "Not doing model surgery on a model with an unexpected vocab size."
    tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
    vocab = tokenizer_json["model"]["vocab"]
    new_vocab = []
    for i, tok in enumerate(vocab):
        if i < start or i >= end:
            new_vocab.append(tok)
    tokenizer_json["model"]["vocab"] = new_vocab
    tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))

    embed = fuyu_model.language_model.model.embed_tokens.weight.detach()
    hidden_size = embed.shape[1]
    new_embed = torch.concat([embed[:start, :], embed[end:, :]])
    new_vocab_size = new_embed.shape[0]
    new_embed = torch.nn.Embedding(new_vocab_size, hidden_size, _weight=new_embed)
    fuyu_model.language_model.model.embed_tokens = new_embed.to(fuyu_model.device)

    head = fuyu_model.language_model.lm_head.weight.detach()
    new_linear = torch.nn.Linear(hidden_size, new_vocab_size, bias=False)
    new_linear_weight = torch.concat([head[:start, :], head[end:, :]])
    new_linear.weight.data = new_linear_weight
    fuyu_model.language_model.lm_head = new_linear.to(fuyu_model.device)

    fuyu_model.config.update(dict(vocab_size=new_vocab_size))
    fuyu_model.language_model.config.update(dict(vocab_size=new_vocab_size))
    return fuyu_model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Remove unused tokens from a fuyu model. usage e.g.: python3 scripts/surgery.py --save_path fuyu-8b-slim-vocab"
    )
    parser.add_argument("--save_path", type=str, default="fuyu-8b-slim-vocab")
    parser.add_argument("--model_name_or_path", type=str, default="adept/fuyu-8b")
    args = parser.parse_args()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    fuyu_config = transformers.FuyuConfig.from_pretrained(args.model_name_or_path)
    model = transformers.FuyuForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=fuyu_config,
        torch_dtype=torch.bfloat16,
    )
    new_model, new_tokenizer = vocab_surgery(model, tokenizer)
    print("Saving surgery models")
    new_model.save_pretrained(args.save_path)
    new_tokenizer.save_pretrained(args.save_path)


if __name__ == "__main__":
    main()
