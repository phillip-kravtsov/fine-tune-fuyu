import torch
import transformers as tr


def main():
    base = "adept/fuyu-8b"
    config = tr.FuyuConfig.from_pretrained(base)
    update = dict(
        num_hidden_layers=2,
        num_attention_heads=32,
        hidden_size=128,
        intermediate_size=512,
    )
    name = "fuyu-tiny-random"
    config.update(update)
    config.text_config.update(update)
    small_model = tr.FuyuForCausalLM(config).to(torch.bfloat16)
    small_model.save_pretrained(name)

    tokenizer = tr.AutoTokenizer.from_pretrained(base)
    tokenizer.save_pretrained(name)


if __name__ == "__main__":
    main()
