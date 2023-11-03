import transformers as tr
import torch

def main():
    config = tr.FuyuConfig.from_pretrained('adept/fuyu-8b')
    config.update(dict(
        num_hidden_layers=6,
        num_attention_heads=32,
        hidden_size=64,
        intermediate_size=256,
    ))
    small_model = tr.FuyuForCausalLM(config).to(torch.bfloat16)
    name = 'fuyu-tiny-random'
    small_model.save_pretrained(name)

    tokenizer = tr.AutoTokenizer.from_pretrained('adept/fuyu-8b')
    tokenizer.save_pretrained(name)

if __name__ == '__main__':
    main()