import transformers
import importlib
from fuyu import ai2d_data, helpers
import torch
from PIL import Image
from transformers import (
    Seq2SeqTrainer,
    BitsAndBytesConfig
)
import bitsandbytes as bnb
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm import tqdm
pretrained_path = "adept/fuyu-8b"
tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_path)
processor = transformers.FuyuProcessor(image_processor=transformers.FuyuImageProcessor(), tokenizer=tokenizer)

!pwd

quantize = True
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
                bnb_4bit_quant_type='nf4',
            ),
        torch_dtype=torch.bfloat16)
    prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
else:
    model = transformers.FuyuForCausalLM.from_pretrained(
        "adept/fuyu-8b",
        device_map="auto",
        torch_dtype=torch.bfloat16)


#tokenizer.bos_token_id - 1011

#model._hf_peft_config_loaded = True
#model.language_model.model._hf_peft_config_loaded = True
#model.language_model.model.layers = model.language_model.model.layers[:4]
model.gradient_checkpointing_enable()
model.language_model.model.gradient_checkpointing_enable()

importlib.reload(ai2d_data)
full_ds = ai2d_data.AI2DDataset("/home/ubuntu/fuyu/ai2d")
train_dataset, val_dataset = full_ds.split(0.9)
data_collator = ai2d_data.DataCollatorForMultiModal(processor=processor, include_labels=True)

lora_module_names = set()
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        names = name.split('.')
        lora_module_names.add(names[0] if len(names)==1 else names[-1])

lora_module_names.remove('lm_head')
lora_module_names.remove('vision_embed_tokens')
config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=list(lora_module_names),
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

for name, module in model.named_modules():
    if isinstance(module, LoraLayer):
        module = module.to(torch.bfloat16)
    if 'norm' in name:
        module = module.to(torch.float32)
    if 'lm_head' in name or 'embed_tokens' in name:
        if hasattr(module, 'weight'):
            if module.weight.dtype == torch.float32:
                module = module.to(torch.bfloat16)
                

per_device_batch_size = 1
learning_rate = 1e-4
scheduler_type = 'constant'
warmup_steps = 0
gradient_accumulation_steps = 1

train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=per_device_batch_size)
eval_dataloader = DataLoader(val_dataset, shuffle=False, collate_fn=data_collator, batch_size=per_device_batch_size)
max_train_steps = len(train_dataloader) # * n_devices 

opt_group_params = [
    {'params': [p for n, p in model.named_parameters() if 'lora' in n],
     'weight_decay': 0.0
    },
]
optimizer = bnb.optim.AdamW(opt_group_params, lr=learning_rate)
lr_scheduler = get_scheduler(
    name=scheduler_type,
    optimizer=optimizer,
    num_warmup_steps = warmup_steps * gradient_accumulation_steps,
    num_training_steps = max_train_steps * gradient_accumulation_steps,
)
model = model.train()
model.print_trainable_parameters()

for step, batch in enumerate(train_dataloader):
    if step > 100:
        break
    cleaned_batch = helpers.clean(batch, fdtype=torch.bfloat16)
    #cleaned_batch['image_patches'] = cleaned_batch['image_patches'].float()
    with torch.autocast('cuda'):
        loss = model(**cleaned_batch).loss
    print('--', torch.cuda.memory_allocated()/1e9, 'GB')
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    print(step, round(loss.item(), 4), torch.cuda.memory_allocated()/1e9, 'GB')
    del loss
    helpers.clear_mem()


torch.cuda.memory_allocated()/1e9

if 'handles' in locals() or 'handles' in globals():
    for handle in handles.values():
        handle.remove()
else:
    handles = {}
for i, layer in enumerate(model.base_model.language_model.model.layers):
    x = i
    handles[i] = layer.register_forward_hook(lambda mod, inp, out: print(torch.cuda.memory_allocated()))

del outputs

helpers.clear_mem()
!nvidia-smi

for name, param in model.named_parameters():
    if param.dtype != torch.bfloat16:
        print(name, param.dtype)



