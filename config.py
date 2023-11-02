from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    model_name_or_path: str = field(default="adept/fuyu-8b")
    max_eval_ids: Optional[int] = field(default=500)
    train_on_questions: bool = field(default=False)
    eval_batch_size: int = field(default=4)
    save_every_steps: int = field(default=1000)
    eval_every_steps: int = field(default=1000)
    per_device_batch_size: int = field(default=1)
    learning_rate: float = field(default=3e-4)
    scheduler_type: str = field(default="constant")
    warmup_steps: int = field(default=200)
    lora: bool = field(default=False)
    lora_r: int = field(default=32)
    lora_alpha: int = field(default=32)
    lora_vision: bool = field(default=False)
    gradient_accumulation_steps: int = field(default=1)
    run_name: Optional[str] = field(default=None)
    weight_decay: float = field(default=0.01)
    do_vocab_surgery: bool = field(default=False)
    seed: Optional[int] = field(default=None)
    skip_abc: bool = field(default=False)
    use_flash_attn: bool = field(default=False)
    profile: bool = field(default=False)
