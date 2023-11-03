from dataclasses import dataclass, field, fields
from typing import Optional, Union, get_args, get_origin


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
    seed: int = field(default=42)
    skip_abc: bool = field(default=False)
    use_flash_attn: bool = field(default=False)
    profile: bool = field(default=False)
    use_packed_sampler: bool = field(default=False)


def parse_args(parser) -> Config:
    for field in fields(Config):
        name = field.name
        default = field.default
        field_type = field.type
        actual_type = field_type
        is_optional = False
        if get_origin(field_type) is Union:
            arg_types = get_args(field_type)
            is_optional = any(t == type(None) for t in arg_types)
            actual_type = (
                [t for t in arg_types if t != type(None)][0]
                if is_optional
                else field_type
            )
        arg_type = field_type if not is_optional else field_type.__args__[0]
        if arg_type == bool:
            assert (
                not default
            ), "Default for bools must be False to simplify store_true."
            parser.add_argument(f"--{name}", action="store_true")
        else:
            parser.add_argument(f"--{name}", type=actual_type, default=default)
    args = parser.parse_args()
    config = Config(**vars(args))
    return config
