from dataclasses import dataclass, field, fields
from typing import List, Optional, Union, get_args, get_origin


@dataclass
class ModelConfig(object):
    model_name_or_path: str = field(default="adept/fuyu-8b")
    fsdp: bool = field(default=False)
    ddp: bool = field(default=False)
    lora: bool = field(default=False)
    lora_r: int = field(default=32)
    lora_alpha: int = field(default=32)
    lora_vision: bool = field(default=False)
    use_flash_attn: bool = field(default=False)
    run_name: Optional[str] = field(default=None)
    patch_prediction: bool = field(default=False)


@dataclass
class EvalConfig(object):
    seed: int = field(default=42)
    max_eval_ids: Optional[int] = field(default=None)
    eval_batch_size: int = field(default=4)


@dataclass
class TrainingConfig(ModelConfig, EvalConfig):
    seed: int = field(default=42)
    profile: bool = field(default=False)
    alpha: float = field(default=1.0)
    train_on_questions: bool = field(default=False)
    save_every_steps: int = field(default=1000)
    eval_every_steps: int = field(default=1000)
    per_device_batch_size: int = field(default=1)
    warmup_steps: int = field(default=200)
    gradient_checkpointing: bool = field(default=False)
    learning_rate: float = field(default=3e-4)
    scheduler_type: str = field(default="constant")
    weight_decay: float = field(default=0.0)
    run_name: Optional[str] = field(default=None)
    use_packed_sampler: bool = field(default=False)
    dataset: str = field(default="ai2d")


def parse_args(parser, dataclasses: List) -> List:
    for cls in dataclasses:
        for field in fields(cls):
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
    configs = []
    for cls in dataclasses:
        keys = {f.name for f in fields(cls) if f.init}
        inputs = {k: v for k, v in vars(args).items() if k in keys}
        for k in keys:
            delattr(args, k)
        obj = cls(**inputs)
        configs.append(obj)
    return configs


def parse_training_args(parser) -> TrainingConfig:
    config: TrainingConfig = parse_args(parser, [TrainingConfig])[0]
    assert not (config.lora and config.fsdp), "Peft is not supported with FSDP."
    return config
