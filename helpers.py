import torch
import gc


def clean(model_inputs, fdtype=torch.bfloat16):
    result = {}
    for k, v in model_inputs.items():
        tensor = v.to("cuda:0")
        if tensor.dtype == torch.float32 or tensor.dtype == torch.float16:
            tensor = tensor.to(fdtype)
        result[k] = tensor
    return result


def clear_mem():
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()


def print_parameters(model):
    for name, _ in model.named_parameters():
        if any(str(i) in name for i in range(1, 10)):
            continue
        if "0" in name:
            print(name.replace("0", "%d"))
        else:
            print(name)
