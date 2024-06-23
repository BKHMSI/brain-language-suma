import torch 
import pickle as pkl
from collections import OrderedDict
from typing import Union, List, Tuple, Dict, Callable

def write_pickle(path, data):
    with open(path, 'wb') as f:
        pkl.dump(data, f)

def read_pickle(path):
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data 

def filter_by_length(dataset, min_len=100, max_len=128):
   dataset = dataset.filter(lambda example: min_len<=len(example["text"].split())<=max_len)
   return dataset

def data_clean(dataset, min_len, max_len):
   """We perform three steps to clean the dataset."""
   # 1- Filter out sequences with len(s)<min_len and len(s)>max_len.
   ## Hint: implement and use the `filter_by_length` function.
   dataset = filter_by_length(dataset, min_len, max_len)
   
   # 2- Remove the samples with = * = \n patterns. (* denotes any possible sequences, e.g. `= = <section> = = \n `)
   dataset = dataset.filter(lambda example:
                           not(example["text"].startswith(" = ") and
                              example["text"].endswith(" = \n")) )

   # 3- Remove Non-English sequences.
   ## Hint: You can use isEnglish(sample) to find non-English sequences.
   dataset = dataset.filter(lambda example: is_english(example["text"]))

   # 4- Lowercase all sequences.
   dataset = dataset.map(lambda x: {"text": x['text'].lower()})
   return dataset

def is_english(sample):
    try:
        sample.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True
    
def _get_layer(module, layer_name: str) -> torch.nn.Module:
    SUBMODULE_SEPARATOR = '.'
    for part in layer_name.split(SUBMODULE_SEPARATOR):
        module = module._modules.get(part)
        assert module is not None, f"No submodule found for layer {layer_name}, at part {part}"
    return module
    
def _register_hook(layer: torch.nn.Module,
                    key: str,
                    target_dict: dict):
    # instantiate parameters to function defaults; otherwise they would change on next function call
    def hook_function(_layer: torch.nn.Module, _input, output: torch.Tensor, key=key):
        # fix for when taking out only the hidden state, this is different from dropout because of residual state
        # see:  https://github.com/huggingface/transformers/blob/c06d55564740ebdaaf866ffbbbabf8843b34df4b/src/transformers/models/gpt2/modeling_gpt2.py#L428
        output = output[0] if isinstance(output, (tuple, list)) else output
        target_dict[key] = output

    hook = layer.register_forward_hook(hook_function)
    return hook

def setup_hooks(model, layer_names):
    """ set up the hooks for recording internal neural activity from the model (aka layer activations) """
    hooks = []
    layer_representations = OrderedDict()

    for layer_name in layer_names:
        layer = _get_layer(model, layer_name)
        hook = _register_hook(layer, key=layer_name,
                                target_dict=layer_representations)
        hooks.append(hook)

    return hooks, layer_representations

def get_num_blocks(model_name, num_blocks):
    if "susan-1" in model_name or "random" in model_name:
        return 1
    return {
        "gpt2": 12,
        "gpt2-medium": 24,
        "gpt2-large": 32,
        "gpt2-xl": 48,
        "Llama-2-7b-hf": 32,
        "Llama-2-13b-hf": 40,
        "susan": num_blocks,
        "susan-simple": 1,
    }[model_name]


def get_layer_names(model_name, num_blocks=None):

    if num_blocks is None:
        num_blocks = get_num_blocks(model_name, num_blocks)

    if "gpt2" in model_name:
        return [f'transformer.h.{block}.{layer_desc}' 
            for block in range(num_blocks) 
            for layer_desc in ['ln_1', 'attn', 'ln_2', 'mlp']
        ]
    elif "Llama-2" in model_name or model_name == "susan":
        return [f'model.layers.{layer_num}.{layer_desc}' 
            for layer_num in range(num_blocks) 
            for layer_desc in ["input_layernorm", "self_attn", "post_attention_layernorm", "mlp"]
        ]
    elif model_name == "susan-1-ln-attn-p2":
        return [f'lm_base.layers.{layer_num}.{layer_desc}' for layer_num in range(2) for layer_desc in ["input_layernorm", "self_attn", "post_attention_layernorm", "mlp"]]
    elif model_name == "susan-simple" or model_name == "susan-1-ln-attn" or model_name == "susan-1-l-attn" or model_name == "susan-1-ln-attn-pos":
        return [f'layers.{layer_num}.{layer_desc}' 
            for layer_num in range(num_blocks) 
            for layer_desc in ["input_layernorm", "self_attn"]
        ]
    elif model_name == "susan-1-attn":
        return [f'model.layers.{layer_num}.{layer_desc}' 
            for layer_num in range(num_blocks) 
            for layer_desc in ["self_attn"]
        ]
    elif model_name == "susan-1-mlp":
        return [f'layers.{layer_num}.{layer_desc}' 
            for layer_num in range(num_blocks) 
            for layer_desc in ["mlp"]
        ]
    elif model_name == "susan-1-ln-mlp":
        return [f'layers.{layer_num}.{layer_desc}' 
            for layer_num in range(num_blocks) 
            for layer_desc in ["input_layernorm", "mlp"]
        ]
    elif model_name == "susan-1-ln-attn-ln-mlp" or model_name == "susan-1-pos-ln-attn-ln-mlp" or "susan-1" in model_name:
        return [f'layers.{layer_num}.{layer_desc}' 
            for layer_num in range(num_blocks) 
            for layer_desc in ["input_layernorm", "self_attn", "post_attention_layernorm", "mlp"]
        ]
    elif "random" in model_name:
        return ['linear']
    else:
        raise ValueError(f"{model_name} not supported currently!")