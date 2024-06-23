import os
import sys
import json

sys.path.append("brain-score-language")

import torch
import random 
import argparse
import warnings
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from brainscore_language import load_benchmark, ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject, get_layer_names, get_layer_names_v0
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM

from modeling_llama import LlamaCustomConfig, LlamaCustomForCausalLM, LlamaModel

def read_json(path):
    with open(path, 'r') as fin:
        data = json.load(fin)
    return data

warnings.filterwarnings('ignore') 
HF_TOKEN = os.environ["HF_TOKEN"]

ATTN_ARCH = [
    "default",
    "x",
    "MHA(x.xT * x)",
    "MHA(x.xT * V(x))",

    "MHA(Q(x).xT * x)",
    "MHA(x.K(x)T * x)",
    "MHA(Q(x).K(x)T * x)",
    "MHA(Q(x).K(x)T * V(x))",
    "O(MHA(Q(x).K(x)T * V(x)))",

    "MHA(Softmax(Q(x).K(x)T) * V(x))",
    "MHA(Softmax(x.K(x)T) * x)",
]

BLOCK_ARCH = [
    "default",
    "mlp",
    "ln1-mlp-res",
    "ln1-mlp-ln2-res",
    "pos-ln1-mlp-ln2-res",
    "ln1-res",
]

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def write_pickle(path, data):
    with open(path, 'wb') as f:
        pkl.dump(data, f)

def score_model(
        model_name: str,
        benchmark_name: str,
        pretrained: bool,
        cuda: int,
        seed: int = 42,
        debug: bool = False,
        overwrite: bool = False,
        custom_model: bool = False,
        use_all_units: bool = False,
        random_language_mask: bool = False,
        mask_threshold: str = "0.05",
        language_mask_nunits = None,
        embed_agg: str = "mean",
        context_size: int = 1024,
        num_heads: int = 32,
        hidden_dim: int = 4096,
        attn_arch: str = "default",
        block_arch: str = "default",
        num_blocks: int = 32,
        num_cycles: int = 1,
        init_range: float = 0.02,
        init_method: str = "normal",
        baseline_method: str = "default",
        generalization_mode: int = 0,
        word_based_tokenizer: bool = False,
):
    seed_everything(seed=seed)

    model_name_ = model_name.split("/")[-1]
    attn_arch_index = ATTN_ARCH.index(attn_arch) if attn_arch in ATTN_ARCH else -1
    block_arch_index = BLOCK_ARCH.index(block_arch) if block_arch in BLOCK_ARCH else -1

    if use_all_units:
        mask_threshold = "-1"
    if language_mask_nunits is not None:
        mask_threshold = language_mask_nunits

    if not custom_model:
        if pretrained:
            model_id = f"model={model_name_}_benchmark={benchmark_name}_pretrained={pretrained}_seed={seed}_agg={embed_agg}_baseline={baseline_method}_rand-lang-mask={random_language_mask}_nunits={mask_threshold}_genz={generalization_mode}_ridge=True"
        else:
             model_id = f"model={model_name_}_d={benchmark_name}_pretrained={pretrained}_seed={seed}_agg={embed_agg}_init-range={init_range}_baseline={baseline_method}_rand-lang-mask={random_language_mask}_nunits={mask_threshold}_genz={generalization_mode}_tok={(not word_based_tokenizer)}_ridge=True"

        if "layers" in benchmark_name:
            model_id += "-layers"

        savepath = f"dumps/scores_{model_id}-hf.pkl"
    else:
        model_id = f"model={model_name_}_d={benchmark_name}_pretrained={pretrained}_seed={seed}_agg={embed_agg}_nh={num_heads}_hd={hidden_dim}_blocks={num_blocks}_cycles={num_cycles}_attn-arch={attn_arch_index}_block-arch={block_arch_index}_init-range={init_range}_baseline={baseline_method}_rand-lang-mask={random_language_mask}_nunits={mask_threshold}_genz={generalization_mode}_tok={(not word_based_tokenizer)}_ridge=True"
        savepath = f"dumps/{model_id}.pkl"
    
    if os.path.exists(savepath) and not debug and not overwrite:
        print(f"> Run Already Exists: {savepath}")
        data = pd.read_pickle(savepath)
        print(data)
        return 

    benchmark = load_benchmark(benchmark_name)
    print(f"> Running {model_id}")

    if not custom_model:
        tokenizer_name = model_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=HF_TOKEN, truncation_side='left')
        config = AutoConfig.from_pretrained(model_name, token=HF_TOKEN)
        if not pretrained:
            print(f"> Using Init-Range: {init_range}")
            config.initializer_range = init_range
    else:

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=HF_TOKEN, truncation_side='left')
        config = LlamaCustomConfig(
            hidden_size=hidden_dim,
            num_hidden_layers=num_blocks,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
        )

        config.embeddings_only = False
        config.context_size = context_size

        config.num_attention_heads = num_heads 
        config.num_hidden_layers = num_blocks

        if attn_arch != "default":
            config.attn_arch = attn_arch
        else:
            config.attn_arch = "default"

        if block_arch == "default":
            config.use_mlp = True
            config.add_residual = True
            config.use_input_layernorm = True
            config.use_post_layernorm = True
            config.use_pos_emb = True
        else:
            block_arch_parts = block_arch.lower().split("-")
            config.use_mlp = "mlp" in block_arch_parts
            config.add_residual = "res" in block_arch_parts
            config.use_input_layernorm = "ln1" in block_arch_parts
            config.use_post_layernorm = "ln2" in block_arch_parts
            config.use_pos_emb = "pos" in block_arch_parts

        config.num_cycles = num_cycles
        config._attn_implementation = "eager"
        config.initializer_range = init_range
        config.init_method = init_method
        config.use_cache = False
        config.layer_mode = "encoder"
        config.mask_path = None

    if custom_model:
        if args.pretrained:
            config.layer_mode = "localization"
            config.decoder_num_layers = 2
            config.use_input_layernorm = True
            config.add_residual = True
            config.use_mlp = False
            config.use_post_layernorm = False
            config.use_pos_emb = False
            config.attn_arch = "default"
            mask_path_templ = "l-mask_model={model_name}_dataset=fed10_pretrained=False_agg={agg_method}_nunits={language_mask_nunits}_seed={seed}_nheads={num_heads}_nblocks={num_blocks}_ncycles={num_cycles}_init-range=0.02_tok=True_v2.pkl"
            config.mask_path = mask_path_templ.format(
                model_name="suma-1-ln-attn",
                agg_method="last-token",
                language_mask_nunits=4096,
                seed=42,
                num_heads=num_heads,
                num_blocks=1,
                num_cycles=num_cycles
            )
            model_name_or_path = "clm-ckpts/suma-1-ln-attn-loc-3_seed=42_nunits=4096_context=512_nheads=512_ncycles=1_dec=2_hdim=4096_dataset=wikitext103"
            model = LlamaCustomForCausalLM.from_pretrained(model_name_or_path, config=config)
        else:
            model = LlamaModel(config)

    elif not pretrained:
        if config.is_encoder_decoder:
            model = AutoModelForSeq2SeqLM.from_config(config)
        else:
            model = AutoModelForCausalLM.from_config(config)
    else:
        if config.is_encoder_decoder:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=HF_TOKEN, device_map="cpu")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN, device_map="cpu", config=config)            

    model.eval()

    if "layers" in model_id:
        layer_names = get_layer_names_v0(model_name_)
    else:
        layer_names = get_layer_names(model_name_, None)
     
    print("> Layer Names")
    print(layer_names)
    print()
    
    layer_scores = {}
    layer_model = HuggingfaceSubject(model_id=model_id, 
        model=model, 
        tokenizer=tokenizer, 
        region_layer_mapping={
            ArtificialSubject.RecordingTarget.language_system: layer_names
        }, 
        agg_method=embed_agg, 
        device=f"cuda:{cuda}", 
        use_past_key_values=False,
        custom_tokenizer=word_based_tokenizer,
        baseline_method=baseline_method,
        seed=seed,
        random_language_mask=random_language_mask,
        mask_threshold=mask_threshold,
        language_mask_nunits=language_mask_nunits,
        use_language_mask=(not use_all_units),
        generalization_mode=generalization_mode,
    )

    print("> Running")
    layer_scores = benchmark(layer_model)

    if not debug or overwrite:
        print("> Saving")
        write_pickle(savepath, layer_scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('--model-name',  type=str,
                        required=True, help='path of config file')
    parser.add_argument('--benchmark-name',  type=str,
                        default="Pereira2018.384sentences-cka", help='benchmark name')
    parser.add_argument('--pretrained',  action='store_true',
                        help='use pretrained weights')
    parser.add_argument('--debug',  action='store_true',
                        help='debug mode')
    parser.add_argument('--overwrite',  action='store_true',
                        help='debug mode')
    parser.add_argument('--seed', type=int, default=42,
                        help='seed number')
    parser.add_argument('--context-size', type=int, default=4096,
                        help='context size')
    parser.add_argument('--cuda', type=int, default=0,
                        help='cuda device number')
    parser.add_argument('--num-heads', type=int, default=32,
                        help='cuda device number')
    parser.add_argument('--hidden-dim', type=int, default=4096,
                        help='hidden size')
    parser.add_argument('--num-blocks', type=int, default=32,
                        help='number of hidden layers')
    parser.add_argument('--num-cycles', type=str, default="1",
                        help='number of cycles')
    parser.add_argument('--embed-agg', type=str, default='last-token',
                        help='embeddings aggregation',
                        choices=["mean", "wavg", "max", "last-token"])
    parser.add_argument('--custom-model', action='store_true',
                        help='custom model flag')
    parser.add_argument('--use-all-units', action='store_true',
                        help='use all units, do not mask language only')
    parser.add_argument('--random-language-mask', action='store_true',
                        help='random language mask flag')
    parser.add_argument('--mask-threshold', type=str, default="0.05",
                        help='language mask threshold')
    parser.add_argument('--language-mask-nunits', type=str, default=None,
                        help='number of units in language mask')
    parser.add_argument('--attn-arch', type=str, default="default",
                        help='self-attn architecture')
    parser.add_argument('--block-arch', type=str, default="default",
                        help='transformer block architecture')
    parser.add_argument('--init-range', type=float, default=0.02,
                        help='initialization variance')
    parser.add_argument('--init-method', type=str, default="normal",
                        help='initialization method')
    parser.add_argument('--baseline-method', type=str, default="default",
                        help='baseline method')
    parser.add_argument('--generalization-mode', type=int, default=0,
                        help='generalization mode')
    parser.add_argument('--word-based-tokenizer', action='store_true',
                        help='word-based tokenizer')
    args = parser.parse_args()

    if args.num_cycles.isdigit():
        args.num_cycles = int(args.num_cycles)

    score_model(**vars(args))