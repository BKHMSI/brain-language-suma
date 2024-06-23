import os
import torch
import random 
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob 
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from modeling_llama import LlamaCustomConfig, LlamaCustomForCausalLM, LlamaModel

from utils import setup_hooks, get_layer_names, write_pickle

os.environ["TOKENIZERS_PARALLELISM"] = "False"

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class Fed10_LocLangDataset(Dataset):
    def __init__(self, 
                 dirpath, is_pretrained):
        paths = glob(f"{dirpath}/*.csv")
        vocab = set()
        self.is_pretrained = is_pretrained

        data = pd.read_csv(paths[0])
        for path in paths[1:]:
            run_data = pd.read_csv(path)
            data = pd.concat([data, run_data])

        data["sent"] = data["stim2"].apply(str.lower)

        vocab.update(data["stim2"].apply(str.lower).tolist())
        for stimuli_idx in range(3, 14):
            data["sent"] += " " + data[f"stim{stimuli_idx}"].apply(str.lower)
            vocab.update(data[f"stim{stimuli_idx}"].apply(str.lower).tolist())

        self.vocab = sorted(list(vocab))
        self.w2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2w = {i: w for i, w in enumerate(self.vocab)}

        self.sentences = data[data["stim14"]=="S"]["sent"]
        self.non_words = data[data["stim14"]=="N"]["sent"]

    def tokenize(self, sent):
        return torch.tensor([self.w2idx[w]+20_000 for w in sent.split()])

    def __getitem__(self, idx):
        if self.is_pretrained:
            return self.sentences.iloc[idx].strip(), self.non_words.iloc[idx].strip()
        else:
            return self.tokenize(self.sentences.iloc[idx].strip()), self.tokenize(self.non_words.iloc[idx].strip())
        
    def __len__(self):
        return len(self.sentences)
    
    def vocab_size(self):
        return len(self.vocab) + 20_000

def extract_representations(model, 
    input_ids, 
    attention_mask,
    layer_names,
    embed_agg,
    extract_all_activations=False,
):
    
    batch_activations = {layer_name: [] for layer_name in layer_names}
    if not extract_all_activations:
        hooks, layer_representations = setup_hooks(model, layer_names)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    for sample_idx in range(len(input_ids)):
        seq_len = attention_mask[sample_idx].sum()
        for layer_idx, layer_name in enumerate(layer_names):

            if extract_all_activations:
                activations = outputs.internal_states[layer_idx, sample_idx, -1].cpu()
            else:
                if embed_agg == "mean":
                    activations = layer_representations[layer_name][sample_idx][-seq_len:].mean(dim=0).cpu()
                elif embed_agg == "last-token":
                    activations = layer_representations[layer_name][sample_idx][-1].cpu()
                elif embed_agg == "all":
                    activations = layer_representations[layer_name][sample_idx].cpu()
                else:
                    raise ValueError(f"{embed_agg} not implemented")
            
            batch_activations[layer_name] += [activations]

    if not extract_all_activations:
        for hook in hooks:
            hook.remove()

    return batch_activations

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('--model-name',  type=str,
                        default="meta-llama/Llama-2-7b-hf", help='path of config file')
    parser.add_argument('--dataset-name',  type=str,
                        default="fedorenko10", help='dataset name')
    parser.add_argument('--dataset-subset',  type=str,
                        default="wikitext-2-v1", help='subset name of dataset')
    parser.add_argument('--pretrained',  action='store_true',
                        help='use pretrained weights')
    parser.add_argument('--tokenizer-pretrained',  action='store_true',
                        help='use pretrained tokenizer')
    parser.add_argument('--max-samples',  type=int,
                        default=1000, help='maximum number of samples')
    parser.add_argument('--seed',  type=int,
                        default=42, help='seed')
    parser.add_argument('--batch-size',  type=int,
                        default=32, help='batch size')
    parser.add_argument('--embed-agg',  type=str,
                        default="mean", help='embedding aggregation')
    parser.add_argument('--num-attn-heads',  type=int,
                        default=32, help='number of attention heads')
    parser.add_argument('--num-blocks',  type=int,
                        default=1, help='number of blocks')
    parser.add_argument('--num-cycles',  type=str,
                        default="1", help='number of cycles')
    parser.add_argument('--init-range',  type=float,
                        default=0.02, help='initialization range')    
    parser.add_argument('--nopos',  action='store_true',
                        help='no positional encoding flag') 
    parser.add_argument('--overwrite',  action='store_true',
                        help='overwrite flag') 
    parser.add_argument('--cuda',  type=int,
                        default=0, help='number of blocks')
    args = parser.parse_args()

    seed_everything(seed=args.seed)
    num_samples = 240
    extract_all_activations = False # For recurrent models, extract all activations at each pass

    model_name_ = args.model_name.split("/")[-1]
    if "suma" in args.model_name:
        num_cycles = args.num_cycles if "suma-1" in args.model_name else "dynamic"
        if "dynamic" not in num_cycles:
            num_cycles = int(num_cycles)
    else:
        num_cycles = args.num_cycles

    savepath = f"dumps/reps_model={model_name_}_dataset={args.dataset_name}_pretrained={args.pretrained}_agg={args.embed_agg}_seed={args.seed}_nheads={args.num_attn_heads}_nblocks={args.num_blocks}_ncycles={num_cycles}_init-range={args.init_range}_tok={args.tokenizer_pretrained}.pkl"
    if os.path.exists(savepath) and not args.overwrite:
        print(f"> Already Exists: {savepath}")
        exit()

    if "suma" in args.model_name:
        num_attn_heads = args.num_attn_heads
        model_config = LlamaCustomConfig(
            hidden_size=4096,
            num_hidden_layers=args.num_blocks,
            num_attention_heads=num_attn_heads,
            num_key_value_heads=num_attn_heads,
        )

        init_range = args.init_range
                
        model_config.num_cycles = num_cycles
        model_config._attn_implementation = "eager"
        model_config.initializer_range = init_range
        model_config.use_cache = False
        model_config.attn_arch = "default"
        model_config.layer_mode = "encoder"
        model_config.mask_path = None

        if args.model_name == "suma":
            print(f"> Using suma with {model_config.num_cycles} Cycles")
            model_config.use_mlp = True
            model_config.add_residual = True
            model_config.use_input_layernorm = True
            model_config.use_post_layernorm = True
            model_config.use_pos_emb = True
        else:
            block_arch_parts = args.model_name.lower().split("-")
            model_config.use_mlp = "mlp" in block_arch_parts
            model_config.add_residual = "res" in block_arch_parts
            model_config.use_input_layernorm = "ln1" in block_arch_parts
            model_config.use_post_layernorm = "ln2" in block_arch_parts
            model_config.use_pos_emb = "pos" in block_arch_parts

        if args.nopos:
            print(f"> Removing Positional Encoding")
            model_config.use_pos_emb = False

        if args.pretrained:
            model_config.layer_mode = "localization"
            model_config.decoder_num_layers = 2
            model_config.use_input_layernorm = True
            model_config.add_residual = True
            model_config.use_mlp = False
            model_config.use_post_layernorm = False
            model_config.use_pos_emb = False
            model_config.attn_arch = "default"
            mask_path_templ = "l-mask_model={model_name}_dataset=fed10_pretrained=False_agg={agg_method}_nunits={language_mask_nunits}_seed={seed}_nheads={num_heads}_nblocks={num_blocks}_ncycles={num_cycles}_init-range=0.02_tok=True_v2.pkl"
            model_config.mask_path = mask_path_templ.format(
                model_name="suma-1-ln-attn",
                agg_method="last-token",
                language_mask_nunits=4096,
                seed=42,
                num_heads=num_attn_heads,
                num_blocks=1,
                num_cycles=num_cycles
            )

            model_name_or_path = "suma-1-ln-attn-loc-3_seed=42_nunits=4096_context=512_nheads=512_ncycles=1_dec=2_hdim=4096_dataset=wikitext103"
            model = LlamaCustomForCausalLM.from_pretrained(model_name_or_path, config=model_config)
        else:
            model = LlamaModel(model_config)

    elif args.pretrained:
        print("> Using Pretrained Weights")
        config = AutoConfig.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="cpu", torch_dtype=torch.float32)
    else:
        print("> Using Untrained Weights")
        config = AutoConfig.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)

    if args.pretrained or args.tokenizer_pretrained:
        tokenizer_name = "meta-llama/Llama-2-7b-hf" if "suma" in args.model_name or "random-proj" in args.model_name else args.model_name
        print(f"> Using {tokenizer_name} Tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
    else:
        print("> Using word-based Tokenizer")
        tokenizer = None

    if args.model_name.strip() == "meta-llama/Llama-2-7b-hf":
        args.model_name = "Llama-2-7b-hf"

    if args.dataset_name == "fedorenko10":
        dirpath = f"fedorenko10_stimuli"
        lang_dataset = Fed10_LocLangDataset(dirpath, args.tokenizer_pretrained)
        num_samples = 240
    else:
        raise ValueError(f"Dataset {args.dataset_name} not implemented")

    layer_names: list[str] = get_layer_names(model_name_, None)
    if "suma" in args.model_name:
        hidden_dim = 4096 
    else:
        hidden_dim = config.n_embd if "gpt" in args.model_name else config.hidden_size
    
        
    batch_size = args.batch_size

    print(layer_names)

    lang_dataloader = DataLoader(lang_dataset, batch_size=batch_size, num_workers=16)

    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else "cpu"
    print(f"> Using Device: {device}")

    print(model)
    model.eval()
    model.to(device)

    final_layer_representations = {
        "sentences": {layer_name: np.zeros((num_samples, hidden_dim)) for layer_name in layer_names},
        "non-words": {layer_name: np.zeros((num_samples, hidden_dim)) for layer_name in layer_names}
    }
    
    for batch_idx, batch_data in tqdm(enumerate(lang_dataloader), total=len(lang_dataloader)):

        if args.dataset_name == "fedorenko10":
            sents, non_words = batch_data
            if args.pretrained or args.tokenizer_pretrained:
                sent_tokens = tokenizer(sents, truncation=True, max_length=12, return_tensors='pt').to(device)
                non_words_tokens = tokenizer(non_words, truncation=True, max_length=12, return_tensors='pt').to(device)
                assert sent_tokens.input_ids.size(1) == non_words_tokens.input_ids.size(1)
            else:
                assert sents.size(1) == non_words.size(1)
                sent_tokens = {
                    "input_ids": sents.to(device),
                    "attention_mask": torch.ones_like(sents).to(device)
                }

                non_words_tokens = {
                    "input_ids": non_words.to(device),
                    "attention_mask": torch.ones_like(non_words).to(device)
                }

        else:
            sent_tokens = tokenizer(batch_data, padding=True, return_tensors='pt').to(device)
        
            max_seq_len = sent_tokens.input_ids.shape[1]

            non_words_input_ids = torch.ones_like(sent_tokens.input_ids) * tokenizer.pad_token_id
            for sample_idx, attn_mask in enumerate(sent_tokens.attention_mask):
                seq_len = attn_mask.sum()
                non_words_input_ids[sample_idx, -seq_len:] = torch.randint(5, len(tokenizer)-5, size=(seq_len,)) # 5 and -5 to avoid special tokens
            
            non_words_tokens = {
                "input_ids": non_words_input_ids.to(device),
                "attention_mask": sent_tokens.attention_mask,
            }
        
        batch_real_actv = extract_representations(model, sent_tokens["input_ids"], sent_tokens["attention_mask"], layer_names, args.embed_agg, extract_all_activations)
        batch_rand_actv = extract_representations(model, non_words_tokens["input_ids"], non_words_tokens["attention_mask"], layer_names, args.embed_agg, extract_all_activations)

        for layer_name in layer_names:
            final_layer_representations["sentences"][layer_name][batch_idx*batch_size:(batch_idx+1)*batch_size] = torch.stack(batch_real_actv[layer_name]).numpy()
            final_layer_representations["non-words"][layer_name][batch_idx*batch_size:(batch_idx+1)*batch_size] = torch.stack(batch_rand_actv[layer_name]).numpy()

    print(f"> Saving @ {savepath}")
    write_pickle(savepath, final_layer_representations)