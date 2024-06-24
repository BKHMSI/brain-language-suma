import os 
import torch
import random 
import argparse
import numpy as np
from glob import glob 
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm

from extract import extract_representations
from utils import get_layer_names, read_pickle, write_pickle

from llama_custom import LlamaCustomConfig, LlamaCustomForCausalLM


os.environ["TOKENIZERS_PARALLELISM"] = "False"

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def read_text(path):
    with open(path, "r") as fin:
        text = [line.strip() for line in fin.readlines() if line.strip() != ""]
    return ' '.join(text).lower()

def read_group(paths):
    texts = [read_text(path) for path in paths]
    return texts

class Fed10_StimuliDataset(Dataset):
    def __init__(self, texts_1, vocab=None, tokenizer_pretrained=True):
        self.texts_1 = texts_1
        self.vocab = vocab 
        self.tokenizer_pretrained = tokenizer_pretrained
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}

    def tokenize(self, sent):
        return torch.tensor([self.word2idx[w]+25_000 for w in sent.split()[:11]])
    
    def __getitem__(self, idx):
        return self.texts_1[idx] if self.tokenizer_pretrained else self.tokenize(self.texts_1[idx])
        
    def __len__(self):
        return len(self.texts_1)
    
def build_vocab(data):
    vocab = set()
    for text in data:
        vocab.update(text.split())
    return vocab

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('--model-name',  type=str,
                        default="gpt2", help='path of config file')
    parser.add_argument('--pretrained',  action='store_true',
                        help='use pretrained weights')
    parser.add_argument('--tokenizer-pretrained',  action='store_true',
                        help='use pretrained tokenizer')
    parser.add_argument('--num-units',  type=int,
                        default=4096, help='batch size')
    parser.add_argument('--embed-agg',  type=str,
                        default="last-token", help='embedding aggregation method')
    parser.add_argument('--seed',  type=int,
                        default=42, help='seed')
    parser.add_argument('--batch-size',  type=int,
                        default=32, help='batch size')
    parser.add_argument('--num-attn-heads',  type=int,
                        default=32, help='number of attention heads')
    parser.add_argument('--num-blocks',  type=int,
                        default=1, help='number of blocks')
    parser.add_argument('--num-cycles',  type=int,
                        default=1, help='number of cycles')
    parser.add_argument('--init-range',  type=float,
                        default=0.02, help='initialization range')    
    parser.add_argument('--nopos',  action='store_true',
                        help='no positional encoding flag') 
    parser.add_argument('--overwrite',  action='store_true',
                        help='overwrite flag') 
    args = parser.parse_args()

    seed_everything(seed=args.seed)

    dirpath = "fedorenko10_stimuli"

    args.tokenizer_pretrained = args.pretrained or args.tokenizer_pretrained

    if args.model_name == "random-proj":
        model = RandomProjection(4096, 16384, vocab_size=32000)
    elif "suma" in args.model_name:
        print("> Using suma Model")
        config = LlamaCustomConfig(
            hidden_size=4096,
            num_hidden_layers=1,
            num_attention_heads=args.num_attn_heads,
            num_key_value_heads=args.num_attn_heads,
        )

        config.embeddings_only = False

        config.num_attention_heads = args.num_attn_heads 
        config.num_hidden_layers = args.num_blocks

        config.attn_arch = "default"
        block_arch = "ln1-res"

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

        config.num_cycles = args.num_cycles
        config._attn_implementation = "eager"
        config.initializer_range = args.init_range
        config.use_cache = False
        
        model = LlamaCustomForCausalLM(config=config)

    elif args.pretrained:
        print("> Using Pretrained Weights")
        config = AutoConfig.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="cpu", torch_dtype=torch.float32)
    else:
        print("> Using Untrained Weights")
        config = AutoConfig.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)

    if args.tokenizer_pretrained or args.pretrained:
        print(f"> Using Pretrained Tokenizer: {args.model_name}")
        tokenizer_name = "meta-llama/Llama-2-7b-hf" if "suma" in args.model_name or "random-proj" in args.model_name else args.model_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
    else:
        print(f"> Using Word-Based Tokenizer")
        args.tokenizer_pretrained = False
        tokenizer = None 

    layer_names: list[str] = get_layer_names(args.model_name)

    seeds = [42] if args.pretrained else [10, 20, 30, 42, 50]
    n_units = args.num_units
    all_vocab = set()

    for group in ["1_sent", "2_words",  "3_jabsent", "4_jabwords"]:
        paths_1 = sorted(glob(os.path.join(dirpath, group, "*.txt")))
        texts_1 = read_group(paths_1)
        all_vocab.update(build_vocab(texts_1))

    all_vocab = sorted(list(all_vocab))

    for n_units in [4096]:
        
        for seed in seeds:

            results = []
            save_path = f"results/sjwn_magnitude_model={args.model_name}_pretrained={args.pretrained}_nunits={n_units}_nh={args.num_attn_heads}_seed={seed}_tok={args.tokenizer_pretrained}_v2.pkl"

            if os.path.exists(save_path) and not args.overwrite:
                print(f"> File Exists: {save_path}")
                continue 

            print(f"> Num Units: {n_units}")
            lang_mask_path = f"dumps/l-mask_model={args.model_name}_dataset=fed10_pretrained={args.pretrained}_agg={args.embed_agg}_nunits={n_units}_seed={seed}_nheads={args.num_attn_heads}_nblocks={args.num_blocks}_ncycles={args.num_cycles}_init-range=0.02_tok={args.tokenizer_pretrained}_v2.pkl"
            lang_mask_path = f"dumps/l-mask_model={args.model_name}_dataset=fed10_pretrained={args.pretrained}_agg={args.embed_agg}_nunits={n_units}_seed={seed}_nheads={args.num_attn_heads}_nblocks={args.num_blocks}_ncycles={args.num_cycles}_init-range=0.02_tok={args.tokenizer_pretrained}_v2.pkl"
            if not os.path.exists(lang_mask_path):
                lang_mask_path = f"dumps/language-mask_model={args.model_name}_dataset=fedorenko10_pretrained={args.pretrained}_agg={args.embed_agg}_nunits={n_units}_seed={seed}_nheads={args.num_attn_heads}_nblocks={args.num_blocks}_ncycles={args.num_cycles}_init-range=0.02_v2.pkl"
            
            print(lang_mask_path)
            language_mask = read_pickle(lang_mask_path).astype(np.bool_)

            rand_mask = np.zeros_like(language_mask).reshape(-1)
            rand_mask[np.random.choice(rand_mask.shape[0], language_mask.sum(), replace=False)] = 1
            rand_mask = rand_mask.reshape(language_mask.shape)


            for group in [
                "1_sent",
                "2_words", 
                "3_jabsent",
                "4_jabwords",
            ]:

                paths_1 = sorted(glob(os.path.join(dirpath, group, "*.txt")))

                texts_1 = read_group(paths_1)

                dataset = Fed10_StimuliDataset(texts_1, vocab=all_vocab, tokenizer_pretrained=args.tokenizer_pretrained)
                dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

                device = 'cuda:0' if torch.cuda.is_available() else "cpu"
                print(f"> Using Device: {device}")

                print(f"> Extracting Representations for {group}")

                model.to(device)

                if "suma" in args.model_name:
                    hidden_dim = 4096 
                elif args.model_name == 'random-proj':
                    hidden_dim = 16384
                else:
                    hidden_dim = config.n_embd if "gpt" in args.model_name else config.hidden_size

                all_activations_1 = np.empty((len(texts_1), n_units))

                all_activations_rand_1 = np.empty((len(texts_1), n_units))

                for batch_idx, (sentences_1) in tqdm(enumerate(dataloader), total=len(dataloader)):

                    batch_activations_1 = np.empty((len(sentences_1), len(layer_names), hidden_dim))

                    if args.tokenizer_pretrained or args.pretrained:
                        sent_tokens = tokenizer(sentences_1, truncation=True, max_length=11, return_tensors='pt').to(device)
                    else:
                        sent_tokens = {
                            "input_ids": sentences_1.to(device),
                            "attention_mask": torch.ones_like(sentences_1).to(device)
                        }

                    batch_per_layer_activations_1 = extract_representations(model, sent_tokens["input_ids"], sent_tokens["attention_mask"], layer_names, args.embed_agg)

                    for layer_idx, layer_name in enumerate(layer_names):
                        batch_activations_1[:, layer_idx, :] = torch.stack(batch_per_layer_activations_1[layer_name]).numpy()

                    all_activations_1[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size] = batch_activations_1[:, language_mask]
                    all_activations_rand_1[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size] = batch_activations_1[:, rand_mask]

                results.append({
                    "group": group,
                    "lang_activations": all_activations_1,
                    "rand_activations": all_activations_rand_1,
                })

            write_pickle(save_path, results)