#### Model Paramaters
seed=42
num_heads=32
num_units=4096
init_range=0.02
embed_agg=last-token
model=meta-llama/Llama-2-7b-hf

#### Run Localization for Pretrained Models
python extract.py --model-name $model --seed $seed --embed-agg $embed_agg --num-attn-heads $num_heads --init-range $init_range --overwrite --pretrained --tokenizer-pretrained
python localize_fed10.py --model-name $model --seed $seed --embed-agg $embed_agg --num-units $num_units --num-attn-heads $num_heads --init-range $init_range --pretrained --tokenizer-pretrained

#### Run Localization for Untrained Models
for seed in 10 20 30 42 50
do 
    python extract.py --model-name $model --seed $seed --embed-agg $embed_agg --num-attn-heads $num_heads --init-range $init_range --tokenizer-pretrained
    python localize_fed10.py --model-name $model --seed $seed --embed-agg $embed_agg --num-units $num_units --num-attn-heads $num_heads --init-range $init_range --tokenizer-pretrained
done
