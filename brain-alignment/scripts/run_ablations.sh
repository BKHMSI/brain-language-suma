cuda=0
nunits=4096
num_cycles=2
num_heads=512
init_range=0.02
embed_agg=last-token

for seed in 10 20 30 42 50
do
    for benchmark_name in Pereira2018.384 Pereira2018.243 Fedorenko2016 Blank2014 Tuckute2024 Wehbe2014
    do
        python score_model.py --model-name suma-1-ln-mlp \
            --benchmark-name ${benchmark_name}-all \
            --seed $seed \
            --language-mask-nunits $nunits \
            --custom-model \
            --num-cycles $num_cycles \
            --hidden-dim 4096 \
            --num-heads $num_heads \
            --num-blocks 1 \
            --block-arch "ln1-mlp" \
            --attn-arch x \
            --embed-agg $embed_agg \
            --init-range $init_range \
            --cuda $cuda 

        python score_model.py --model-name suma-1-ln-attn \
            --benchmark-name ${benchmark_name}-all \
            --seed $seed \
            --language-mask-nunits $nunits \
            --custom-model \
            --num-cycles $num_cycles \
            --hidden-dim 4096 \
            --num-heads $num_heads \
            --num-blocks 1 \
            --block-arch "ln1-res" \
            --attn-arch default \
            --embed-agg $embed_agg \
            --init-range $init_range \
            --cuda $cuda
    

        python score_model.py --model-name suma-1-ln-attn-ln-mlp \
            --benchmark-name ${benchmark_name}-all \
            --seed $seed \
            --language-mask-nunits $nunits \
            --custom-model \
            --num-cycles $num_cycles \
            --hidden-dim 4096 \
            --num-heads $num_heads \
            --num-blocks 1 \
            --block-arch "ln1-mlp-ln2-res" \
            --attn-arch default \
            --embed-agg $embed_agg \
            --init-range $init_range \
            --cuda $cuda
    
        python score_model.py --model-name suma-1 \
            --benchmark-name ${benchmark_name}-all \
            --seed $seed \
            --language-mask-nunits $nunits \
            --custom-model \
            --num-cycles $num_cycles \
            --hidden-dim 4096 \
            --num-heads $num_heads \
            --num-blocks 1 \
            --block-arch default \
            --attn-arch default \
            --embed-agg $embed_agg \
            --init-range $init_range \
            --cuda $cuda
    done
done
