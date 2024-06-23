model_name=$1
is_trained=$2
nunits=$3
cuda=$4

init_range=0.02
embed_agg=last-token

# datasets = Pereira2018.384 Pereira2018.243 Fedorenko2016 Blank2014 Tuckute2024 Wehbe2014
# baselines = default shuffle random-words random

if [ $model_name = "suma" ]; then
    for seed in 10 20 30 42 50
    do
        for benchmark_name in Pereira2018.384 Pereira2018.243 Fedorenko2016 Blank2014 Tuckute2024 Wehbe2014
        do
            for baseline in default
            do
                python score_model.py --model-name suma-1-ln-attn \
                    --benchmark-name ${benchmark_name}-all \
                    --language-mask-nunits $nunits \
                    --embed-agg $embed_agg \
                    --baseline $baseline \
                    --init-range $init_range \
                    --custom-model \
                    --num-cycles 2 \
                    --hidden-dim 4096 \
                    --num-heads 512 \
                    --num-blocks 1 \
                    --block-arch "ln1-res" \
                    --attn-arch default \
                    --seed $seed \
                    --cuda $cuda
            done
        done
    done

elif [ $is_trained = "untrained" ]; then
    for benchmark_name in Pereira2018.384 Pereira2018.243 Fedorenko2016 Blank2014 Tuckute2024 Wehbe2014
    do
        for baseline in default
        do
            for seed in 10 20 30 42 50
            do
                python score_model.py --model-name $model_name \
                    --benchmark-name ${benchmark_name}-all \
                    --language-mask-nunits $nunits \
                    --init-range $init_range \
                    --embed-agg $embed_agg \
                    --baseline $baseline \
                    --seed $seed \
                    --cuda $cuda
            done
        done
    done

else
   
    for benchmark_name in Pereira2018.384 Pereira2018.243 Fedorenko2016 Blank2014 Tuckute2024 Wehbe2014
    do
        for baseline in default
        do
            python score_model.py --model-name $model_name \
            --benchmark-name ${benchmark_name}-all \
            --language-mask-nunits $nunits \
            --embed-agg $embed_agg \
            --baseline $baseline \
            --pretrained \
            --cuda $cuda \
            --seed 42
        done
    done
fi