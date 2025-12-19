#!/bin/bash
# Pipeline for the "simple" ffn and lstm nn models

for seed in 2235 6312 6068 9742 8880 2197 669 6256 3309 2541 8643 7785 195 6914 29; do
        python ../../run_simple_models.py --model_label "ffn" --seed ${seed}  --device "cpu" 
done

# for seed in 2235; do
#         python ../../run_simple_models.py --model_label "lstm" --seed ${seed}  --device "cpu" 
# done
