#!/bin/bash

env_name=ShadowHandBlock-v1
alpha=0.01
num_steps=1500000
start_steps=1000
n_update_batches=400

# ShadowHandBlock-v1 LCERRM
for seed in 456  # 123 456 789
do
  python main_sac_her.py --env-name $env_name --num-steps $num_steps --start-steps $start_steps --lcerrm --alpha $alpha --seed $seed --eval-timesteps 1000 --n-update-batches $n_update_batches --her-normalize
done