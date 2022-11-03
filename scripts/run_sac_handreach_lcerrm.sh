#!/bin/bash

env_name=ShadowHandReach-v1
alpha=0.01
num_steps=50000
start_steps=500
n_update_batches=200

# ShadowHandReach-v1 HER LCERRM
for seed in 123 456 789
do
  python main_sac_her.py --env-name $env_name --num-steps $num_steps --start-steps $start_steps --lcerrm --alpha $alpha --seed $seed --eval-timesteps 500 --n-update-batches $n_update_batches --her-normalize
done

