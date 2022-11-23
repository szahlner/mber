#!/bin/bash

env_name=ShadowHandBlock-v1
alpha=0.01
num_steps=1500000
start_steps=1000
# n_update_batches=200

# ShadowHandBlock-v1 HER NUB 1
for seed in 123 456 789
do
  python main_sac_her.py --env-name $env_name --num-steps $num_steps --start-steps $start_steps --her --alpha $alpha --seed $seed --eval-timesteps 1000 --her-normalize
done