#!/bin/bash

env_name=Hopper-v2
target_entropy=-1
num_steps=125000
start_steps=5000
updates_per_step=20

# Hopper-v2 LCERRM
for seed in 123  # 123 456 789
do
  python main_sac.py --env-name $env_name --target-entropy $target_entropy --num-steps $num_steps --start-steps $start_steps --lcerrm --updates-per-step $updates_per_step --seed $seed
done
