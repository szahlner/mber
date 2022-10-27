#!/bin/bash

env_name=HalfCheetah-v2
target_entropy=-3
num_steps=400000
start_steps=5000
updates_per_step=40

# HalfCheetah-v2 LCERCC
for seed in 456  # 123 456 789
do
  python main_sac.py --env-name $env_name --target-entropy $target_entropy --num-steps $num_steps --start-steps $start_steps --lcercc --updates-per-step $updates_per_step --seed $seed
done
