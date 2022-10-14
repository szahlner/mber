#!/bin/bash

env_name=Hopper-v2
target_entropy=-1
num_steps=125000
start_steps=5000
updates_per_step=20

# Hopper-v2 LCERCC
for seed in 456
do
  python main_sac.py --env-name $env_name --target-entropy $target_entropy --num-steps $num_steps --start-steps $start_steps --lcercc --updates-per-step $updates_per_step --seed $seed
done
