#!/bin/bash

env_name=InvertedPendulum-v2
target_entropy=-0.05
num_steps=16000
start_steps=1000
updates_per_step=10

# InvertedPendulum-v2 LCERCC
for seed in 123 456 789
do
  python main_sac.py --env-name $env_name --target-entropy $target_entropy --num-steps $num_steps --start-steps $start_steps --lcercc --updates-per-step $updates_per_step --seed $seed --eval-timesteps 250
done
