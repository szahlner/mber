#!/bin/bash

env_name=Hopper-v2
target_entropy=-1
num_steps=125000
start_steps=5000
updates_per_step=20
rollout_min_epoch=0
rollout_max_epoch=1
rollout_min_length=5
rollout_max_length=5

# Hopper-v2 SLAP
for seed in 123 456 789
do
  python main_sac.py --env-name $env_name --target-entropy $target_entropy --num-steps $num_steps --start-steps $start_steps --slapp --updates-per-step $updates_per_step --v-ratio 0.95 --rollout-min-epoch $rollout_min_epoch --rollout-max-epoch $rollout_max_epoch --rollout-min-length $rollout_min_length --rollout-max-length $rollout_max_length --seed $seed
done
