#!/bin/bash

env_name=HalfCheetah-v2
num_steps=400000
start_steps=5000
updates_per_step=40
rollout_min_epoch=20
rollout_max_epoch=150
rollout_min_length=1
rollout_max_length=1

# HalfCheetah-v2 UTD 1
for seed in 123 456 789
do
  python main_ddpg.py --env-name $env_name --num-steps $num_steps --start-steps $start_steps --updates-per-step 1 --seed $seed
done

# HalfCheetah-v2 UTD 40
for seed in 123 456 789
do
  python main_ddpg.py --env-name $env_name --num-steps $num_steps --start-steps $start_steps --updates-per-step $updates_per_step --seed $seed
done

# HalfCheetah-v2 MBPO
for seed in 123 456 789
do
  python main_ddpg.py --env-name $env_name --num-steps $num_steps --start-steps $start_steps --model-based --updates-per-step $updates_per_step --v-ratio 0.95 --rollout-min-epoch $rollout_min_epoch --rollout-max-epoch $rollout_max_epoch --rollout-min-length $rollout_min_length --rollout-max-length $rollout_max_length --seed $seed
done
