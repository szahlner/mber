#!/bin/bash

env_name=InvertedPendulum-v2
num_steps=16000
start_steps=500
updates_per_step=10
rollout_min_epoch=1
rollout_max_epoch=15
rollout_min_length=1
rollout_max_length=1

# InvertedPendulum-v2 NMER
for seed in 123 456 789
do
  python main_ddpg.py --env-name $env_name --num-steps $num_steps --start-steps $start_steps --nmer --updates-per-step $updates_per_step --seed $seed --eval-timesteps 250
done

# InvertedPendulum-v2 NMER MBPO
for seed in 123 456 789
do
  python main_ddpg.py --env-name $env_name --num-steps $num_steps --start-steps $start_steps --nmer --model-based --updates-per-step $updates_per_step --v-ratio 0.95 --rollout-min-epoch $rollout_min_epoch --rollout-max-epoch $rollout_max_epoch --rollout-min-length $rollout_min_length --rollout-max-length $rollout_max_length --seed $seed --eval-timesteps 250
done
