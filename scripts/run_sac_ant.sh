#!/bin/bash

env_name=Ant-v2
target_entropy=-4
num_steps=300000
start_steps=5000
updates_per_step=20
rollout_min_epoch=20
rollout_max_epoch=100
rollout_min_length=1
rollout_max_length=25

# InvertedPendulum-v2 UTD 1
for seed in 123 456 789
do
  python main_sac.py --env-name $env_name --target-entropy $target_entropy --num-steps $num_steps --start-steps $start_steps --updates-per-step 1 --seed $seed
done

# InvertedPendulum-v2 UTD 10
for seed in 123 456 789
do
  python main_sac.py --env-name $env_name --target-entropy $target_entropy --num-steps $num_steps --start-steps $start_steps --updates-per-step $updates_per_step --seed $seed
done

# InvertedPendulum-v2 MBPO
for seed in 123 456 789
do
  python main_sac.py --env-name $env_name --target-entropy $target_entropy --num-steps $num_steps --start-steps $start_steps --model-based --updates-per-step $updates_per_step --v-ratio 0.95 --rollout-min-epoch $rollout_min_epoch --rollout-max-epoch $rollout_max_epoch --rollout-min-length $rollout_min_length --rollout-max-length $rollout_max_length --seed $seed
done
