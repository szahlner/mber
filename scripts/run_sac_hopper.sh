#!/bin/bash

env_name=Hopper-v2
target_entropy=-1
num_steps=125000
start_steps=5000
updates_per_step=20

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
  python main_sac.py --env-name $env_name --target-entropy $target_entropy --num-steps $num_steps --n-training-samples 100000 --start-steps $start_steps --model-based --updates-per-step $updates_per_step --v-ratio 0.95 --seed $seed
done
