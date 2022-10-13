#!/bin/bash

env_name=Walker2d-v2
target_entropy=-3
num_steps=300000
start_steps=5000
updates_per_step=20
rollout_min_epoch=20
rollout_max_epoch=150
rollout_min_length=1
rollout_max_length=1

# Walker2d-v2 NMER
for seed in 123 456 789
do
  python main_sac.py --env-name $env_name --target-entropy $target_entropy --num-steps $num_steps --start-steps $start_steps --nmer --updates-per-step $updates_per_step --seed $seed
done

# Walker2d-v2 NMER MBPO
# for seed in 123 456 789
# do
#   python main_sac.py --env-name $env_name --target-entropy $target_entropy --num-steps $num_steps --start-steps $start_steps --nmer --model-based --updates-per-step $updates_per_step --v-ratio 0.95 --rollout-min-epoch $rollout_min_epoch --rollout-max-epoch $rollout_max_epoch --rollout-min-length $rollout_min_length --rollout-max-length $rollout_max_length --seed $seed
# done
