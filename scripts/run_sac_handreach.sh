#!/bin/bash

env_name=ShadowHandReach-v1
alpha=0.01
num_steps=50000
start_steps=500
n_update_batches=1000
rollout_min_epoch=0
rollout_max_epoch=1
rollout_min_length=3
rollout_max_length=3

# FetchReach-v1 UTD 1
for seed in 123 456 789
do
  python main_sac_her.py --env-name $env_name --num-steps $num_steps --start-steps $start_steps --alpha $alpha --n-update-batches 20 --seed $seed --eval-timesteps 500
done

# FetchReach-v1 UTD 20
for seed in 123 456 789
do
  python main_sac_her.py --env-name $env_name --num-steps $num_steps --start-steps $start_steps --alpha $alpha --n-update-batches $n_update_batches --seed $seed --eval-timesteps 500
done

# FetchReach-v1 HER UTD 1
for seed in 123 456 789
do
  python main_sac_her.py --env-name $env_name --num-steps $num_steps --her --her-normalize --alpha $alpha --start-steps $start_steps --n-update-batches 20 --seed $seed --eval-timesteps 500
done

# FetchReach-v1 HER UTD 20
for seed in 123 456 789
do
  python main_sac_her.py --env-name $env_name --num-steps $num_steps --her --her-normalize --alpha $alpha --start-steps $start_steps --n-update-batches $n_update_batches --seed $seed --eval-timesteps 500
done

# FetchReach-v1 HER MBPO UTD 20
for seed in 123 456 789
do
  python main_sac_her.py --env-name $env_name --num-steps $num_steps --her --her-normalize --alpha $alpha --model-based --start-steps $start_steps --n-update-batches $n_update_batches --v-ratio 0.95 --rollout-min-epoch $rollout_min_epoch --rollout-max-epoch $rollout_max_epoch --rollout-min-length $rollout_min_length --rollout-max-length $rollout_max_length --seed $seed --eval-timesteps 500
done
