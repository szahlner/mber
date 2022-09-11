#!/bin/bash

env_name=FetchPush-v1
num_steps=300000
start_steps=500
n_update_batches=1000
# rollout_min_epoch=0
# rollout_max_epoch=1
# rollout_min_length=3
# rollout_max_length=3

# FetchPush-v1 UTD 1
for seed in 123 456 789
do
  python main_sac_her.py --env-name $env_name --num-steps $num_steps --start-steps $start_steps --n-update-batches 20 --seed $seed
done

# FetchPush-v1 HER UTD 1
for seed in 123 456 789
do
  python main_sac_her.py --env-name $env_name --num-steps $num_steps --her --her-normalize --start-steps $start_steps --n-update-batches 20 --seed $seed
done

# FetchPush-v1 UTD 20
for seed in 123 456 789
do
  python main_sac_her.py --env-name $env_name --num-steps $num_steps --start-steps $start_steps --n-update-batches $n_update_batches --seed $seed
done

# FetchPush-v1 HER UTD 20
for seed in 123 456 789
do
  python main_sac_her.py --env-name $env_name --num-steps $num_steps --her --her-normalize --start-steps $start_steps --n-update-batches $n_update_batches --seed $seed
done