#!/bin/bash

env_name=FetchReach-v1
num_steps=30000
start_steps=500
n_update_batches=200
# rollout_min_epoch=0
# rollout_max_epoch=1
# rollout_min_length=3
# rollout_max_length=3

# FetchReach-v1 SAC UTD 1
for seed in 123 456 789
do
  python main_sac_her.py --env-name $env_name --num-steps $num_steps --start-steps $start_steps --seed $seed --eval-timesteps 500
done

# FetchReach-v1 SAC UTD 10
for seed in 123 456 789
do
  python main_sac_her.py --env-name $env_name --num-steps $num_steps --start-steps $start_steps --seed $seed --eval-timesteps 500 --n-update-batches $n_update_batches
done

# FetchReach-v1 HER NUB  1
for seed in 123 456 789
do
  python main_sac_her.py --env-name $env_name --num-steps $num_steps --start-steps $start_steps --her --seed $seed --eval-timesteps 500
done

# FetchReach-v1 HER NUB 10
for seed in 123 456 789
do
  python main_sac_her.py --env-name $env_name --num-steps $num_steps --start-steps $start_steps --her --seed $seed --eval-timesteps 500 --n-update-batches $n_update_batches
done

# FetchReach-v1 HER MBPO NUB 10
for seed in 123 456 789
do
  python main_sac_her.py --env-name $env_name --num-steps $num_steps --start-steps $start_steps --model-based --seed $seed --eval-timesteps 500 --n-update-batches $n_update_batches
done

# FetchReach-v1 HER NMER NUB 10
for seed in 123 456 789
do
  python main_sac_her.py --env-name $env_name --num-steps $num_steps --start-steps $start_steps --nmer --her --seed $seed --eval-timesteps 500 --n-update-batches $n_update_batches
done

# FetchReach-v1 HER LCERCC NUB 10
for seed in 123 456 789
do
  python main_sac_her.py --env-name $env_name --num-steps $num_steps --start-steps $start_steps --lcercc --seed $seed --eval-timesteps 500 --n-update-batches $n_update_batches
done

# FetchReach-v1 HER LCERRM NUB 10
for seed in 123 456 789
do
  python main_sac_her.py --env-name $env_name --num-steps $num_steps --start-steps $start_steps --lcerrm --seed $seed --eval-timesteps 500 --n-update-batches $n_update_batches
done