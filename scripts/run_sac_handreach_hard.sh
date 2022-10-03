#!/bin/bash

env_name=ShadowHandReachHard-v1
alpha=0.01
num_steps=50000
start_steps=500
n_update_batches=1000
# rollout_min_epoch=0
# rollout_max_epoch=1
# rollout_min_length=3
# rollout_max_length=3

# ShadowHandReach-v1 UTD 1
for seed in 123 456 789
do
  python main_sac_her.py --env-name $env_name --num-steps $num_steps --start-steps $start_steps --alpha $alpha --seed $seed --eval-timesteps 500 --her-normalize
done

# ShadowHandReach-v1 UTD 10
for seed in 123 456 789
do
  python main_sac_her.py --env-name $env_name --num-steps $num_steps --start-steps $start_steps --alpha $alpha --seed $seed --eval-timesteps 500 --n-update-batches $n_update_batches --her-normalize
done

# ShadowHandReach-v1 HER UTD 1
for seed in 123 456 789
do
  python main_sac_her.py --env-name $env_name --num-steps $num_steps --start-steps $start_steps --her --alpha $alpha --seed $seed --eval-timesteps 500 --her-normalize
done

# ShadowHandReach-v1 HER UTD 10
for seed in 123 456 789
do
  python main_sac_her.py --env-name $env_name --num-steps $num_steps --start-steps $start_steps --her --alpha $alpha --seed $seed --eval-timesteps 500 --n-update-batches $n_update_batches --her-normalize
done

# ShadowHandReach-v1 HER MBPO UTD 10
for seed in 123 456 789
do
  python main_sac_her.py --env-name $env_name --num-steps $num_steps --start-steps $start_steps --model-based --alpha $alpha --seed $seed --eval-timesteps 500 --n-update-batches $n_update_batches --her-normalize
done

# ShadowHandReach-v1 HER NMER
for seed in 123 456 789
do
  python main_sac_her.py --env-name $env_name --num-steps $num_steps --start-steps $start_steps --nmer --alpha $alpha --seed $seed --eval-timesteps 500 --n-update-batches $n_update_batches --her-normalize
done

# ShadowHandReach-v1 HER SLAPP
for seed in 123 456 789
do
  python main_sac_her.py --env-name $env_name --num-steps $num_steps --start-steps $start_steps --slapp --alpha $alpha --seed $seed --eval-timesteps 500 --n-update-batches $n_update_batches --her-normalize
done
