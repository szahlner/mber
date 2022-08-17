#!/bin/bash

# InvertedPendulum-v2 UTD 1
for seed in 123 456 789
do
  python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --start-steps 500 --updates-per-step 1 --seed $seed --eval-timesteps 250
done

# InvertedPendulum-v2 UTD 10
for seed in 123 456 789
do
  python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --start-steps 500 --updates-per-step 10 --seed $seed --eval-timesteps 250
done

# InvertedPendulum-v2 MBPO
for seed in 123 456 789
do
  python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --n-training-samples 100000 --start-steps 500 --model-based --updates-per-step 10 --v-ratio 0.95 --seed $seed --eval-timesteps 250
done
