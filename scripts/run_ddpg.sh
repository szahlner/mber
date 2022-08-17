#!/bin/bash

# InvertedPendulum-v2 UTD 1
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --start-steps 500 --updates-per-step 1 --seed 1 --eval-timesteps 250
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --start-steps 500 --updates-per-step 1 --seed 12 --eval-timesteps 250
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --start-steps 500 --updates-per-step 1 --seed 123 --eval-timesteps 250
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --start-steps 500 --updates-per-step 1 --seed 1234 --eval-timesteps 250
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --start-steps 500 --updates-per-step 1 --seed 12345 --eval-timesteps 250


# InvertedPendulum-v2 UTD 10
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --start-steps 500 --updates-per-step 10 --seed 1 --eval-timesteps 250
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --start-steps 500 --updates-per-step 10 --seed 12 --eval-timesteps 250
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --start-steps 500 --updates-per-step 10 --seed 123 --eval-timesteps 250
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --start-steps 500 --updates-per-step 10 --seed 1234 --eval-timesteps 250
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --start-steps 500 --updates-per-step 10 --seed 12345 --eval-timesteps 250

# InvertedPendulum-v2 MBPO
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --n-training-samples 100000 --start-steps 500 --model-based --updates-per-step 10 --v-ratio 0.95 --seed 1 --eval-timesteps 250
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --n-training-samples 100000 --start-steps 500 --model-based --updates-per-step 10 --v-ratio 0.95 --seed 12 --eval-timesteps 250
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --n-training-samples 100000 --start-steps 500 --model-based --updates-per-step 10 --v-ratio 0.95 --seed 123 --eval-timesteps 250
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --n-training-samples 100000 --start-steps 500 --model-based --updates-per-step 10 --v-ratio 0.95 --seed 1234 --eval-timesteps 250
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --n-training-samples 100000 --start-steps 500 --model-based --updates-per-step 10 --v-ratio 0.95 --seed 12345 --eval-timesteps 250

# InvertedPendulum-v2 NMER MBPO
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --n-training-samples 100000 --start-steps 500 --nmer --model-based --updates-per-step 10 --v-ratio 0.95 --k-neighbours 3 --seed 1 --eval-timesteps 250
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --n-training-samples 100000 --start-steps 500 --nmer --model-based --updates-per-step 10 --v-ratio 0.95 --k-neighbours 3 --seed 12 --eval-timesteps 250
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --n-training-samples 100000 --start-steps 500 --nmer --model-based --updates-per-step 10 --v-ratio 0.95 --k-neighbours 3 --seed 123 --eval-timesteps 250
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --n-training-samples 100000 --start-steps 500 --nmer --model-based --updates-per-step 10 --v-ratio 0.95 --k-neighbours 3 --seed 1234 --eval-timesteps 250
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --n-training-samples 100000 --start-steps 500 --nmer --model-based --updates-per-step 10 --v-ratio 0.95 --k-neighbours 3 --seed 12345 --eval-timesteps 250

# InvertedPendulum-v2 NMER MBPO
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --n-training-samples 100000 --start-steps 500 --nmer --model-based --updates-per-step 10 --v-ratio 0.95 --k-neighbours 5 --seed 1 --eval-timesteps 250
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --n-training-samples 100000 --start-steps 500 --nmer --model-based --updates-per-step 10 --v-ratio 0.95 --k-neighbours 5 --seed 12 --eval-timesteps 250
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --n-training-samples 100000 --start-steps 500 --nmer --model-based --updates-per-step 10 --v-ratio 0.95 --k-neighbours 5 --seed 123 --eval-timesteps 250
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --n-training-samples 100000 --start-steps 500 --nmer --model-based --updates-per-step 10 --v-ratio 0.95 --k-neighbours 5 --seed 1234 --eval-timesteps 250
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --n-training-samples 100000 --start-steps 500 --nmer --model-based --updates-per-step 10 --v-ratio 0.95 --k-neighbours 5 --seed 12345 --eval-timesteps 250

# InvertedPendulum-v2 NMER MBPO
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --n-training-samples 100000 --start-steps 500 --nmer --model-based --updates-per-step 10 --v-ratio 0.95 --k-neighbours 7 --seed 1 --eval-timesteps 250
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --n-training-samples 100000 --start-steps 500 --nmer --model-based --updates-per-step 10 --v-ratio 0.95 --k-neighbours 7 --seed 12 --eval-timesteps 250
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --n-training-samples 100000 --start-steps 500 --nmer --model-based --updates-per-step 10 --v-ratio 0.95 --k-neighbours 7 --seed 123 --eval-timesteps 250
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --n-training-samples 100000 --start-steps 500 --nmer --model-based --updates-per-step 10 --v-ratio 0.95 --k-neighbours 7 --seed 1234 --eval-timesteps 250
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --n-training-samples 100000 --start-steps 500 --nmer --model-based --updates-per-step 10 --v-ratio 0.95 --k-neighbours 7 --seed 12345 --eval-timesteps 250

# InvertedPendulum-v2 NMER MBPO
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --n-training-samples 100000 --start-steps 500 --nmer --model-based --updates-per-step 10 --v-ratio 0.95 --k-neighbours 9 --seed 1 --eval-timesteps 250
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --n-training-samples 100000 --start-steps 500 --nmer --model-based --updates-per-step 10 --v-ratio 0.95 --k-neighbours 9 --seed 12 --eval-timesteps 250
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --n-training-samples 100000 --start-steps 500 --nmer --model-based --updates-per-step 10 --v-ratio 0.95 --k-neighbours 9 --seed 123 --eval-timesteps 250
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --n-training-samples 100000 --start-steps 500 --nmer --model-based --updates-per-step 10 --v-ratio 0.95 --k-neighbours 9 --seed 1234 --eval-timesteps 250
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --n-training-samples 100000 --start-steps 500 --nmer --model-based --updates-per-step 10 --v-ratio 0.95 --k-neighbours 9 --seed 12345 --eval-timesteps 250

# InvertedPendulum-v2 NMER MBPO
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --n-training-samples 100000 --start-steps 500 --nmer --model-based --updates-per-step 10 --v-ratio 0.95 --k-neighbours 11 --seed 1 --eval-timesteps 250
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --n-training-samples 100000 --start-steps 500 --nmer --model-based --updates-per-step 10 --v-ratio 0.95 --k-neighbours 11 --seed 12 --eval-timesteps 250
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --n-training-samples 100000 --start-steps 500 --nmer --model-based --updates-per-step 10 --v-ratio 0.95 --k-neighbours 11 --seed 123 --eval-timesteps 250
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --n-training-samples 100000 --start-steps 500 --nmer --model-based --updates-per-step 10 --v-ratio 0.95 --k-neighbours 11 --seed 1234 --eval-timesteps 250
python main_ddpg.py --env-name InvertedPendulum-v2 --num-steps 16000 --n-training-samples 100000 --start-steps 500 --nmer --model-based --updates-per-step 10 --v-ratio 0.95 --k-neighbours 11 --seed 12345 --eval-timesteps 250