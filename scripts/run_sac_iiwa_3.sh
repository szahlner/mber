#!/bin/bash

env_name=IIWA-extended-Lift-Position
target_entropy=-3
num_steps=500000
start_steps=5000
# updates_per_step=20

# IIWA14_extended UTD 1
for seed in 123 #  123 456 789
do
  python main_sac.py --env-name $env_name --target-entropy $target_entropy --num-steps $num_steps --start-steps $start_steps --updates-per-step 1 --seed $seed
done
