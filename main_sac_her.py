import argparse
import datetime
import os
import time
import gym
import numpy as np
import itertools
import torch
import random
import json
from policy.sac_her import SAC
from utils.utils import get_env_params
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple


def main(args):
    args.cuda = True if torch.cuda.is_available() else False

    # Environment
    env = gym.make(args.env_name)
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    eval_env = gym.make(args.env_name)
    eval_env.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    env_params = get_env_params(env)

    # Experience
    Experience = namedtuple(
        "Experience",
        field_names="state action reward next_state mask"
    )

    # Agent
    # agent = SAC(env_params, args)
    # Agent
    agent = SAC(env_params["obs"] + env_params["goal"], env.action_space, args)

    # Tensorboard
    writer = SummaryWriter(
        "runs/{}_SAC_{}_{}_{}{}{}{}{}_vr{}_ur{}{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                          args.env_name,
                                                          args.policy,
                                                          args.seed,
                                                          "_autotune" if args.automatic_entropy_tuning else "",
                                                          "_mb" if args.model_based else "",
                                                          "_nmer" if args.nmer else "",
                                                          "_her" if args.her else "",
                                                          args.v_ratio,
                                                          args.updates_per_step,
                                                          "_deterministic" if args.deterministic_model else "",
                                                          )
        )

    # Save args/config to file
    config_path = os.path.join(writer.log_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(args.__dict__, f, indent=4)

    # Memory
    from utils.replay_memory import ReplayMemory
    memory = ReplayMemory(args.replay_size, args.seed)

    # Normalizer
    # from utils.utils import Normalizer
    # o_norm = Normalizer(size=env_params['obs'])
    # g_norm = Normalizer(size=env_params['goal'])

    # Exploration Loop
    total_numsteps = 0

    while total_numsteps < args.start_steps:
        episode_trajectory = []
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while not done and total_numsteps < args.start_steps:
            action = env.action_space.sample()  # Sample random action

            next_state, reward, done, info = env.step(action)  # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            episode_trajectory.append(Experience(state, action, reward, next_state, mask))
            state = next_state

        # Fill up replay memory
        steps_taken = len(episode_trajectory)

        # Normal experience replay
        for t in range(steps_taken):
            state_, action, reward, next_state_, mask = episode_trajectory[t]
            state = np.concatenate((state_["observation"], state_["desired_goal"]), axis=-1)
            next_state = np.concatenate((next_state_["observation"], next_state_["desired_goal"]), axis=-1)
            memory.push(state, action, reward, next_state, mask)  # Append transition to memory

            if args.her:
                for _ in range(args.her_replay_k):
                    future_idx = random.randint(t, steps_taken)  # index of future time step
                    goal_new = episode_trajectory[future_idx].next_state["achieved_goal"]  # take future next_state achieved goal and set as goal
                    reward_new = env.compute_reward(next_state_["achieved_goal"], goal_new, info=None)
                    state = np.concatenate((state_["observation"], goal_new), axis=-1)
                    next_state = np.concatenate((next_state_["observation"], goal_new), axis=-1)
                    memory.push(state, action, reward_new, next_state, 1.0)  # Append transition to memory

    # Training Loop
    total_numsteps = 0
    updates = 0

    for i_episode in itertools.count(1):
        episode_trajectory = []
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()
        time_start = time.time()

        while not done:
            state_ = np.concatenate((state["observation"], state["desired_goal"]), axis=-1)
            action = agent.select_action(state_)  # Sample action from policy

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
                                                                                                         args.batch_size,
                                                                                                         updates)
                    updates += 1
                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)

            next_state, reward, done, _ = env.step(action)  # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            episode_trajectory.append(Experience(state, action, reward, next_state, mask))
            state = next_state

            if total_numsteps % args.eval_timesteps == 0 and args.eval is True:
                avg_reward_eval = 0.
                episodes_eval = 10
                total_success_rate = []
                for _ in range(episodes_eval):
                    state_eval = eval_env.reset()
                    episode_reward_eval = 0
                    done_eval = False
                    per_success_rate = []
                    while not done_eval:
                        state_eval_ = np.concatenate((state_eval["observation"], state_eval["desired_goal"]), axis=-1)
                        action_eval = agent.select_action(state_eval_, evaluate=True)

                        next_state_eval, reward_eval, done_eval, info = eval_env.step(action_eval)
                        episode_reward_eval += reward_eval
                        per_success_rate.append(info["is_success"])

                        state_eval = next_state_eval
                    avg_reward_eval += episode_reward_eval
                    total_success_rate.append(per_success_rate)

                avg_reward_eval /= episodes_eval
                total_success_rate = np.array(total_success_rate)
                total_success_rate = np.mean(total_success_rate[:, -1])

                writer.add_scalar('avg_reward/test_timesteps', avg_reward_eval, total_numsteps)
                writer.add_scalar('avg_reward/test_success_rate', total_success_rate, total_numsteps)

                print("----------------------------------------")
                print("Timestep Eval - Test Episodes: {}, Avg. Reward: {}, Avg. Success: {}".format(episodes_eval, round(avg_reward_eval, 2), round(total_success_rate, 3)))
                print("----------------------------------------")

        # Fill up replay memory
        steps_taken = len(episode_trajectory)

        # Normal experience replay
        for t in range(steps_taken):
            state_, action, reward, next_state_, mask = episode_trajectory[t]
            state = np.concatenate((state_["observation"], state_["desired_goal"]), axis=-1)
            next_state = np.concatenate((next_state_["observation"], next_state_["desired_goal"]), axis=-1)
            memory.push(state, action, reward, next_state, mask)  # Append transition to memory

            if args.her:
                for _ in range(args.replay_k):
                    future_idx = random.randint(t, steps_taken)  # index of future time step
                    goal_new = episode_trajectory[future_idx].next_state["achieved_goal"]  # take future next_state achieved goal and set as goal
                    reward_new = env.compute_reward(next_state_["achieved_goal"], goal_new, info=None)
                    state = np.concatenate((state_["observation"], goal_new), axis=-1)
                    next_state = np.concatenate((next_state_["observation"], goal_new), axis=-1)
                    memory.push(state, action, reward_new, next_state, 1.0)  # Append transition to memory

        writer.add_scalar('reward/train_timesteps', episode_reward, total_numsteps)
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}, time/step: {}s".format(i_episode,
                                                                                                      total_numsteps,
                                                                                                      episode_steps,
                                                                                                      round(episode_reward, 2),
                                                                                                      round((time.time() - time_start) / episode_steps, 3)))

        if total_numsteps > args.num_steps:
            break

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--env-name', default="FetchReach-v1",
                        help='Mujoco Gym environment (default: FetchReach-v1)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic-entropy-tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num-steps', type=int, default=50000, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden-size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates-per-step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start-steps', type=int, default=500, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target-update-interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay-size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--target-entropy', type=float, default=-1, metavar='N',
                        help='Target entropy to use (default: -1)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    parser.add_argument('--model-based', action="store_true",
                        help='use Model-based (default: False)')
    parser.add_argument('--v-ratio', type=float, default=1.0, metavar='N',
                        help='virtual ratio (default: 1.0)')
    parser.add_argument('--eval-timesteps', type=int, default=1000, metavar='N',
                        help='when to eval the policy (default: 1000)')
    parser.add_argument('--update-env-model', type=int, default=250, metavar='N',
                        help='when to update the environment model (default: 500)')
    parser.add_argument('--n-training-samples', type=int, default=100000, metavar='N',
                        help='number of samples to train the environment model on (default: 100000)')
    parser.add_argument('--n-rollout-samples', type=int, default=100000, metavar='N',
                        help='number of samples to rollout the environment model on (default: 100000)')
    parser.add_argument('--model-retain-epochs', type=int, default=1, metavar='A',
                        help='retain epochs (default: 1)')
    parser.add_argument('--epoch-length', type=int, default=1000, metavar='N',
                        help='steps per epoch (default: 1000)')
    parser.add_argument('--rollout-min-epoch', type=int, default=20, metavar='N',
                        help='rollout min epoch (default: 20)')
    parser.add_argument('--rollout-max-epoch', type=int, default=150, metavar='N',
                        help='rollout max epoch (default: 150)')
    parser.add_argument('--rollout-min-length', type=int, default=1, metavar='N',
                        help='rollout min length (default: 1)')
    parser.add_argument('--rollout-max-length', type=int, default=15, metavar='N',
                        help='rollout max length (default: 15)')
    parser.add_argument('--deterministic-model', action="store_true",
                        help='use Model-based deterministic model (default: False)')
    parser.add_argument('--nmer', action="store_true",
                        help='use NMER (default: False)')
    parser.add_argument('--k-neighbours', type=int, default=10, metavar='N',
                        help='amount of neighbours to use (default: 10)')
    parser.add_argument('--per', action="store_true",
                        help='use PER (default: False)')
    parser.add_argument('--slapp', action="store_true",
                        help='use SLAPP (default: False)')
    parser.add_argument('--her-replay-strategy', type=str, default="future", metavar='N',
                        help='replay strategy to use (default: "future"')
    parser.add_argument('--her-replay-k', type=int, default=4, metavar='N',
                        help='replay k to use (default: 4)')
    parser.add_argument('--her', action="store_true",
                        help='use HER (default: False)')

    args = parser.parse_args()

    main(args)
