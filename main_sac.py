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
from policy.sac import SAC
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple
from scipy.interpolate import RBFInterpolator


def main(args):
    args.cuda = True if torch.cuda.is_available() else False

    # Environment
    if args.env_name == "IIWA14_extended":
        import robosuite as suite
        from robosuite.wrappers import GymWrapper
        from robosuite.controllers import load_controller_config

        env_id = "Lift"
        env = GymWrapper(
            suite.make(
                env_id=env_id,
                robots="IIWA14_extended",  # use Sawyer robot
                use_camera_obs=False,  # do not use pixel observations
                has_offscreen_renderer=False,  # not needed since not using pixel obs
                has_renderer=True,  # make sure we can render to the screen
                reward_shaping=True,  # use dense rewards
                control_freq=20,  # control should happen fast enough so that simulation looks smooth
                horizon=1000,
                controller_configs=load_controller_config(default_controller="OSC_POSE"),
            )
        )
    else:
        env = gym.make(args.env_name)

    if args.env_name == "Ant-v2":
        from environment.ant_truncated import AntTruncatedV2
        env = AntTruncatedV2(env)

    env.seed(args.seed)
    env.action_space.seed(args.seed)
    eval_env = gym.make(args.env_name)

    if args.env_name == "Ant-v2":
        from environment.ant_truncated import AntTruncatedV2
        eval_env = AntTruncatedV2(eval_env)

    eval_env.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Model-based
    if args.model_based:
        from utils.utils import get_predicted_states

        state_size = np.prod(env.observation_space.shape)
        action_size = np.prod(env.action_space.shape)

        if args.deterministic_model:
            from mdp.mdp_model_deterministic import EnsembleDynamicsModel

            env_model = EnsembleDynamicsModel(
                network_size=7,
                elite_size=5,
                state_size=state_size,
                action_size=action_size,
                reward_size=1,
                hidden_size=200,
                dropout_rate=0.05,
                use_decay=True
            )
        else:
            from mdp.mdp_model import EnsembleDynamicsModel

            env_model = EnsembleDynamicsModel(
                network_size=7,
                elite_size=5,
                state_size=state_size,
                action_size=action_size,
                reward_size=1,
                hidden_size=200,
                use_decay=True
            )

    # Experience
    Experience = namedtuple(
        "Experience",
        field_names="state action reward next_state mask"
    )
    # Agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)

    # Tensorboard
    writer = SummaryWriter(
        "runs/{}_SAC_{}_{}_{}{}{}{}{}{}_vr{}_ur{}{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                            args.env_name, args.policy, args.seed,
                                                            "_autotune" if args.automatic_entropy_tuning else "",
                                                            "_mb" if args.model_based else "",
                                                            "_nmer" if args.nmer else "",
                                                            "_per" if args.per else "",
                                                            "_slapp" if args.slapp else "",
                                                            args.v_ratio, args.updates_per_step,
                                                            "_deterministic" if args.deterministic_model else "",
                                                            )
        )

    # Save args/config to file
    config_path = os.path.join(writer.log_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(args.__dict__, f, indent=4)

    # Memory
    if args.model_based:
        if args.nmer:
            from utils.replay_memory import MbpoNmerReplayMemory
            memory = MbpoNmerReplayMemory(
                args.replay_size, args.seed,
                v_ratio=args.v_ratio, env_name=args.env_name, args=args,
                k_neighbours=args.k_neighbours
            )
        else:
            from utils.replay_memory import MbpoReplayMemory
            memory = MbpoReplayMemory(args.replay_size, args.seed, v_ratio=args.v_ratio, env_name=args.env_name, args=args)
    else:
        if args.nmer and args.per:
            from utils.replay_memory import PerNmerReplayMemory
            state_size = np.prod(env.observation_space.shape)
            action_size = np.prod(env.action_space.shape)
            memory = PerNmerReplayMemory(args.replay_size, args.seed, state_dim=state_size, action_dim=action_size,
                                         env_name=args.env_name, k_neighbours=args.k_neighbours)
        elif args.nmer:
            from utils.replay_memory import NmerReplayMemory
            memory = NmerReplayMemory(args.replay_size, args.seed, env_name=args.env_name, k_neighbours=args.k_neighbours)
        elif args.per:
            from utils.replay_memory import PerReplayMemory
            state_size = np.prod(env.observation_space.shape)
            action_size = np.prod(env.action_space.shape)
            memory = PerReplayMemory(args.replay_size, args.seed, state_dim=state_size, action_dim=action_size)
        elif args.slapp:
            from utils.utils import termination_fn
            from utils.replay_memory import SimpleLocalApproximationReplayMemory
            from sklearn.neighbors import NearestNeighbors
            state_size = np.prod(env.observation_space.shape)
            action_size = np.prod(env.action_space.shape)
            memory = SimpleLocalApproximationReplayMemory(args.replay_size, args.seed,
                                                          state_dim=state_size, action_dim=action_size,
                                                          v_ratio=args.v_ratio, env_name=args.env_name, args=args)

        else:
            from utils.replay_memory import ReplayMemory
            memory = ReplayMemory(args.replay_size, args.seed)

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
            state, action, reward, next_state, mask = episode_trajectory[t]
            memory.push(state, action, reward, next_state, mask)  # Append transition to memory

    if args.slapp:
        o = memory.buffer["state"][:len(memory)]
        a = memory.buffer["action"][:len(memory)]
        r = memory.buffer["reward"][:len(memory)]
        o_2 = memory.buffer["next_state"][:len(memory)]
        memory.update_clusters(o, a, r, o_2)

    if args.nmer:
        memory.update_neighbours()

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
            action = agent.select_action(state)  # Sample action from policy

            if args.model_based and total_numsteps % args.update_env_model == 0:
                # Get real samples from environment
                batch_size = len(memory)
                o, a, r, o_2, _ = memory.sample_r(batch_size=batch_size)

                # Difference
                d_o = o_2 - o
                inputs = np.concatenate((o, a), axis=-1)
                labels = np.concatenate((np.expand_dims(r, axis=-1), d_o), axis=-1)

                # Train the environment model
                env_model.train(inputs, labels, batch_size=256, holdout_ratio=0.2)

                # Resize buffer capacity
                current_epoch = int(total_numsteps / args.epoch_length)
                memory.set_rollout_length(current_epoch)
                memory.resize_v_memory()

                # Rollout the environment model
                o, _, _, _, _ = memory.sample_r(batch_size=args.n_rollout_samples)

                for n in range(memory.rollout_length):
                    a = agent.select_action(o)
                    r, o_2, d = get_predicted_states(env_model, o, a, args.env_name)
                    # Push into memory
                    for k in range(len(o)):
                        memory.push_v(o[k], a[k], float(r[k]), o_2[k], float(not d[k]))
                    nonterm_mask = ~d.squeeze(-1)
                    if nonterm_mask.sum() == 0:
                        break
                    o = o_2[nonterm_mask]

                if args.nmer:
                    memory.update_v_neighbours()

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    if args.per:
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters_per(memory,
                                                                                                                 args.batch_size,
                                                                                                                 updates)
                    else:
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
                for _ in range(episodes_eval):
                    state_eval = eval_env.reset()
                    episode_reward_eval = 0
                    done_eval = False
                    while not done_eval:
                        action_eval = agent.select_action(state_eval, evaluate=True)

                        next_state_eval, reward_eval, done_eval, _ = eval_env.step(action_eval)
                        episode_reward_eval += reward_eval

                        state_eval = next_state_eval
                    avg_reward_eval += episode_reward_eval
                avg_reward_eval /= episodes_eval

                writer.add_scalar('avg_reward/test_timesteps', avg_reward_eval, total_numsteps)

                print("----------------------------------------")
                print("Timestep Eval - Test Episodes: {}, Avg. Reward: {}".format(episodes_eval, round(avg_reward_eval, 2)))
                print("----------------------------------------")

        # Fill up replay memory
        steps_taken = len(episode_trajectory)
        # Normal experience replay
        for t in range(steps_taken):
            state, action, reward, next_state, mask = episode_trajectory[t]
            memory.push(state, action, reward, next_state, mask)  # Append transition to memory

        if args.slapp:
            o = memory.buffer["state"][total_numsteps - steps_taken:total_numsteps]
            a = memory.buffer["action"][total_numsteps - steps_taken:total_numsteps]
            r = memory.buffer["reward"][total_numsteps - steps_taken:total_numsteps]
            o_2 = memory.buffer["next_state"][total_numsteps - steps_taken:total_numsteps]
            memory.update_clusters(o, a, r, o_2)

        if args.nmer:
            memory.update_neighbours()

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
    parser.add_argument('--env-name', default="Hopper-v2",
                        help='Mujoco Gym environment (default: Hopper-v2)')
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
    parser.add_argument('--automatic-entropy-tuning', type=bool, default=True, metavar='G',
                        help='Automaically adjust α (default: True)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num-steps', type=int, default=125001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden-size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates-per-step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start-steps', type=int, default=5000, metavar='N',
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
    parser.add_argument('--v-ratio', type=float, default=0.95, metavar='N',
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

    args = parser.parse_args()

    main(args)
