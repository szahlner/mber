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
    if "ShadowHandReach" in args.env_name:
        import shadowhand_gym

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

    # Model-based
    if args.model_based:
        from utils.utils import get_predicted_states_her

        state_size = env_params["obs"] + 2 * env_params["goal"]
        action_size = env_params["action"]

        if args.deterministic_model:
            from mdp.mdp_model_deterministic import EnsembleDynamicsModel

            env_model = EnsembleDynamicsModel(
                network_size=7,
                elite_size=5,
                state_size=state_size,
                action_size=action_size,
                reward_size=0,
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
                reward_size=0,
                hidden_size=200,
                use_decay=True
            )

    # Experience
    Experience = namedtuple(
        "Experience",
        field_names="state action reward next_state mask"
    )

    # Agent
    agent = SAC(env_params["obs"] + env_params["goal"], env.action_space, args)

    # Tensorboard
    writer = SummaryWriter(
        "runs/{}_SAC_{}_{}_{}{}{}{}{}{}{}_vr{}_ur{}_nub{}{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                                    args.env_name,
                                                                    args.policy,
                                                                    args.seed,
                                                                    "_autotune" if args.automatic_entropy_tuning else "",
                                                                    "_mb" if args.model_based else "",
                                                                    "_nmer" if args.nmer else "",
                                                                    "_her" if args.her else "",
                                                                    "_lcercc" if args.lcercc else "",
                                                                    "_lcerrm" if args.lcerrm else "",
                                                                    args.v_ratio,
                                                                    args.updates_per_step,
                                                                    args.n_update_batches,
                                                                    "_deterministic" if args.deterministic_model else "",
                                                                    )
        )

    # Save args/config to file
    config_path = os.path.join(writer.log_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(args.__dict__, f, indent=4)

    # Memory
    if args.model_based:
        from utils.her.replay_memory import HerSampler, HerMbpoReplayMemory
        sampler = HerSampler("future", args.her_replay_k, env.compute_reward)
        memory = HerMbpoReplayMemory(env_params, args.replay_size, v_ratio=args.v_ratio, args=args,
                                     sample_func=sampler.sample_her_transitions, normalize=args.her_normalize)
    else:
        if args.her:
            from utils.her.replay_memory import HerSampler, HerReplayMemory
            sampler = HerSampler("future", args.her_replay_k, env.compute_reward)
            memory = HerReplayMemory(env_params, args.replay_size,
                                     sample_func=sampler.sample_her_transitions, normalize=args.her_normalize)
        elif args.nmer:
            from utils.her.replay_memory import HerNmerReplayMemory
            memory = HerNmerReplayMemory(env_params, args.replay_size, args=args, normalize=args.her_normalize)
        elif args.lcercc:
            from utils.her.replay_memory import HerLocalClusterExperienceReplayClusterCenterReplayMemory
            memory = HerLocalClusterExperienceReplayClusterCenterReplayMemory(env_params, args.replay_size, args=args,
                                                                              normalize=args.her_normalize)
        elif args.lcerrm:
            from utils.her.replay_memory import HerLocalClusterExperienceReplayRandomMemberReplayMemory
            memory = HerLocalClusterExperienceReplayRandomMemberReplayMemory(env_params, args.replay_size, args=args,
                                                                             normalize=args.her_normalize)
        else:
            from utils.her.replay_memory import SimpleReplayMemory
            memory = SimpleReplayMemory(env_params, args.replay_size, args=args, normalize=args.her_normalize)

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
        o, ag, g, a = [], [], [], []

        # Normal experience replay
        for t in range(steps_taken):
            state, action, reward, next_state, mask = episode_trajectory[t]
            # Append transition to memory
            o.append(state["observation"]), ag.append(state["achieved_goal"])
            g.append(state["desired_goal"]), a.append(action)

        o.append(next_state["observation"]), ag.append(next_state["achieved_goal"])
        o, ag, g, a = np.array([o]), np.array([ag]), np.array([g]), np.array([a])
        memory.push_episode([o, ag, g, a])

    if args.lcercc or args.lcerrm:
        o = memory.buffers["obs"][:len(memory)]
        ag = memory.buffers["ag"][:len(memory)]
        a = memory.buffers["actions"][:len(memory)]
        o_2 = memory.buffers["obs_next"][:len(memory)]
        ag_2 = memory.buffers["ag_next"][:len(memory)]
        g = memory.buffers["g"][:len(memory)]
        memory.update_clusters(o, ag, a, o_2, ag_2, g)

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
            if args.her_normalize:
                state_norm = memory.o_norm.normalize(state["observation"])
                goal_norm = memory.g_norm.normalize(state["desired_goal"])
                state_ = np.concatenate((state_norm, goal_norm), axis=-1)
            else:
                state_ = np.concatenate((state["observation"], state["desired_goal"]), axis=-1)
            action = agent.select_action(state_)  # Sample action from policy

            if args.model_based and total_numsteps % args.update_env_model == 0:
                # Get real samples from environment
                batch_size = max(len(memory), 10000)
                transitions = memory.sample_r(batch_size=batch_size, return_transitions=True)

                inputs = np.concatenate((transitions["obs"], transitions["ag"], transitions["g"], transitions["actions"]), axis=-1)
                # Difference
                labels = np.concatenate((transitions["obs_next"], transitions["ag_next"], transitions["g"]), axis=-1) - np.concatenate((transitions["obs"], transitions["ag"], transitions["g"]), axis=-1)

                # Train the environment model
                env_model.train(inputs, labels, batch_size=256, holdout_ratio=0.2)

                # Resize buffer capacity
                current_epoch = int(total_numsteps / args.epoch_length)
                current_rollout_length = memory.rollout_length
                memory.set_rollout_length(current_epoch)
                if current_rollout_length != memory.rollout_length:
                    memory.resize_v_memory()

                # Rollout the environment model
                # o, o_ag, o_g, _, _, _, _, _, _ = memory.sample_r(batch_size=args.n_rollout_samples)
                transitions = memory.sample_r(batch_size=args.n_rollout_samples, return_transitions=True)
                o, o_ag, o_g = transitions["obs"], transitions["ag"], transitions["g"]

                # Preallocate
                v_state = np.empty(shape=(args.n_rollout_samples, memory.rollout_length + 1, env_params["obs"]))
                v_state_ag = np.empty(shape=(args.n_rollout_samples, memory.rollout_length + 1, env_params["goal"]))
                v_state_g = np.empty(shape=(args.n_rollout_samples, memory.rollout_length, env_params["goal"]))
                v_action = np.empty(shape=(args.n_rollout_samples, memory.rollout_length, env_params["action"]))

                # Rollout
                for n in range(memory.rollout_length):
                    if args.her_normalize:
                        o_norm = memory.o_norm.normalize(o)
                        g_norm = memory.g_norm.normalize(o_g)
                        o_ = np.concatenate((o_norm, g_norm), axis=-1)
                    else:
                        o_ = np.concatenate((o, o_g), axis=-1)
                    a = agent.select_action(o_)
                    o_2, o_2_ag = get_predicted_states_her(env_model, o, o_ag, o_g, a, env_params)
                    # Push into memory
                    v_state[:, n], v_state_ag[:, n], v_state_g[:, n], v_action[:, n] = o, o_ag, o_g, a
                    o, o_ag = o_2, o_2_ag
                v_state[:, -1], v_state_ag[:, -1] = o, o_ag

                for n in range(len(v_state)):
                    memory.push_v([v_state[n][None, :], v_state_ag[n][None, :],
                                   v_state_g[n][None, :], v_action[n][None, :]])

            if len(memory) > args.batch_size and args.updates_per_step > 0:
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
                        if args.her_normalize:
                            state_norm = memory.o_norm.normalize(state_eval["observation"])
                            goal_norm = memory.g_norm.normalize(state_eval["desired_goal"])
                            state_eval_ = np.concatenate((state_norm, goal_norm), axis=-1)
                        else:
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

                if args.lcercc or args.lcerrm:
                    memory.save_cluster_centers(total_numsteps, writer.log_dir)

                print("----------------------------------------")
                print("Timestep Eval - Test Episodes: {}, Avg. Reward: {}, Avg. Success: {}".format(episodes_eval, round(avg_reward_eval, 2), round(total_success_rate, 3)))
                print("----------------------------------------")

        # Fill up replay memory
        steps_taken = len(episode_trajectory)

        o, ag, g, a = [], [], [], []

        # Normal experience replay
        for t in range(steps_taken):
            state, action, reward, next_state, mask = episode_trajectory[t]
            # Append transition to memory
            o.append(state["observation"]), ag.append(state["achieved_goal"])
            g.append(state["desired_goal"]), a.append(action)

        o.append(next_state["observation"]), ag.append(next_state["achieved_goal"])
        o, ag, g, a = np.array([o]), np.array([ag]), np.array([g]), np.array([a])
        memory.push_episode([o, ag, g, a])

        if args.lcercc or args.lcerrm:
            o = memory.buffers["obs"][total_numsteps - steps_taken:total_numsteps]
            ag = memory.buffers["ag"][total_numsteps - steps_taken:total_numsteps]
            a = memory.buffers["actions"][total_numsteps - steps_taken:total_numsteps]
            o_2 = memory.buffers["obs_next"][total_numsteps - steps_taken:total_numsteps]
            ag_2 = memory.buffers["ag_next"][total_numsteps - steps_taken:total_numsteps]
            g = memory.buffers["g"][total_numsteps - steps_taken:total_numsteps]
            memory.update_clusters(o, ag, a, o_2, ag_2, g)

        if args.nmer:
            memory.update_neighbours()

        if len(memory) > args.batch_size and args.n_update_batches > 0:
            # Number of updates per step in environment
            for i in range(args.n_update_batches):
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
    parser.add_argument('--gamma', type=float, default=0.98, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='G',
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
    parser.add_argument('--updates-per-step', type=int, default=0, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start-steps', type=int, default=500, metavar='N',
                        help='Steps sampling random actions (default: 500)')
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
    parser.add_argument('--rollout-min-epoch', type=int, default=0, metavar='N',
                        help='rollout min epoch (default: 0)')
    parser.add_argument('--rollout-max-epoch', type=int, default=1, metavar='N',
                        help='rollout max epoch (default: 1)')
    parser.add_argument('--rollout-min-length', type=int, default=3, metavar='N',
                        help='rollout min length (default: 3)')
    parser.add_argument('--rollout-max-length', type=int, default=3, metavar='N',
                        help='rollout max length (default: 3)')
    parser.add_argument('--deterministic-model', action="store_true",
                        help='use Model-based deterministic model (default: False)')
    parser.add_argument('--nmer', action="store_true",
                        help='use NMER (default: False)')
    parser.add_argument('--k-neighbours', type=int, default=10, metavar='N',
                        help='amount of neighbours to use (default: 10)')
    parser.add_argument('--per', action="store_true",
                        help='use PER (default: False)')
    parser.add_argument('--her-replay-strategy', type=str, default="future", metavar='N',
                        help='replay strategy to use (default: "future"')
    parser.add_argument('--her-replay-k', type=int, default=4, metavar='N',
                        help='replay k to use (default: 4)')
    parser.add_argument('--her', action="store_true",
                        help='use HER (default: False)')
    parser.add_argument('--her-normalize', action="store_true",
                        help='use HER normalize (default: False)')
    parser.add_argument('--n-update-batches', type=int, default=20,
                        help='updates per rollout (default: 20)')
    parser.add_argument('--lcercc', action="store_true",
                        help='use LCERCC (default: False)')
    parser.add_argument('--lcerrm', action="store_true",
                        help='use LCERRM (default: False)')

    args = parser.parse_args()

    assert args.updates_per_step > 0 and args.n_update_batches == 0 or \
           args.updates_per_step == 0 and args.n_update_batches > 0, "One of --updates-per-step or --n-update-batches must be zero (0)"

    assert args.lcercc and not args.lcerrm or not args.lcercc and args.lcerrm or \
           not args.lcercc and not args.lcerrm, "LCERCC and LCERRM must not be both active at the same time"

    main(args)
