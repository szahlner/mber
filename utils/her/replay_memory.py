import os
import threading
import gym
import numpy as np
import pickle
import copy
from mpi4py import MPI
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans


class Normalizer:
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range
        # some local information
        self.local_sum = np.zeros(self.size, np.float32)
        self.local_sumsq = np.zeros(self.size, np.float32)
        self.local_count = np.zeros(1, np.float32)
        # get the total sum sumsq and sum count
        self.total_sum = np.zeros(self.size, np.float32)
        self.total_sumsq = np.zeros(self.size, np.float32)
        self.total_count = np.ones(1, np.float32)
        # get the mean and std
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)
        # thread locker
        self.lock = threading.Lock()

    # update the parameters of the normalizer
    def update(self, v):
        v = v.reshape(-1, self.size)
        # do the computing
        with self.lock:
            self.local_sum += v.sum(axis=0)
            self.local_sumsq += (np.square(v)).sum(axis=0)
            self.local_count[0] += v.shape[0]

    # sync the parameters across the cpus
    def sync(self, local_sum, local_sumsq, local_count):
        local_sum[...] = self._mpi_average(local_sum)
        local_sumsq[...] = self._mpi_average(local_sumsq)
        local_count[...] = self._mpi_average(local_count)
        return local_sum, local_sumsq, local_count

    def recompute_stats(self):
        with self.lock:
            local_count = self.local_count.copy()
            local_sum = self.local_sum.copy()
            local_sumsq = self.local_sumsq.copy()
            # reset
            self.local_count[...] = 0
            self.local_sum[...] = 0
            self.local_sumsq[...] = 0
        # synrc the stats
        sync_sum, sync_sumsq, sync_count = self.sync(local_sum, local_sumsq, local_count)
        # update the total stuff
        self.total_sum += sync_sum
        self.total_sumsq += sync_sumsq
        self.total_count += sync_count
        # calculate the new mean and std
        self.mean = self.total_sum / self.total_count
        self.std = np.sqrt(np.maximum(np.square(self.eps), (self.total_sumsq / self.total_count) - np.square(
            self.total_sum / self.total_count)))

    # average across the cpu's data
    def _mpi_average(self, x):
        buf = np.zeros_like(x)
        MPI.COMM_WORLD.Allreduce(x, buf, op=MPI.SUM)
        buf /= MPI.COMM_WORLD.Get_size()
        return buf

    # normalize the observation
    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        return np.clip((v - self.mean) / (self.std), -clip_range, clip_range)


class HerSampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k

        if self.replay_strategy == "future":
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0

        self.reward_func = reward_func

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions

        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}

        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # replace go with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag

        # to get the params to re-compute reward
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        return transitions


class HerReplayMemory:
    def __init__(self, env_params, buffer_size, sample_func, normalize=True):
        self.env_params = env_params
        self.T = env_params["max_timesteps"]
        self.size = buffer_size // self.T
        self.normalize = normalize

        # Memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func

        # Create the buffer to store info
        self.buffers = {
            "obs": np.empty([self.size, self.T + 1, self.env_params["obs"]]),
            "ag": np.empty([self.size, self.T + 1, self.env_params["goal"]]),
            "g": np.empty([self.size, self.T, self.env_params["goal"]]),
            "actions": np.empty([self.size, self.T, self.env_params["action"]]),
        }

        # Thread lock
        self.lock = threading.Lock()

        # Normalizer
        if self.normalize:
            self.o_norm = Normalizer(self.env_params["obs"])
            self.g_norm = Normalizer(self.env_params["goal"])

    def __len__(self):
        return self.n_transitions_stored

    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs,
                       'ag': mb_ag,
                       'g': mb_g,
                       'actions': mb_actions,
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.sample_func(buffer_temp, num_transitions)
        # pre-process the obs and g
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    # Store the episode
    def push_episode(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        batch_size = mb_obs.shape[0]

        if self.normalize:
            self._update_normalizer(episode_batch)

        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)

            # Store the information
            self.buffers["obs"][idxs] = mb_obs
            self.buffers["ag"][idxs] = mb_ag
            self.buffers["g"][idxs] = mb_g
            self.buffers["actions"][idxs] = mb_actions
            self.n_transitions_stored += self.T * batch_size

    # Sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}

        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]

        temp_buffers["obs_next"] = temp_buffers["obs"][:, 1:, :]
        temp_buffers["ag_next"] = temp_buffers["ag"][:, 1:, :]

        # Sample transitions
        transitions = self.sample_func(temp_buffers, batch_size)

        if self.normalize:
            o, g = self.o_norm.normalize(transitions["obs"]), self.g_norm.normalize(transitions["g"])
            o_2 = self.o_norm.normalize(transitions["obs_next"])
        else:
            o, g, o_2 = transitions["obs"], transitions["g"], transitions["obs_next"]
        obs = np.concatenate((o, g), axis=-1)
        actions, rewards = transitions["actions"], transitions["r"].squeeze()
        obs_next = np.concatenate((o_2, g), axis=-1)
        done = np.ones_like(rewards)

        return obs, actions, rewards, obs_next, done

    def _get_storage_idx(self, inc=None):
        inc = inc or 1

        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)

        self.current_size = min(self.size, self.current_size + inc)

        if inc == 1:
            idx = idx[0]

        return idx


class HerMbpoReplayMemory(HerReplayMemory):
    def __init__(self, env_params, buffer_size, sample_func, v_ratio=0.95, normalize=True, args=None):
        super().__init__(env_params, buffer_size, sample_func, normalize=normalize)

        assert args is not None, "args must not be None"

        self.args = args

        self.v_ratio = v_ratio
        self.rollout_length = 1
        v_env_params = copy.copy(env_params)
        v_env_params["max_timesteps"] = self.rollout_length

        self.v_buffer = HerReplayMemory(v_env_params, buffer_size, sample_func=sample_func, normalize=False)
        self.r_buffer = SimpleReplayMemory(env_params, buffer_size, args=args, normalize=False)

        self.env = gym.make(args.env_name)

    # Store the episode
    def push_episode(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        batch_size = mb_obs.shape[0]

        if self.normalize:
            self._update_normalizer(episode_batch)

        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)

            # Store the information
            self.buffers["obs"][idxs] = mb_obs
            self.buffers["ag"][idxs] = mb_ag
            self.buffers["g"][idxs] = mb_g
            self.buffers["actions"][idxs] = mb_actions
            self.n_transitions_stored += self.T * batch_size

        self.r_buffer.push_episode(episode_batch)

    def sample_r(self, batch_size, return_transitions=False):
        return self.r_buffer.sample(batch_size, return_transitions=return_transitions)

    def set_rollout_length(self, current_epoch):
        self.rollout_length = int(
            min(
                max(
                    self.args.rollout_min_length + (current_epoch - self.args.rollout_min_epoch) / (self.args.rollout_max_epoch - self.args.rollout_min_epoch) * (self.args.rollout_max_length - self.args.rollout_min_length),
                    self.args.rollout_min_length
                ),
                self.args.rollout_max_length
            )
        )

    def resize_v_memory(self):
        rollouts_per_epoch = self.args.n_rollout_samples * self.args.epoch_length / self.args.update_env_model
        model_steps_per_epoch = int(self.rollout_length * rollouts_per_epoch)
        v_capacity = self.args.model_retain_epochs * model_steps_per_epoch

        temp_buffers = {}

        with self.lock:
            for key in self.v_buffer.buffers.keys():
                temp_buffers[key] = self.v_buffer.buffers[key][:self.v_buffer.current_size]

        v_env_params = copy.copy(self.env_params)
        v_env_params["max_timesteps"] = self.rollout_length
        self.v_buffer = HerReplayMemory(v_env_params, v_capacity,
                                        sample_func=self.v_buffer.sample_func, normalize=self.v_buffer.normalize)

        for n in range(len(temp_buffers["obs"])):
            self.v_buffer.push_episode([temp_buffers["obs"][n], temp_buffers["ag"][n],
                                        temp_buffers["g"][n], temp_buffers["actions"][n]])

    def push_v(self, episode_batch):
        self.v_buffer.push_episode(episode_batch)

    # Sample the data from the replay buffer
    def sample(self, batch_size):
        if len(self.v_buffer) > 0:
            v_batch_size = int(self.v_ratio * batch_size)
            batch_size = batch_size - v_batch_size

            if batch_size == 0:
                v_obs, v_actions, v_rewards, v_obs_next, v_done = self.v_buffer.sample(v_batch_size)
                if self.normalize:
                    o, g = v_obs[:, :self.env_params["obs"]], v_obs[:, self.env_params["obs"]:]
                    o_norm, g_norm = self.o_norm.normalize(o), self.g_norm.normalize(g)
                    v_obs = np.concatenate((o_norm, g_norm), axis=-1)

                    o_2, g_2 = v_obs_next[:, :self.env_params["obs"]], v_obs_next[:, self.env_params["obs"]:]
                    o_2_norm, g_2_norm = self.o_norm.normalize(o_2), self.g_norm.normalize(g_2)
                    v_obs_next = np.concatenate((o_2_norm, g_2_norm), axis=-1)
                return v_obs, v_actions, v_rewards, v_obs_next, v_done

            if v_batch_size == 0:
                obs, actions, rewards, obs_next, done = super().sample(batch_size)
                return obs, actions, rewards, obs_next, done

            v_obs, v_actions, v_rewards, v_obs_next, v_done = self.v_buffer.sample(v_batch_size)
            if self.normalize:
                o, g = v_obs[:, :self.env_params["obs"]], v_obs[:, self.env_params["obs"]:]
                o_norm, g_norm = self.o_norm.normalize(o), self.g_norm.normalize(g)
                v_obs = np.concatenate((o_norm, g_norm), axis=-1)

                o_2, g_2 = v_obs_next[:, :self.env_params["obs"]], v_obs_next[:, self.env_params["obs"]:]
                o_2_norm, g_2_norm = self.o_norm.normalize(o_2), self.g_norm.normalize(g_2)
                v_obs_next = np.concatenate((o_2_norm, g_2_norm), axis=-1)
            obs, actions, rewards, obs_next, done = super().sample(batch_size)
        else:
            obs, actions, rewards, obs_next, done = super().sample(batch_size)
            return obs, actions, rewards, obs_next, done

        state = np.concatenate((obs, v_obs), axis=0)
        action = np.concatenate((actions, v_actions), axis=0)
        reward = np.concatenate((rewards, v_rewards), axis=0)
        next_state = np.concatenate((obs_next, v_obs_next), axis=0)
        done = np.concatenate((done, v_done), axis=0)

        return state, action, reward, next_state, done


class SimpleReplayMemory:
    def __init__(self, env_params, buffer_size, args=None, normalize=True):
        assert args is not None, "args must not be None"

        self.env_params = env_params

        # Memory management
        self.current_size = 0
        self.pointer = 0
        self.max_size = buffer_size

        # Create the buffer to store info
        self.buffers = {
            "obs": np.empty([buffer_size, self.env_params["obs"]]),
            "obs_next": np.empty([buffer_size, self.env_params["obs"]]),
            "ag": np.empty([buffer_size, self.env_params["goal"]]),
            "ag_next": np.empty([buffer_size, self.env_params["goal"]]),
            "g": np.empty([buffer_size, self.env_params["goal"]]),
            "actions": np.empty([buffer_size, self.env_params["action"]]),
        }

        # Thread lock
        self.lock = threading.Lock()

        # Normalizer
        self.normalize = normalize
        if self.normalize:
            self.o_norm = Normalizer(self.env_params["obs"])
            self.g_norm = Normalizer(self.env_params["goal"])

        self.args = args
        self.T = env_params["max_timesteps"]
        self.env = gym.make(args.env_name)

    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, _ = episode_batch

        # update
        self.o_norm.update(mb_obs)
        self.g_norm.update(mb_g)
        self.g_norm.update(mb_ag)

        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def push_episode(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        batch_size = mb_obs.shape[0]

        if self.normalize:
            self._update_normalizer(episode_batch)

        # Reshape
        mb_obs_next = mb_obs[:, 1:, :].reshape((self.T * batch_size, self.env_params["obs"]))
        mb_obs = mb_obs[:, :-1, :].reshape((self.T * batch_size, self.env_params["obs"]))
        mb_ag_next = mb_ag[:, 1:, :].reshape((self.T * batch_size, self.env_params["goal"]))
        mb_ag = mb_ag[:, :-1, :].reshape((self.T * batch_size, self.env_params["goal"]))
        mb_g = mb_g.reshape((self.T * batch_size, self.env_params["goal"]))
        mb_actions = mb_actions.reshape((self.T * batch_size, self.env_params["action"]))
        for n in range(len(mb_obs)):
            self.push_transition([mb_obs[n], mb_obs_next[n], mb_ag[n], mb_ag_next[n], mb_g[n], mb_actions[n]])

    # Store the episode
    def push_transition(self, transition):
        obs, obs_next, ag, ag_next, g, actions = transition

        with self.lock:
            self.buffers["obs"][self.pointer] = obs
            self.buffers["obs_next"][self.pointer] = obs_next
            self.buffers["ag"][self.pointer] = ag
            self.buffers["ag_next"][self.pointer] = ag_next
            self.buffers["g"][self.pointer] = g
            self.buffers["actions"][self.pointer] = actions

        self.pointer = (self.pointer + 1) % self.max_size
        self.current_size = min(self.current_size + 1, self.max_size)

    # Sample the data from the replay buffer
    def sample(self, batch_size, return_transitions=False):
        idx = np.random.randint(0, self.current_size, size=batch_size)

        transitions = {
            "obs": self.buffers["obs"][idx],
            "obs_next": self.buffers["obs_next"][idx],
            "ag": self.buffers["ag"][idx],
            "ag_next": self.buffers["ag_next"][idx],
            "g": self.buffers["g"][idx],
            "actions": self.buffers["actions"][idx],
        }

        if return_transitions:
            return transitions

        if self.normalize:
            o, g = self.o_norm.normalize(transitions["obs"]), self.g_norm.normalize(transitions["g"])
            o_2 = self.o_norm.normalize(transitions["obs_next"])
        else:
            o, g, o_2 = transitions["obs"], transitions["g"], transitions["obs_next"]
        obs = np.concatenate((o, g), axis=-1)
        actions = transitions["actions"]
        rewards = self.env.compute_reward(transitions["ag_next"], transitions["g"], None)
        obs_next = np.concatenate((o_2, g), axis=-1)
        done = np.ones_like(rewards)

        return obs, actions, rewards, obs_next, done

    def __len__(self):
        return self.current_size


class HerNmerReplayMemory(SimpleReplayMemory):
    def __init__(self, env_params, buffer_size, normalize=False, args=None, k_neighbours=10):
        super().__init__(env_params, buffer_size, args=args, normalize=normalize)

        # Interpolation settings
        self.k_neighbours = k_neighbours
        self.nn_indices = None

    def update_neighbours(self):
        # Get whole buffer
        state = self.buffers["obs"][:self.current_size]
        goal = self.buffers["g"][:self.current_size]
        action = self.buffers["actions"][:self.current_size]

        # Construct Z-space
        z_space = np.concatenate((state, goal, action), axis=-1)
        z_space_norm = StandardScaler(with_mean=False).fit_transform(z_space)

        # NearestNeighbors - object
        k_nn = NearestNeighbors(n_neighbors=self.k_neighbours).fit(z_space_norm)
        self.nn_indices = k_nn.kneighbors(z_space_norm, return_distance=False)

    # Sample the data from the replay buffer
    def sample(self, batch_size, return_transitions=False):
        assert self.nn_indices is not None, "Memory not prepared yet! Call .update_neighbours()"

        # Sample
        sample_indices = np.random.randint(len(self), size=batch_size)
        nn_indices = self.nn_indices[sample_indices].copy()

        # Remove itself, shuffle and chose
        nn_indices = nn_indices[:, 1:]
        indices = np.random.rand(*nn_indices.shape).argsort(axis=1)
        nn_indices = np.take_along_axis(nn_indices, indices, axis=1)
        nn_indices = nn_indices[:, 0]

        # Actually sample
        state, ag = self.buffers["obs"][sample_indices], self.buffers["ag"][sample_indices]
        next_state, next_ag = self.buffers["obs_next"][sample_indices], self.buffers["ag_next"][sample_indices]
        action, g = self.buffers["actions"][sample_indices], self.buffers["g"][sample_indices]

        nn_state, nn_ag = self.buffers["obs"][nn_indices], self.buffers["ag"][nn_indices]
        nn_next_state, nn_next_ag = self.buffers["obs_next"][nn_indices], self.buffers["ag_next"][nn_indices]
        nn_action, nn_g = self.buffers["actions"][nn_indices], self.buffers["g"][nn_indices]

        delta_state = (next_state - state).copy()
        delta_ag = (next_ag - ag).copy()
        nn_delta_state = (nn_next_state - nn_state).copy()
        nn_delta_ag = (nn_next_ag - nn_ag).copy()

        # Linearly interpolate sample and neighbor points
        mixing_param = np.random.uniform(size=(len(state), 1))
        state = state * mixing_param + nn_state * (1 - mixing_param)
        action = action * mixing_param + nn_action * (1 - mixing_param)
        delta_state = delta_state * mixing_param + nn_delta_state * (1 - mixing_param)
        next_state = state + delta_state
        ag = ag * mixing_param + nn_ag * (1 - mixing_param)
        delta_ag = delta_ag * mixing_param + nn_delta_ag * (1 - mixing_param)
        next_ag = ag + delta_ag
        g = g * mixing_param + nn_g * (1 - mixing_param)

        # Include HER style
        current_episode = sample_indices // self.T
        current_episode_timestep = sample_indices % self.T
        current_episode_steps_left = self.T - current_episode_timestep - 1

        future_probability = 1 - 1 / (1 + self.args.her_replay_k)
        use_her = np.random.uniform(size=(len(sample_indices),)) < future_probability

        future_offset = np.random.uniform(size=batch_size) * current_episode_steps_left
        future_offset = future_offset.astype(int)
        future_tmp = future_offset + current_episode_timestep
        future_t = current_episode * self.T + future_tmp

        nn_current_episode = nn_indices // self.T
        nn_current_episode_timestep = nn_indices % self.T
        nn_current_episode_steps_left = self.T - nn_current_episode_timestep - 1

        nn_future_offset = np.random.uniform(size=batch_size) * nn_current_episode_steps_left
        nn_future_offset = nn_future_offset.astype(int)
        nn_future_tmp = nn_future_offset + nn_current_episode_timestep
        nn_future_t = nn_current_episode * self.T + nn_future_tmp

        her_ag = self.buffers["ag"][future_t][use_her] * mixing_param[use_her] + self.buffers["ag"][nn_future_t][use_her] * (1 - mixing_param)[use_her]

        # Replace goal
        g[use_her] = her_ag

        reward = self.env.compute_reward(next_ag, g, None)
        mask = np.ones_like(reward)

        if self.normalize:
            state, g = self.o_norm.normalize(state), self.g_norm.normalize(g)
            next_state = self.o_norm.normalize(next_state)

        state = np.concatenate((state, g), axis=-1)
        next_state = np.concatenate((next_state, g), axis=-1)

        return state, action, reward, next_state, mask


class HerSlappReplayMemory(SimpleReplayMemory):
    def __init__(self, env_params, buffer_size, normalize=False, args=None, debug=False):
        super().__init__(env_params, buffer_size, args=args, normalize=normalize)

        # Cluster settings
        self.n_clusters = self.T
        self.scaler = StandardScaler()
        self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=args.seed, batch_size=2048, reassignment_ratio=0)
        self.clusters = [StandardScaler() for _ in range(self.n_clusters)]

        self.max_action = self.env.action_space.high
        self.min_action = self.env.action_space.low

        self.debug = debug
        self.cluster_centers_kmeans = []
        self.cluster_centers = []
        self.timesteps = []

    def save_cluster_centers(self, timesteps, save_path):
        if not self.debug:
            return

        self.cluster_centers_kmeans.append(self.kmeans.cluster_centers_.copy())
        cc = np.empty(shape=(self.n_clusters, 2 * self.env_params["obs"] + self.env_params["action"] + 3 * self.env_params["goal"]))
        for n in range(self.n_clusters):
            cc[n] = self.clusters[n].mean_.copy()
        self.cluster_centers.append(cc)
        self.timesteps.append(timesteps)

        data = {
            "cluster_centers_kmeans": np.array(self.cluster_centers_kmeans),
            "cluster_centers": np.array(self.cluster_centers),
            "timesteps": np.array(self.timesteps),
        }

        save_path = os.path.join(save_path, "cluster_centers.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

    def update_clusters(self, o, ag, a, o_2, ag_2, g):
        z_space = np.concatenate((o, g, a), axis=-1)
        self.scaler.partial_fit(z_space)
        z_space_norm = self.scaler.transform(z_space)
        self.kmeans = self.kmeans.partial_fit(z_space_norm)

        # max startup steps, else max max_timesteps
        labels = self.kmeans.labels_
        z_space = np.concatenate((z_space, ag, o_2, ag_2), axis=-1)
        for n in range(len(labels)):
            self.clusters[labels[n]] = self.clusters[labels[n]].partial_fit(z_space[n].reshape(1, -1))

    # Sample the data from the replay buffer
    def sample(self, batch_size, return_transitions=False):
        # Actually sample
        sample_indices = np.random.randint(len(self), size=batch_size)

        state, ag = self.buffers["obs"][sample_indices], self.buffers["ag"][sample_indices]
        next_state, next_ag = self.buffers["obs_next"][sample_indices], self.buffers["ag_next"][sample_indices]
        action, g = self.buffers["actions"][sample_indices], self.buffers["g"][sample_indices]

        z_space = np.concatenate((state, g, action), axis=-1)
        z_space_norm = self.scaler.transform(z_space)
        cluster_labels = self.kmeans.predict(z_space_norm)

        obs_dim, action_dim, g_dim = self.env_params["obs"], self.env_params["action"], self.env_params["goal"]

        v_state = np.empty(shape=(batch_size, obs_dim))
        v_ag = np.empty(shape=(batch_size, g_dim))
        v_action = np.empty(shape=(batch_size, action_dim))
        v_next_state = np.empty(shape=(batch_size, obs_dim))
        v_next_ag = np.empty(shape=(batch_size, g_dim))
        v_g = np.empty(shape=(batch_size, g_dim))
        for n in range(batch_size):
            mu, std = self.clusters[cluster_labels[n]].mean_, self.clusters[cluster_labels[n]].scale_ * 0.01
            v_state[n] = mu[:obs_dim] + np.random.normal(size=mu[:obs_dim].shape) * std[:obs_dim]

            g_dim_start, g_dim_end = obs_dim, obs_dim + g_dim
            v_g[n] = mu[g_dim_start:g_dim_end] + np.random.normal(size=mu[g_dim_start:g_dim_end].shape) * std[g_dim_start:g_dim_end]

            action_dim_start, action_dim_end = obs_dim + g_dim, obs_dim + g_dim + action_dim
            v_action[n] = mu[action_dim_start:action_dim_end] + np.random.normal(size=mu[action_dim_start:action_dim_end].shape) * std[action_dim_start:action_dim_end]

            ag_dim_start, ag_dim_end = obs_dim + g_dim + action_dim, obs_dim + 2 * g_dim + action_dim
            v_ag[n] = mu[ag_dim_start:ag_dim_end] + np.random.normal(size=mu[ag_dim_start:ag_dim_end].shape) * std[ag_dim_start:ag_dim_end]

            obs_next_dim_start, obs_next_dim_end = obs_dim + 2 * g_dim + action_dim, 2 * obs_dim + 2 * g_dim + action_dim
            v_next_state[n] = mu[obs_next_dim_start:obs_next_dim_end] + np.random.normal(size=mu[obs_next_dim_start:obs_next_dim_end].shape) * std[obs_next_dim_start:obs_next_dim_end]

            v_next_ag[n] = mu[-g_dim:] + np.random.normal(size=mu[-g_dim:].shape) * std[-g_dim:]
        v_action = np.clip(v_action, self.min_action, self.max_action)

        delta_state = (next_state - state).copy()
        delta_ag = (next_ag - ag).copy()
        v_delta_state = (v_next_state - v_state).copy()
        v_delta_ag = (v_next_ag - v_ag).copy()

        # Linearly interpolate sample and neighbor points
        mixing_param = np.random.uniform(size=(len(state), 1))
        state = state * mixing_param + v_state * (1 - mixing_param)
        action = action * mixing_param + v_action * (1 - mixing_param)
        delta_state = delta_state * mixing_param + v_delta_state * (1 - mixing_param)
        next_state = state + delta_state
        ag = ag * mixing_param + v_ag * (1 - mixing_param)
        delta_ag = delta_ag * mixing_param + v_delta_ag * (1 - mixing_param)
        next_ag = ag + delta_ag
        g = g * mixing_param + v_g * (1 - mixing_param)

        # Include HER style
        current_episode = sample_indices // self.T
        current_episode_timestep = sample_indices % self.T
        current_episode_steps_left = self.T - current_episode_timestep - 1

        future_probability = 1 - 1 / (1 + self.args.her_replay_k)
        use_her = np.random.uniform(size=(len(sample_indices),)) < future_probability

        future_offset = np.random.uniform(size=batch_size) * current_episode_steps_left
        future_offset = future_offset.astype(int)
        future_tmp = future_offset + current_episode_timestep
        future_t = current_episode * self.T + future_tmp

        future_state, future_ag = self.buffers["obs"][future_t], self.buffers["ag"][future_t]
        future_action, future_g = self.buffers["actions"][future_t], self.buffers["g"][future_t]

        future_z_space = np.concatenate((future_state, future_g, future_action), axis=-1)
        future_z_space_norm = self.scaler.transform(future_z_space)
        future_cluster_labels = self.kmeans.predict(future_z_space_norm)

        v_future_ag = np.empty(shape=(batch_size, g_dim))
        for n in range(batch_size):
            mu, std = self.clusters[future_cluster_labels[n]].mean_, self.clusters[future_cluster_labels[n]].scale_ * 0.01
            ag_dim_start, ag_dim_end = obs_dim + g_dim + action_dim, obs_dim + 2 * g_dim + action_dim
            v_future_ag[n] = mu[ag_dim_start:ag_dim_end] + np.random.normal(size=mu[ag_dim_start:ag_dim_end].shape) * std[ag_dim_start:ag_dim_end]

        her_ag = future_ag[use_her] * mixing_param[use_her] + v_future_ag[use_her] * (1 - mixing_param)[use_her]

        # Replace goal
        g[use_her] = her_ag

        reward = self.env.compute_reward(next_ag, g, None)
        mask = np.ones_like(reward)

        if self.normalize:
            state, g = self.o_norm.normalize(state), self.g_norm.normalize(g)
            next_state = self.o_norm.normalize(next_state)

        state = np.concatenate((state, g), axis=-1)
        next_state = np.concatenate((next_state, g), axis=-1)

        return state, action, reward, next_state, mask
