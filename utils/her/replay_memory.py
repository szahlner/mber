import threading
import numpy as np
from mpi4py import MPI


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
        v_env_params = env_params
        v_env_params["max_timesteps"] = self.rollout_length
        self.v_buffer = HerReplayMemory(v_env_params, buffer_size, sample_func=sample_func, normalize=normalize)

        self.r_buffer = SimpleReplayMemory(env_params, buffer_size)

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

        # Reshape
        mb_obs_next =  mb_obs[:, 1:, :].reshape((self.T * batch_size, self.env_params["obs"]))
        mb_obs = mb_obs[:, :-1, :].reshape((self.T * batch_size, self.env_params["obs"]))
        mb_ag_next = mb_ag[:, 1:, :].reshape((self.T * batch_size, self.env_params["goal"]))
        mb_ag = mb_ag[:, :-1, :].reshape((self.T * batch_size, self.env_params["goal"]))
        mb_g = mb_g.reshape((self.T * batch_size, self.env_params["goal"]))
        mb_actions = mb_actions.reshape((self.T * batch_size, self.env_params["action"]))
        for n in range(len(mb_obs)):
            self.r_buffer.push_transition([mb_obs[n], mb_obs_next[n], mb_ag[n], mb_ag_next[n], mb_g[n], mb_actions[n]])

    def sample_r(self, batch_size):
        return self.r_buffer.sample(batch_size)

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

        v_env_params = self.env_params
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
                return v_obs, v_actions, v_rewards, v_obs_next, v_done

            if v_batch_size == 0:
                obs, actions, rewards, obs_next, done = super().sample(batch_size)
                return obs, actions, rewards, obs_next, done

            v_obs, v_actions, v_rewards, v_obs_next, v_done = self.v_buffer.sample(v_batch_size)
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
    def __init__(self, env_params, buffer_size):
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
    def sample(self, batch_size):
        idx = np.random.randint(0, self.current_size, size=batch_size)

        transitions = {
            "obs": self.buffers["obs"][idx],
            "obs_next": self.buffers["obs_next"][idx],
            "ag": self.buffers["ag"][idx],
            "ag_next": self.buffers["ag_next"][idx],
            "g": self.buffers["g"][idx],
            "actions": self.buffers["actions"][idx],
        }

        return transitions
