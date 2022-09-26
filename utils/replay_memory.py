import os
import pickle
import random
import numpy as np
import gym
from utils.utils import termination_fn
from utils.segment_tree import MinSegmentTree, SumSegmentTree
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans


class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity


class HerReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, state_ag, state_g, action, reward, next_state, next_state_ag, next_state_g, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, state_ag, state_g, action, reward, next_state, next_state_ag, next_state_g, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state_, _, state_g, action, reward, next_state_, _, next_state_g, done = map(np.stack, zip(*batch))
        state = np.concatenate((state_, state_g), axis=-1)
        next_state = np.concatenate((next_state_, next_state_g), axis=-1)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity


class MbpoReplayMemory(ReplayMemory):
    def __init__(self, capacity, seed, v_capacity=None, v_ratio=1.0, env_name="Hopper-v2", args=None):
        super().__init__(capacity, seed)

        assert args is not None, "args must not be None"

        if v_capacity is None:
            self.v_capacity = capacity
        else:
            self.v_capacity = v_capacity
        self.v_buffer = []
        self.v_position = 0

        # MBPO settings
        self.args = args
        self.rollout_length = 1  # always start with 1
        self.v_ratio = v_ratio
        self.env_name = env_name

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
        self.v_capacity = self.args.model_retain_epochs * model_steps_per_epoch

        state, action, reward, next_state, done = map(np.stack, zip(*self.buffer))
        self.v_buffer = []
        self.v_position = 0

        for n in range(len(state)):
            self.push_v(state[n], action[n], float(reward[n]), next_state[n], float(done[n]))

    def push_v(self, state, action, reward, next_state, done):
        if len(self.v_buffer) < self.v_capacity:
            self.v_buffer.append(None)
        self.v_buffer[self.v_position] = (state, action, reward, next_state, done)
        self.v_position = (self.v_position + 1) % self.v_capacity

    def sample_r(self, batch_size):
        if batch_size < len(self.buffer):
            state, action, reward, next_state, done = super().sample(batch_size=batch_size)
        else:
            sample_indices = np.random.randint(len(self.buffer), size=batch_size)
            state_, action_, reward_, next_state_, done_ = map(np.stack, zip(*self.buffer))
            state, action, reward = state_[sample_indices], action_[sample_indices], reward_[sample_indices]
            next_state, done = next_state_[sample_indices], done_[sample_indices]
        return state, action, reward, next_state, done

    def sample_v(self, batch_size):
        batch = random.sample(self.v_buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def sample(self, batch_size):
        if len(self.v_buffer) > 0:
            v_batch_size = int(self.v_ratio * batch_size)
            batch_size = batch_size - v_batch_size

            if batch_size == 0:
                v_state, v_action, v_reward, v_next_state, v_done = self.sample_v(batch_size=v_batch_size)
                return v_state, v_action, v_reward, v_next_state, v_done

            if v_batch_size == 0:
                state, action, reward, next_state, done = self.sample_r(batch_size=batch_size)
                return state, action, reward, next_state, done

            state, action, reward, next_state, done = self.sample_r(batch_size=batch_size)
            v_state, v_action, v_reward, v_next_state, v_done = self.sample_v(batch_size=v_batch_size)
        else:
            state, action, reward, next_state, done = self.sample_r(batch_size=batch_size)
            return state, action, reward, next_state, done

        state = np.concatenate((state, v_state), axis=0)
        action = np.concatenate((action, v_action), axis=0)
        reward = np.concatenate((reward, v_reward), axis=0)
        next_state = np.concatenate((next_state, v_next_state), axis=0)
        done = np.concatenate((done, v_done), axis=0)

        return state, action, reward, next_state, done


class MbpoHerReplayMemory(HerReplayMemory):
    def __init__(self, capacity, seed, v_capacity=None, v_ratio=1.0, env_name="Hopper-v2", args=None):
        super().__init__(capacity, seed)

        assert args is not None, "args must not be None"

        if v_capacity is None:
            self.v_capacity = capacity
        else:
            self.v_capacity = v_capacity
        self.v_buffer = []
        self.v_position = 0

        # MBPO settings
        self.args = args
        self.rollout_length = 1  # always start with 1
        self.v_ratio = v_ratio
        self.env_name = env_name

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
        self.v_capacity = self.args.model_retain_epochs * model_steps_per_epoch * (1 + self.args.her_replay_k)

        state, state_ag, state_g, action, reward, next_state, next_state_ag, next_state_g, done = map(np.stack, zip(*self.buffer))
        self.v_buffer = []
        self.v_position = 0

        for n in range(len(state)):
            self.push_v(state[n], state_ag[n], state_g[n], action[n], float(reward[n]), next_state[n], next_state_ag[n], next_state_g[n], float(done[n]))

    def push_v(self, state, state_ag, state_g, action, reward, next_state, next_state_ag, next_state_g, done):
        if len(self.v_buffer) < self.v_capacity:
            self.v_buffer.append(None)
        self.v_buffer[self.v_position] = (state, state_ag, state_g, action, reward, next_state, next_state_ag, next_state_g, done)
        self.v_position = (self.v_position + 1) % self.v_capacity

    def sample_r(self, batch_size):
        if batch_size < len(self.buffer):
            batch = random.sample(self.buffer, batch_size)
            state, state_ag, state_g, action, reward, next_state, next_state_ag, next_state_g, done = map(np.stack, zip(*batch))
            return state, state_ag, state_g, action, reward, next_state, next_state_ag, next_state_g, done
        else:
            sample_indices = np.random.randint(len(self.buffer), size=batch_size)
            state, state_ag, state_g, action, reward, next_state, next_state_ag, next_state_g, done = map(np.stack, zip(*[self.buffer[n] for n in sample_indices]))
        return state, state_ag, state_g, action, reward, next_state, next_state_ag, next_state_g, done

    def sample_v(self, batch_size):
        batch = random.sample(self.v_buffer, batch_size)
        state, state_ag, state_g, action, reward, next_state, next_state_ag, next_state_g, done = map(np.stack, zip(*batch))
        return state, state_ag, state_g, action, reward, next_state, next_state_ag, next_state_g, done

    def sample(self, batch_size):
        if len(self.v_buffer) > 0:
            v_batch_size = int(self.v_ratio * batch_size)
            batch_size = batch_size - v_batch_size

            if batch_size == 0:
                v_state_, _, v_state_g, v_action, v_reward, v_next_state_, _, v_next_state_g, v_done = self.sample_v(batch_size=v_batch_size)
                v_state = np.concatenate((v_state_, v_state_g), axis=-1)
                v_next_state = np.concatenate((v_next_state_, v_next_state_g), axis=-1)
                return v_state, v_action, v_reward, v_next_state, v_done

            if v_batch_size == 0:
                state_, _, state_g, action, reward, next_state_, _, next_state_g, done = self.sample_r(batch_size=batch_size)
                state = np.concatenate((state_, state_g), axis=-1)
                next_state = np.concatenate((next_state_, next_state_g), axis=-1)
                return state, action, reward, next_state, done

            state_, _, state_g, action, reward, next_state_, _, next_state_g, done = self.sample_r(batch_size=batch_size)
            state = np.concatenate((state_, state_g), axis=-1)
            next_state = np.concatenate((next_state_, next_state_g), axis=-1)

            v_state_, _, v_state_g, v_action, v_reward, v_next_state_, _, v_next_state_g, v_done = self.sample_v(batch_size=v_batch_size)
            v_state = np.concatenate((v_state_, v_state_g), axis=-1)
            v_next_state = np.concatenate((v_next_state_, v_next_state_g), axis=-1)
        else:
            state_, _, state_g, action, reward, next_state_, _, next_state_g, done = self.sample_r(batch_size=batch_size)
            state = np.concatenate((state_, state_g), axis=-1)
            next_state = np.concatenate((next_state_, next_state_g), axis=-1)
            return state, action, reward, next_state, done

        state = np.concatenate((state, v_state), axis=0)
        action = np.concatenate((action, v_action), axis=0)
        reward = np.concatenate((reward, v_reward), axis=0)
        next_state = np.concatenate((next_state, v_next_state), axis=0)
        done = np.concatenate((done, v_done), axis=0)

        return state, action, reward, next_state, done


class NmerReplayMemory(ReplayMemory):
    def __init__(self, capacity, seed, k_neighbours=10, env_name="Hopper-v2"):
        super().__init__(capacity, seed)

        # Interpolation settings
        self.k_neighbours = k_neighbours
        self.nn_indices = None

        self.env_name = env_name

    def update_neighbours(self):
        # Get whole buffer
        state, action, _, _, _ = map(np.stack, zip(*self.buffer))

        # Construct Z-space
        z_space = np.concatenate((state, action), axis=-1)
        z_space_norm = StandardScaler(with_mean=False).fit_transform(z_space)

        # NearestNeighbors - object
        k_nn = NearestNeighbors(n_neighbors=self.k_neighbours).fit(z_space_norm)
        self.nn_indices = k_nn.kneighbors(z_space_norm, return_distance=False)

    def sample(self, batch_size):
        assert self.nn_indices is not None, "Memory not prepared yet! Call .update_neighbours()"

        # Sample
        sample_indices = np.random.randint(len(self.buffer), size=batch_size)
        nn_indices = self.nn_indices[sample_indices].copy()

        # Remove itself, shuffle and chose
        nn_indices = nn_indices[:, 1:]
        indices = np.random.rand(*nn_indices.shape).argsort(axis=1)
        nn_indices = np.take_along_axis(nn_indices, indices, axis=1)
        nn_indices = nn_indices[:, 0]

        # Actually sample
        state, action, reward, next_state, _ = map(np.stack, zip(*[self.buffer[n] for n in sample_indices]))
        nn_state, nn_action, nn_reward, nn_next_state, _ = map(np.stack, zip(*[self.buffer[n] for n in nn_indices]))

        delta_state = (next_state - state).copy()
        nn_delta_state = (nn_next_state - nn_state).copy()

        # Linearly interpolate sample and neighbor points
        mixing_param = np.random.uniform(size=(len(state), 1))
        state = state * mixing_param + nn_state * (1 - mixing_param)
        action = action * mixing_param + nn_action * (1 - mixing_param)
        reward = reward * mixing_param.squeeze() + nn_reward * (1 - mixing_param).squeeze()
        delta_state = delta_state * mixing_param + nn_delta_state * (1 - mixing_param)
        next_state = state + delta_state

        done = termination_fn(self.env_name, state, action, next_state)
        mask = np.invert(done).astype(float).squeeze()

        return state, action, reward, next_state, mask


class MbpoNmerReplayMemory(MbpoReplayMemory):
    def __init__(self, capacity, seed, v_capacity=None, v_ratio=1.0, env_name="Hopper-v2", args=None, k_neighbours=10):
        super().__init__(capacity, seed, v_capacity=v_capacity, v_ratio=v_ratio, env_name=env_name, args=args)

        # Interpolation settings
        self.k_neighbours = k_neighbours
        self.nn_indices = None
        self.v_nn_indices = None

    def update_neighbours(self):
        # Get whole buffer
        state, action, _, _, _ = map(np.stack, zip(*self.buffer))

        # Construct Z-space
        z_space = np.concatenate((state, action), axis=-1)
        z_space_norm = StandardScaler(with_mean=False).fit_transform(z_space)

        # NearestNeighbors - object
        k_nn = NearestNeighbors(n_neighbors=self.k_neighbours).fit(z_space_norm)
        self.nn_indices = k_nn.kneighbors(z_space_norm, return_distance=False)

    def update_v_neighbours(self):
        # Get whole buffer
        v_state, v_action, _, _, _ = map(np.stack, zip(*self.v_buffer))

        # Construct Z-space
        v_z_space = np.concatenate((v_state, v_action), axis=-1)
        v_z_space_norm = StandardScaler(with_mean=False).fit_transform(v_z_space)

        # NearestNeighbors - object
        v_k_nn = NearestNeighbors(n_neighbors=self.k_neighbours).fit(v_z_space_norm)
        self.v_nn_indices = v_k_nn.kneighbors(v_z_space_norm, return_distance=False)

    def sample(self, batch_size):
        assert self.nn_indices is not None, "Memory not prepared yet! Call .update_neighbours()"
        assert self.v_nn_indices is not None, "Memory not prepared yet! Call .update_v_neighbours()"

        v_batch_size = int(self.v_ratio * batch_size)
        batch_size = batch_size - v_batch_size

        # Sample
        v_sample_indices = np.random.randint(len(self.v_buffer), size=v_batch_size)
        v_nn_indices = self.v_nn_indices[v_sample_indices].copy()

        sample_indices = np.random.randint(len(self.buffer), size=batch_size)
        nn_indices = self.nn_indices[sample_indices].copy()

        # Remove itself, shuffle and chose
        v_nn_indices = v_nn_indices[:, 1:]
        v_indices = np.random.rand(*v_nn_indices.shape).argsort(axis=1)
        v_nn_indices = np.take_along_axis(v_nn_indices, v_indices, axis=1)
        v_nn_indices = v_nn_indices[:, 0]

        nn_indices = nn_indices[:, 1:]
        indices = np.random.rand(*nn_indices.shape).argsort(axis=1)
        nn_indices = np.take_along_axis(nn_indices, indices, axis=1)
        nn_indices = nn_indices[:, 0]

        # Actually sample
        v_state, v_action, v_reward, v_next_state, _ = map(np.stack, zip(*[self.v_buffer[n] for n in v_sample_indices]))
        v_nn_state, v_nn_action, v_nn_reward, v_nn_next_state, _ = map(np.stack, zip(*[self.v_buffer[n] for n in v_nn_indices]))

        v_delta_state = (v_next_state - v_state).copy()
        v_nn_delta_state = (v_nn_next_state - v_nn_state).copy()

        state, action, reward, next_state, _ = map(np.stack, zip(*[self.buffer[n] for n in sample_indices]))
        nn_state, nn_action, nn_reward, nn_next_state, _ = map(np.stack, zip(*[self.buffer[n] for n in nn_indices]))

        delta_state = (next_state - state).copy()
        nn_delta_state = (nn_next_state - nn_state).copy()

        # Linearly interpolate sample and neighbor points
        v_mixing_param = np.random.uniform(size=(len(v_state), 1))
        v_state = v_state * v_mixing_param + v_nn_state * (1 - v_mixing_param)
        v_action = v_action * v_mixing_param + v_nn_action * (1 - v_mixing_param)
        v_reward = v_reward * v_mixing_param.squeeze() + v_nn_reward * (1 - v_mixing_param).squeeze()
        v_delta_state = v_delta_state * v_mixing_param + v_nn_delta_state * (1 - v_mixing_param)
        v_next_state = v_state + v_delta_state

        v_done = termination_fn(self.env_name, v_state, v_action, v_next_state)
        v_mask = np.invert(v_done).astype(float).squeeze()

        mixing_param = np.random.uniform(size=(len(state), 1))
        state = state * mixing_param + nn_state * (1 - mixing_param)
        action = action * mixing_param + nn_action * (1 - mixing_param)
        reward = reward * mixing_param.squeeze() + nn_reward * (1 - mixing_param).squeeze()
        delta_state = delta_state * mixing_param + nn_delta_state * (1 - mixing_param)
        next_state = state + delta_state

        done = termination_fn(self.env_name, state, action, next_state)
        mask = np.invert(done).astype(float).squeeze()

        # Concatenate
        state = np.concatenate((state, v_state), axis=0)
        action = np.concatenate((action, v_action), axis=0)
        reward = np.concatenate((reward, v_reward), axis=0)
        next_state = np.concatenate((next_state, v_next_state), axis=0)
        mask = np.concatenate((mask, v_mask), axis=0)

        return state, action, reward, next_state, mask


class BaseReplayMemory:
    def __init__(self, capacity, seed, state_dim, action_dim):
        random.seed(seed)
        self.capacity = capacity
        self.size = 0
        self.position = 0

        self.buffer = {
            "state": np.empty(shape=(capacity, state_dim)),
            "next_state": np.empty(shape=(capacity, state_dim)),
            "action": np.empty(shape=(capacity, action_dim)),
            "reward": np.empty(shape=(capacity, 1)),
            "mask": np.empty(shape=(capacity, 1)),
        }

        self.action_dim = action_dim
        self.state_dim = state_dim

    def push(self, state, action, reward, next_state, done):
        self.buffer["state"][self.position] = state
        self.buffer["next_state"][self.position] = next_state
        self.buffer["action"][self.position] = action
        self.buffer["reward"][self.position] = reward
        self.buffer["mask"][self.position] = done

        self.position += 1
        if self.position % self.capacity == 0:
            self.position = 0

        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if batch_size < len(self):
            sample_indices = np.random.choice(len(self), size=batch_size, replace=False)
        else:
            sample_indices = np.random.randint(len(self), size=batch_size)

        state = self.buffer["state"][sample_indices]
        action = self.buffer["action"][sample_indices]
        reward = self.buffer["reward"][sample_indices]
        next_state = self.buffer["next_state"][sample_indices]
        mask = self.buffer["mask"][sample_indices]

        return state, action, reward, next_state, mask

    def __len__(self):
        return self.size

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}.pkl".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        data = {"buffer": self.buffer, "position": self.position, "size": self.size}
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            data = pickle.load(f)
            self.buffer = data["buffer"]
            self.position = data["position"]
            self.size = data["size"]


class PerReplayMemory(BaseReplayMemory):
    def __init__(self, capacity, seed, state_dim, action_dim, priority_eps=1e-6, alpha=0.4, beta=0.6):
        super().__init__(capacity, seed, state_dim=state_dim, action_dim=action_dim)

        self.alpha = alpha
        self.beta = beta
        self.priority_eps = priority_eps

        self.max_priority = 1.0
        self.tree_position = 0

        # tree capacity must be positive and a power of 2
        tree_capacity = 1
        while tree_capacity < self.capacity:
            tree_capacity *= 2

        self.sum_segment_tree = SumSegmentTree(tree_capacity)
        self.min_segment_tree = MinSegmentTree(tree_capacity)

    def push(self, state, action, reward, next_state, done):
        super().push(state, action, reward, next_state, done)

        self.sum_segment_tree[self.tree_position] = self.max_priority ** self.alpha
        self.min_segment_tree[self.tree_position] = self.max_priority ** self.alpha

        self.tree_position += 1
        if self.tree_position % self.capacity == 0:
            self.tree_position = 0

    def sample(self, batch_size):
        sample_indices = self._sample_proportional(batch_size)

        state = self.buffer["state"][sample_indices]
        action = self.buffer["action"][sample_indices]
        reward = self.buffer["reward"][sample_indices]
        next_state = self.buffer["next_state"][sample_indices]
        mask = self.buffer["mask"][sample_indices]

        weights = np.array([self._calculate_weight(n, self.beta) for n in sample_indices])

        return state, action, reward, next_state, mask, weights[:, None], sample_indices

    def update_priorities(self, indices, priorities):
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_segment_tree[idx] = priority ** self.alpha
            self.min_segment_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self, batch_size):
        sample_indices = []
        p_total = self.sum_segment_tree.sum(0, len(self) - 1)
        segment = p_total / batch_size

        for n in range(batch_size):
            a = segment * n
            b = segment * (n + 1)
            upper_bound = random.uniform(a, b)
            idx = self.sum_segment_tree.retrieve(upper_bound)
            sample_indices.append(idx)

        return sample_indices

    def _calculate_weight(self, idx, beta):
        p_min = self.min_segment_tree.min() / self.sum_segment_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        p_sample = self.sum_segment_tree[idx] / self.sum_segment_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight


class PerNmerReplayMemory(PerReplayMemory):
    def __init__(self, capacity, seed, state_dim, action_dim, priority_eps=1e-6, alpha=0.4, beta=0.6, k_neighbours=10,
                 env_name="Hopper-v2"):
        super().__init__(capacity, seed, state_dim=state_dim, action_dim=action_dim,
                         priority_eps=priority_eps, alpha=alpha, beta=beta)

        # Interpolation settings
        self.k_neighbours = k_neighbours
        self.nn_indices = None

        self.env_name = env_name

    def update_neighbours(self):
        # Construct Z-space
        z_space = np.concatenate((self.buffer["state"][:self.position], self.buffer["action"][:self.position]), axis=-1)
        z_space_norm = StandardScaler(with_mean=False).fit_transform(z_space)

        # NearestNeighbors - object
        k_nn = NearestNeighbors(n_neighbors=self.k_neighbours).fit(z_space_norm)
        self.nn_indices = k_nn.kneighbors(z_space_norm, return_distance=False)

    def sample(self, batch_size):
        assert self.nn_indices is not None, "Memory not prepared yet! Call .update_neighbours()"

        sample_indices = self._sample_proportional(batch_size)
        nn_indices = self.nn_indices[sample_indices].copy()

        # Remove itself, shuffle and chose
        nn_indices = nn_indices[:, 1:]
        indices = np.random.rand(*nn_indices.shape).argsort(axis=1)
        nn_indices = np.take_along_axis(nn_indices, indices, axis=1)
        nn_indices = nn_indices[:, 0]

        state, nn_state = self.buffer["state"][sample_indices], self.buffer["state"][nn_indices]
        action, nn_action = self.buffer["action"][sample_indices], self.buffer["action"][nn_indices]
        reward, nn_reward = self.buffer["reward"][sample_indices], self.buffer["reward"][nn_indices]
        next_state, nn_next_state = self.buffer["next_state"][sample_indices], self.buffer["next_state"][nn_indices]

        delta_state = (next_state - state).copy()
        nn_delta_state = (nn_next_state - nn_state).copy()

        # Linearly interpolate sample and neighbor points
        mixing_param = np.random.uniform(size=(len(state), 1))
        state = state * mixing_param + nn_state * (1 - mixing_param)
        action = action * mixing_param + nn_action * (1 - mixing_param)
        reward = reward * mixing_param + nn_reward * (1 - mixing_param)
        delta_state = delta_state * mixing_param + nn_delta_state * (1 - mixing_param)
        next_state = state + delta_state

        done = termination_fn(self.env_name, state, action, next_state)
        mask = np.invert(done).astype(float)

        weights = np.array([self._calculate_weight(n, self.beta) for n in sample_indices])

        return state, action, reward, next_state, mask, weights[:, None], sample_indices


class SimpleLocalApproximationReplayMemory(BaseReplayMemory):
    def __init__(self, capacity, seed, state_dim, action_dim,
                 v_capacity=None, v_ratio=0.95, env_name="Hopper-v2", args=None):
        super().__init__(capacity, seed, state_dim=state_dim, action_dim=action_dim)

        assert args is not None, "args must not be None"

        if v_capacity is None:
            self.v_capacity = args.updates_per_step * args.epoch_length
        else:
            self.v_capacity = v_capacity
        self.v_buffer = BaseReplayMemory(self.v_capacity, seed, state_dim, action_dim)

        self.n_clusters = args.epoch_length
        self.scaler = StandardScaler()
        self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=seed, batch_size=2048, reassignment_ratio=0)

        self.clusters = [StandardScaler() for _ in range(self.n_clusters)]

        env = gym.make(env_name)
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low

        self.v_ratio = v_ratio
        self.env_name = env_name
        self.args = args
        self.rollout_length = 0
        self.seed = seed

        self.debug = False
        self.cluster_centers_kmeans = []
        self.cluster_centers = []
        self.timesteps = []

    def save_cluster_centers(self, timesteps, save_path):
        if not self.debug:
            return

        self.cluster_centers_kmeans.append(self.kmeans.cluster_centers_.copy())
        cc = np.empty(shape=(self.n_clusters, 2 * self.state_dim + 1 + self.action_dim))
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

    def update_clusters(self, o, a, r, o_2):
        z_space = np.concatenate((o, a), axis=-1)
        self.scaler.partial_fit(z_space)
        z_space_norm = self.scaler.transform(z_space)
        self.kmeans = self.kmeans.partial_fit(z_space_norm)

        # max startup steps, else max max_timesteps
        labels = self.kmeans.labels_
        z_space = np.concatenate((z_space, r, o_2), axis=-1)
        for n in range(len(labels)):
            self.clusters[labels[n]] = self.clusters[labels[n]].partial_fit(z_space[n].reshape(1, -1))

    def push_v(self, state, action, reward, next_state, done):
        self.v_buffer.push(state, action, reward, next_state, done)

    def sample_v(self, batch_size):
        return self.v_buffer.sample(batch_size=batch_size)

    def sample_r(self, batch_size):
        return super().sample(batch_size=batch_size)

    def sample(self, batch_size):
        state, action, reward, next_state, done = self.sample_r(batch_size=batch_size)

        z_space = np.concatenate((state, action), axis=-1)
        z_space_norm = self.scaler.transform(z_space)
        cluster_labels = self.kmeans.predict(z_space_norm)

        v_state = np.empty(shape=(batch_size, self.state_dim))
        v_action = np.empty(shape=(batch_size, self.action_dim))
        v_reward = np.empty(shape=(batch_size, 1))
        v_next_state = np.empty(shape=(batch_size, self.state_dim))
        for n in range(batch_size):
            mu, std = self.clusters[cluster_labels[n]].mean_, self.clusters[cluster_labels[n]].scale_ * 0.01
            v_state[n] = mu[:self.state_dim] + np.random.normal(size=mu[:self.state_dim].shape) * std[:self.state_dim]
            v_action[n] = mu[self.state_dim:self.state_dim + self.action_dim] + np.random.normal(size=mu[self.state_dim:self.state_dim + self.action_dim].shape) * std[self.state_dim:self.state_dim + self.action_dim]
            v_reward[n] = mu[self.state_dim + self.action_dim:self.state_dim + self.action_dim + 1] + np.random.normal(size=mu[self.state_dim + self.action_dim:self.state_dim + self.action_dim + 1].shape) * std[self.state_dim + self.action_dim:self.state_dim + self.action_dim + 1]
            v_next_state[n] = mu[-self.state_dim:] + np.random.normal(size=mu[-self.state_dim:].shape) * std[-self.state_dim:]
        v_action = np.clip(v_action, self.min_action, self.max_action)

        delta_state = (next_state - state).copy()
        v_delta_state = (v_next_state - v_state).copy()

        # Linearly interpolate sample and neighbor points
        mixing_param = np.random.uniform(size=(len(state), 1))
        state = state * mixing_param + v_state * (1 - mixing_param)
        action = action * mixing_param + v_action * (1 - mixing_param)
        reward = (reward * mixing_param + v_reward * (1 - mixing_param)).squeeze()
        delta_state = delta_state * mixing_param + v_delta_state * (1 - mixing_param)
        next_state = state + delta_state

        done = termination_fn(self.env_name, state, action, next_state)
        mask = np.invert(done).astype(float).squeeze()

        return state, action, reward, next_state, mask
