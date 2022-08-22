import os
import pickle
import random
import numpy as np
from utils.utils import termination_fn
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


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
