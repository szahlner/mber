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


class MBPOReplayMemory(ReplayMemory):
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
