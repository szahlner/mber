import numpy as np
import math
import torch
import threading
from mpi4py import MPI


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def termination_fn(env_name, obs, act, next_obs):
    if env_name == "Hopper-v2":
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        height = next_obs[:, 0]
        angle = next_obs[:, 1]
        not_done = np.isfinite(next_obs).all(axis=-1) \
                   * np.abs(next_obs[:, 1:] < 100).all(axis=-1) \
                   * (height > .7) \
                   * (np.abs(angle) < .2)

        done = ~not_done
        done = done[:, None]
        return done
    elif env_name == "Walker2d-v2":
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        height = next_obs[:, 0]
        angle = next_obs[:, 1]
        not_done = (height > 0.8) \
                   * (height < 2.0) \
                   * (angle > -1.0) \
                   * (angle < 1.0)
        done = ~not_done
        done = done[:, None]
        return done
    elif env_name == "Ant-v2":
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        x = next_obs[:, 0]
        not_done = np.isfinite(next_obs).all(axis=-1) \
                   * (x >= 0.2) \
                   * (x <= 1.0)

        done = ~not_done
        done = done[:, None]
        return done
    elif env_name == "InvertedPendulum-v2":
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        notdone = np.isfinite(next_obs).all(axis=-1) \
                  * (np.abs(next_obs[:, 1]) <= .2)
        done = ~notdone
        done = done[:, None]
        return done
    elif env_name == "InvertedDoublePendulum-v2":
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        sin1, cos1 = next_obs[:, 1], next_obs[:, 3]
        sin2, cos2 = next_obs[:, 2], next_obs[:, 4]
        theta_1 = np.arctan2(sin1, cos1)
        theta_2 = np.arctan2(sin2, cos2)
        y = 0.6 * (cos1 + np.cos(theta_1 + theta_2))

        done = y <= 1
        done = done[:, None]
        return done
    else:
        # HalfCheetah-v2 goes in here too
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
        return np.zeros((len(obs), 1), dtype=np.bool)


def get_predicted_states(model, state, action, env_name, deterministic=False):
    inputs = np.concatenate((state, action), axis=-1)

    ensemble_model_means, ensemble_model_vars = model.predict(inputs)
    ensemble_model_means[:, :, 1:] += state

    if deterministic:
        ensemble_samples = ensemble_model_means
    else:
        ensemble_model_stds = np.sqrt(ensemble_model_vars)
        ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

    num_models, batch_size, _ = ensemble_model_means.shape
    model_idxes = np.random.choice(model.elite_model_idxes, size=batch_size)
    batch_idxes = np.arange(0, batch_size)

    samples = ensemble_samples[model_idxes, batch_idxes]
    new_reward, new_next_state = samples[:, :1], samples[:, 1:]

    new_done = termination_fn(env_name, state, action, new_next_state)

    return new_reward, new_next_state, new_done


def get_env_params(env):
    obs = env.reset()

    params = {
        "obs": obs["observation"].shape[0],
        "action": env.action_space.shape[0],
        "action_max": env.action_space.high[0],
        "max_timesteps": env._max_episode_steps,
        "goal": obs["desired_goal"].shape[0],
        "action_space": env.action_space,
    }

    return params


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
