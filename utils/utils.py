import numpy as np
import math
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


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


def get_predicted_states_her(model, state, ag, g, action, env_params, deterministic=False):
    inputs = np.concatenate((state, ag, g, action), axis=-1)

    ensemble_model_means, ensemble_model_vars = model.predict(inputs)
    ensemble_model_means += np.concatenate((state, ag, g), axis=-1)

    if deterministic:
        ensemble_samples = ensemble_model_means
    else:
        ensemble_model_stds = np.sqrt(ensemble_model_vars)
        ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

    num_models, batch_size, _ = ensemble_model_means.shape
    model_idxes = np.random.choice(model.elite_model_idxes, size=batch_size)
    batch_idxes = np.arange(0, batch_size)

    samples = ensemble_samples[model_idxes, batch_idxes]
    new_next_state, new_next_state_ag = samples[:, :env_params["obs"]], samples[:, env_params["obs"]:env_params["obs"] + env_params["goal"]]

    return new_next_state, new_next_state_ag


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


class TensorMinibatchKMeans:
    def __init__(self, eps=1e-3, **kwargs):
        self._labels = None
        self._cluster_centers = None
        self._cluster_centers_std = None

        self._dimensions = None
        self._cluster_centers_sum = None
        self._cluster_centers_count = None

        self._kmeans = KMeans(**kwargs)

        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        self._eps = torch.tensor(eps ** 2, dtype=torch.float, device=self._device)

    @property
    def n_clusters(self):
        return self._kmeans.n_clusters

    @property
    def cluster_centers_(self):
        return self._cluster_centers.detach().cpu().numpy()

    @property
    def cluster_centers_std_(self):
        return self._cluster_centers_std.detach().cpu().numpy()

    @property
    def labels_(self):
        return self._labels.detach().cpu().numpy().astype(int)

    def fit(self, X):
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        self._kmeans = self._kmeans.fit(X)

        self._labels = torch.tensor(self._kmeans.labels_, dtype=torch.int, device=self._device)
        self._cluster_centers = torch.tensor(self._kmeans.cluster_centers_, dtype=torch.float, device=self._device)

        self._dimensions = self._cluster_centers.shape[1]
        self._cluster_centers_sum = torch.zeros_like(self._cluster_centers, dtype=torch.float, device=self._device)
        self._cluster_centers_sum_sq = torch.zeros_like(self._cluster_centers, dtype=torch.float, device=self._device)
        self._cluster_centers_count = torch.zeros_like(self._cluster_centers, dtype=torch.float, device=self._device)

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float, device=self._device)
        for n in range(len(self._labels)):
            ones = torch.ones(size=(self._dimensions,), dtype=torch.int, device=self._device)
            self._cluster_centers_count[self._labels[n]] += ones
            self._cluster_centers_sum[self._labels[n]] += X[n]
            self._cluster_centers_sum_sq[self._labels[n]] += X[n] ** 2

        squared_mean = self._cluster_centers_sum_sq / self._cluster_centers_count
        max_or_eps = torch.max(self._eps, squared_mean - torch.square(self._cluster_centers))
        self._cluster_centers_std = torch.pow(max_or_eps, 0.5)

    def partial_fit(self, X):
        if self._labels is None:
            self.fit(X)
            return self

        # Assign to cluster centers
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float, device=self._device)
        dist = torch.cdist(X, self._cluster_centers, p=2)
        _, idx = torch.min(dist, dim=-1)

        # Update cluster centers
        for n in range(len(idx)):
            ones = torch.ones(size=(self._dimensions,), dtype=torch.int, device=self._device)
            self._cluster_centers_count[idx[n]] += ones
            self._cluster_centers_sum[idx[n]] += X[n]
            self._cluster_centers_sum_sq[idx[n]] += X[n] ** 2

        # Update properties
        # self._labels = torch.cat((self._labels, idx), dim=-1)
        self._labels = idx
        self._cluster_centers = self._cluster_centers_sum / self._cluster_centers_count

        squared_mean = self._cluster_centers_sum_sq / self._cluster_centers_count
        max_or_eps = torch.max(self._eps, squared_mean - torch.square(self._cluster_centers))
        self._cluster_centers_std = torch.pow(max_or_eps, 0.5)

        return self

    def predict(self, X, batch_size=20000):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float, device=self._device)
        size = len(X)
        if size > batch_size:
            idx = torch.empty(size=(size,), dtype=torch.int, device=self._device)
            for start_pos in range(0, size, batch_size):
                X_ = X[start_pos:start_pos + batch_size]
                dist = torch.cdist(X_, self._cluster_centers, p=2)
                _, idx[start_pos:start_pos + batch_size] = torch.min(dist, dim=-1)
        else:
            dist = torch.cdist(X, self._cluster_centers, p=2)
            _, idx = torch.min(dist, dim=-1)
        return idx
