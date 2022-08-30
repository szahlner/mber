import numpy as np
import math
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


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


def get_neighboring_states(memory, state, action, env_name, n_neighbors=7):
    # Z-space
    position = len(memory)
    buffer_state, buffer_action = memory.buffer["state"][:position], memory.buffer["action"][:position]

    z_space_buffer = np.concatenate((buffer_state, buffer_action), axis=-1)
    scaler = StandardScaler(with_mean=False)
    z_space_norm_buffer = scaler.fit_transform(z_space_buffer)

    # NearestNeighbors - object
    k_nn = NearestNeighbors(n_neighbors=n_neighbors).fit(z_space_norm_buffer)

    # Query NN for rollout
    z_space = np.concatenate((state, action), axis=-1)
    z_space_norm = scaler.transform(z_space)
    nn_indices = k_nn.kneighbors(z_space_norm, return_distance=False)

    # Remove itself, shuffle and chose
    nn_state = memory.buffer["state"][nn_indices]
    nn_next_state = memory.buffer["next_state"][nn_indices]
    nn_reward = memory.buffer["reward"][nn_indices]

    nn_delta_state = nn_next_state - nn_state
    nn_delta_state = np.mean(nn_delta_state, axis=1)

    next_state = state + nn_delta_state
    reward = np.mean(nn_reward, axis=1)
    done = termination_fn(env_name, state, action, next_state)

    return reward, next_state, done
