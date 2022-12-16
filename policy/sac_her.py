import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils.utils import soft_update, hard_update
from policy.sac_her_model import Actor, Critic
from policy.sac_her_model import GaussianPolicy, QNetwork, DeterministicPolicy


class SAC_(object):
    def __init__(self, env_params, args):
        self.env_params = env_params

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.critic = Critic(env_params).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = Critic(env_params).to(self.device)
        hard_update(self.critic_target, self.critic)

        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if args.automatic_entropy_tuning:
            if args.target_entropy is not None:
                self.target_entropy = args.target_entropy
            else:
                self.target_entropy = -torch.prod(torch.Tensor(env_params.action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
            self.alpha = self.log_alpha.cpu().exp().item()
        else:
            # Alpha
            self.alpha = args.alpha

        self.policy = Actor(env_params).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)


class SAC(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if args.automatic_entropy_tuning:
                if args.target_entropy is not None:
                    self.target_entropy = args.target_entropy
                else:
                    self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
                self.alpha = self.log_alpha.cpu().exp().item()
            else:
                # Alpha
                self.alpha = args.alpha

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(
                self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(
                self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target

        # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    def update_parameters_per(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch, weights_batch, indices = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device)
        weights_batch = torch.FloatTensor(weights_batch).to(self.device)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target

        # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss_element_wise = (qf1 - next_q_value).pow(2)
        qf2_loss_element_wise = (qf2 - next_q_value).pow(2)
        qf_loss_element_wise = qf1_loss_element_wise + qf2_loss_element_wise
        qf_loss = (qf_loss_element_wise * weights_batch).mean()
        qf1_loss = (qf1_loss_element_wise * weights_batch).detach().mean()
        qf2_loss = (qf2_loss_element_wise * weights_batch).detach().mean()

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss_element_wise = (self.alpha * log_pi) - min_qf_pi
        policy_loss = (policy_loss_element_wise * weights_batch).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        # PER: update priorities
        new_priorities = qf_loss_element_wise
        new_priorities += policy_loss_element_wise.pow(2)
        new_priorities += memory.priority_eps
        new_priorities = new_priorities.data.cpu().numpy().squeeze()
        memory.update_priorities(indices, new_priorities)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_checkpoint(self, env_name, ckpt_path, suffix=None, memory=None):
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        if suffix is None:
            file_name = "sac_ckpt_{}.zip".format(env_name)
        else:
            file_name = "sac_ckpt_{}_{}.zip".format(env_name, suffix)

        ckpt_path = os.path.join(ckpt_path, file_name)

        data = {
            "policy_state_dict": self.policy.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "critic_optimizer_state_dict": self.critic_optim.state_dict(),
            "policy_optimizer_state_dict": self.policy_optim.state_dict()
        }

        if hasattr(self, "alpha_optim"):
            data["alpha"] = self.alpha
            data["log_alpha"] = self.log_alpha
            data["alpha_optimizer_state_dict"] = self.alpha_optim.state_dict()

        if memory is not None:
            data["obs_mean"] = memory.o_norm.mean
            data["obs_std"] = memory.o_norm.std
            data["goal_mean"] = memory.g_norm.mean
            data["goal_std"] = memory.g_norm.std

        print("Saving models to {}".format(ckpt_path))
        torch.save(data, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False, load_norm_stats=False):
        print("Loading models from {}".format(ckpt_path))
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.policy.load_state_dict(ckpt["policy_state_dict"])
            self.critic.load_state_dict(ckpt["critic_state_dict"])
            self.critic_target.load_state_dict(ckpt["critic_target_state_dict"])
            self.critic_optim.load_state_dict(ckpt["critic_optimizer_state_dict"])
            self.policy_optim.load_state_dict(ckpt["policy_optimizer_state_dict"])

            if hasattr(self, "alpha_optim"):
                self.alpha = ckpt["alpha"]
                self.log_alpha = ckpt["log_alpha"]
                self.alpha_optim.load_state_dict(ckpt["alpha_optimizer_state_dict"])

            norm_stats = None
            if load_norm_stats:
                norm_stats = {
                    "obs_mean": ckpt["obs_mean"],
                    "obs_std": ckpt["obs_std"],
                    "goal_mean": ckpt["goal_mean"],
                    "goal_std": ckpt["goal_std"],
                }

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()

            return norm_stats
