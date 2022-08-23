import copy
import numpy as np
import torch
import torch.nn.functional as F
from policy.ddpg_model import Actor, Critic


class DDPG(object):
    def __init__(self, num_inputs, action_space, args):
        self.max_action = action_space.high[0]
        self.gamma = args.gamma
        self.tau = args.tau
        self.expl_noise = args.expl_noise

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.actor = Actor(num_inputs, action_space.shape[0], self.max_action).to(device=self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr_actor)

        self.critic = Critic(num_inputs, action_space.shape[0]).to(device=self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr_critic)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state).detach().cpu().numpy()
        if evaluate is False:
            action += np.random.normal(0, self.max_action * self.expl_noise, size=action.shape)
            action = np.clip(action, -self.max_action, self.max_action)
        return action

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            # Compute the target Q value
            target_Q = self.critic_target(next_state_batch, self.actor_target(next_state_batch))
            target_Q = reward_batch + mask_batch * self.gamma * target_Q

        # Get current Q estimate
        current_Q = self.critic(state_batch, action_batch)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item()

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
            # Compute the target Q value
            target_Q = self.critic_target(next_state_batch, self.actor_target(next_state_batch))
            target_Q = reward_batch + mask_batch * self.gamma * target_Q

        # Get current Q estimate
        current_Q = self.critic(state_batch, action_batch)

        # Compute critic loss
        critic_loss_element_wise = (current_Q - target_Q).pow(2)
        critic_loss = (critic_loss_element_wise * weights_batch).mean()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss_element_wise = -self.critic(state_batch, self.actor(state_batch))
        actor_loss = actor_loss_element_wise.mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # PER: update priorities
        new_priorities = critic_loss_element_wise
        new_priorities += actor_loss_element_wise.pow(2)
        new_priorities += memory.priority_eps
        new_priorities = new_priorities.data.cpu().numpy().squeeze()
        memory.update_priorities(indices, new_priorities)

        return critic_loss.item(), actor_loss.item()

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
