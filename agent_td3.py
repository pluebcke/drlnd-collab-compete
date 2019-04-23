from collections import namedtuple
import numpy as np

import torch
import torch.nn.functional as functional
import torch.optim as optim

from model_td3 import Actor, Critic
from memory import ReplayMemory

Experience = namedtuple('Experience', 'states actions rewards last_states dones')

class MAT3DAgent():
    def __init__(self, env, brain, brain_name, device, settings):

        # Set variables regarding the environment
        self.env = env
        self.brain_name = brain_name
        self.device = device
        action_size = brain.vector_action_space_size
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        state_size = states.shape[1]

        self.action_size = action_size
        self.state_size = state_size
        self.batch_size = settings['batch_size']
        self.number_agents = len(env_info.agents)

        settings['action_size'] = action_size
        settings['state_size'] = state_size
        # Initialize the agents
        self.agents = [T3DAgent(device, settings) for _ in range(self.number_agents)]

        # Save some of the settings into class member variables
        self.pretrain_steps = settings['pretrain_steps']
        self.gamma = settings['gamma']
        self.tau = settings['tau']

        self.action_noise = settings['action_noise']
        self.action_noise_decay = settings['action_noise_decay']
        self.action_noise_min = settings['action_noise_min']
        self.action_clip = settings['action_clip']
        self.target_action_noise = settings['target_action_noise']
        self.target_noise_clip = settings['target_noise_clip']
        self.optimize_every = settings['optimize_actor_every']

        # Initialize replay memory and episode generator
        self.memory = ReplayMemory(device, settings['buffer_size'])
        self.generator = self.play_episode()

        self.number_steps = 0
        return

    def set_action_noise(self, std):
        for agent in self.agents:
            agent.set_action_noise(std)
        return

    def pretrain(self):
        # In the pretrain method a number of steps, given by self.pretrain_steps, is played.
        # The agents are following a random policy and only the learn step for the critic function is performed
        # The idea of using a pretrain phase before starting regular episodes
        # is from https://github.com/whiterabbitobj/Continuous_Control/
        print("Random sampling of " + str(self.pretrain_steps) + " steps")
        env = self.env
        brain_name = self.brain_name
        env_info = env.reset(train_mode=True)[brain_name]
        number_agents = env_info.vector_observations.shape[0]
        for step in range(self.pretrain_steps):
            actions = [np.random.uniform(-1, 1, self.action_size) for _ in range(number_agents)]
            states = env_info.vector_observations
            actions = np.array(actions)
            env_info = env.step(actions.flatten())[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            if rewards[0] > 0 or rewards[1] > 0 or step%5 == 0:
                rewards = np.array(rewards)
                self.memory.add(Experience(states, actions, rewards, next_states, dones))
            if np.any(dones):
                env_info = env.reset(train_mode=True)[brain_name]

            if step%5 == 0:
                self.learn(True)

    def play_episode(self):
        # The idea of generating episodes in an "experience generator" is taken from
        # "Deep Reinforcement Learning Hands-On" by Maxim Lapan

        print("Starting episode generator")
        # Initialize the environment
        env = self.env
        brain_name = self.brain_name
        env_info = env.reset(train_mode=True)[brain_name]
        # Initialize episode_rewards and get the first state
        episode_rewards = []
        # Run episode step by step
        while True:
            states = env_info.vector_observations
            # get actions from different agents
            actions = np.array([self.agents[i].get_action(states[i, :]) for i in range(self.number_agents)])
            env_info = env.step(actions.flatten())[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            episode_rewards.append(rewards)
            rewards = np.array(rewards)
            self.memory.add(Experience(states, actions, rewards, next_states, dones))
            if np.any(dones):
                reward_sum = np.sum(episode_rewards, axis = 0)
                agent_reward = np.max(reward_sum)
                std_reward = np.std(agent_reward)
                mean_reward = np.mean(agent_reward)
                episode_rewards = []
                env_info = env.reset(train_mode=True)[brain_name]
                # decrease noise
                self.action_noise = max(self.action_noise * self.action_noise_decay, self.action_noise_min)
                self.set_action_noise(self.action_noise)
                yield mean_reward, std_reward
            else:
                yield -1, -1

    def take_step(self):
        return next(self.generator)

    def learn(self, is_pre_train = False):
        self.number_steps += 1
        if self.memory.number_samples() <= self.batch_size:
            return -1, -1, -1

        critic_loss_a = 0
        critic_loss_b = 0
        actor_loss = 0

        for i in range(self.number_agents):
            # Get experience from replay memory
            experiences = self.memory.sample_batch(self.batch_size)

            # Convert list of experiences into pytorch tensors for batch evaluation
            s0 = []
            a0 = []
            s1 = []
            r = []
            d = []

            for j, exp in enumerate(experiences):
                s0.append(exp.states)
                a0.append(exp.actions.flatten())
                s1.append(exp.last_states)
                r.append(exp.rewards)
                d.append(exp.dones)

            obs0 = torch.from_numpy(np.array(s0).swapaxes(0,1)).float().to(self.device)
            obs1 = torch.from_numpy(np.array(s1).swapaxes(0,1)).float().to(self.device)

            s0 = np.array(s0).reshape((self.batch_size, -1))
            s0 = torch.from_numpy(s0).float().to(self.device)
            s1 = np.array(s1).reshape((self.batch_size, -1))
            s1 = torch.from_numpy(np.array(s1)).float().to(self.device)
            a0 = torch.from_numpy(np.array(a0)).float().to(self.device)

            #  The ideas to predict the actions for all agents in a0_pred (instead of using the values from a0 and
            #  only predict the action for the current agent) is from:
            #  https://github.com/whiterabbitobj/Collaborate_Compete
            #  The combination of torch.cat with list comprehensions is something I first noticed in this implementation,
            #  eventhough I've later seen it at other places as well
            a0_pred = torch.cat([self.agents[i].actor_local(obs0[i,:,:]) for i in range(self.number_agents)], dim=1)
            a1_pred = torch.cat([self.agents[i].actor_target(obs1[i,:,:]) for i in range(self.number_agents)], dim=1)

            r_i = torch.from_numpy(np.array(r)[:, i]).float().to(self.device).unsqueeze(-1)
            d_i = torch.from_numpy(np.array(d).astype(np.float)[:, i]).float().to(self.device).unsqueeze(-1)
            cla_i, clb_i = self.agents[i].optimize_critic(s0, a0, r_i, s1, a1_pred, d_i)
            if self.number_steps % self.optimize_every == 0 and not is_pre_train:
                ali = self.agents[i].optimize_actor(s0, a0_pred)
                actor_loss += ali

            critic_loss_a += cla_i
            critic_loss_b += clb_i

        return actor_loss, critic_loss_a, critic_loss_b


    def save_nets(self, model_save_path):
        for i in range(self.number_agents):
            self.agents[i].save_nets(model_save_path, i)
        return

class T3DAgent:
    def __init__(self, device, settings):
        self.device = device
        action_size = settings['action_size']
        state_size = settings['state_size']
        self.action_size = action_size
        self.state_size = state_size

        # Initialize actor local and target networks
        self.actor_local = Actor(state_size, action_size, settings['actor_settings']).to(device)
        self.actor_target = Actor(state_size, action_size, settings['actor_settings']).to(device)
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=settings['lr_actor'])

        # Initialize critic networks
        self.critic_local = Critic(2*state_size, 2*action_size, settings['critic_settings']).to(device)
        self.critic_target = Critic(2*state_size, 2*action_size, settings['critic_settings']).to(device)
        self.critic_target.load_state_dict(self.critic_local.state_dict())
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=settings['lr_critic'])

        # Save some of the settings into class member variables
        self.gamma = settings['gamma']
        self.tau = settings['tau']

        self.action_noise = settings['action_noise']
        self.action_clip = settings['action_clip']
        self.target_action_noise = settings['target_action_noise']
        self.target_noise_clip = settings['target_noise_clip']
        self.optimize_every = settings['optimize_actor_every']

        self.number_steps = 0
        return

    def set_action_noise(self, std):
        self.action_noise = std
        return

    def get_action(self, states):
        with torch.no_grad():
            states_t = torch.from_numpy(states).type(torch.FloatTensor).unsqueeze(0).to(self.device)
            actions = self.actor_local.get_action(states_t).cpu().detach().numpy()
            actions += self.action_noise * np.random.normal(size=actions.shape)
            actions = np.clip(actions, -self.action_clip, self.action_clip)
        return actions

    def optimize_actor(self, s0_all, a0_all_pred):
        # Calc policy loss
        actor_loss = -self.critic_local.get_qa(s0_all, a0_all_pred).mean()
        # Update actor nn
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # slow update
        self.slow_update(self.tau)
        return -actor_loss.cpu().detach().numpy()

    def optimize_critic(self, s0_all, a0_all, r, s1_all, a1_all, d):
        # The ideas of adding noise to the next state a1 as well as the critic loss that takes q1_expected and
        # q2_expected as arguments at the same time are from the implementation of the authors of the TD3 manuscript
        # at https://github.com/sfujim/TD3/
        with torch.no_grad():
            noise = torch.randn_like(a1_all).to(self.device)
            noise = noise * torch.tensor(self.target_action_noise).expand_as(noise).to(self.device)
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            a1_all = (a1_all + noise).clamp(-self.action_clip, self.action_clip)
            qa_target, qb_target = self.critic_target(s1_all, a1_all)
            q_target = torch.min(qa_target, qb_target)
            q_target = r + self.gamma * (1.0 - d) * q_target
        qa_expected, qb_expected = self.critic_local(s0_all, a0_all)

        critic_loss_a = functional.mse_loss(qa_expected, q_target)
        critic_loss_b = functional.mse_loss(qb_expected, q_target)
        critic_loss = critic_loss_a + critic_loss_b
        # Update critic nn
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        return critic_loss_a.cpu().detach().numpy(), critic_loss_b.cpu().detach().numpy()

    def slow_update(self, tau):
        for target_par, local_par in zip(self.actor_target.parameters(), self.actor_local.parameters()):
            target_par.data.copy_(tau * local_par.data + (1.0 - tau) * target_par.data)
        for target_par, local_par in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            target_par.data.copy_(tau * local_par.data + (1.0 - tau) * target_par.data)
        return

    def load_nets(self, actor_file_path, critic_file_path = None):
        self.actor_local.load_state_dict(torch.load(actor_file_path))
        self.actor_local.eval()
        if critic_file_path:
            self.critic_local.load_state_dict(torch.load(critic_file_path))
            self.critic_local.eval()
        return

    def save_nets(self, model_save_path, agent_number):
        actor_path = model_save_path + "actor_" + str(agent_number) + "_net.pt"
        torch.save(self.actor_local.state_dict(), actor_path)
        critic_path = model_save_path + "critic_"+ str(agent_number) + "_net.pt"
        torch.save(self.critic_local.state_dict(), critic_path)
        return
