from collections import deque
import time

import numpy as np
import torch

from unityagents import UnityEnvironment
from agent_td3 import MAT3DAgent
from tools import create_save_path, save_results

#  Paths: Set the path for the unity environment in the variable env_path
#  base_save_path is the main folder in which results will be stored
env_path = './data/Tennis_Linux_NoVis/Tennis.x86_64'
base_save_path = './results/'

#  Initializing settings for the agents
actor_settings = dict(layer1_size=512, layer2_size=256, out_noise=3e-3)
critic_settings = dict(layer1_size=512, layer2_size=256, out_noise=3e-3)

settings1 = {
    'batch_size': 256,           # Number of experience samples per training step
    'buffer_size': int(3e6),     # Max number of samples in the replay memory
    'gamma': 0.99,               # Reward decay factor
    'tau': 2e-3,                 # Update rate for the slow update of the target networks
    'lr_actor': 5e-4,            # Actor learning rate
    'lr_critic': 1e-3,           # Critic learning rate
    'action_noise': 1.0,         # Noise added during episodes played
    'action_noise_decay': 0.999,   # Decay of action noise
    'action_noise_min': 0.15,     # minimum value of action noise
    'action_clip': 1.0,          # Actions are clipped to +/- action_clip
    'target_action_noise': 0.6,  # Noise added during the critic update step
    'target_noise_clip': 0.3,    # Noise clip for the critic update step
    'number_steps': 1,           # Number of steps for roll-out, currently not used
    'optimize_actor_every': 2,   # Update the actor only every X update steps
    'pretrain_steps': int(10000), # Number of random actions played before training starts
    'actor_settings': actor_settings,
    'critic_settings': critic_settings}

#  settings2 are much more agressive setting, following the settings that were presented in
#  https://github.com/tommytracey/DeepRL-P3-Collaboration-Competition
#  and https://github.com/gtg162y/DRLND/tree/master/P3_Collab_Compete
settings2 = {
    'batch_size': 256,           # Number of experience samples per training step
    'buffer_size': int(3e6),     # Max number of samples in the replay memory
    'gamma': 0.99,               # Reward decay factor
    'tau': 2e-3,                 # Update rate for the slow update of the target networks
    'lr_actor': 5e-4,            # Actor learning rate
    'lr_critic': 1e-3,           # Critic learning rate
    'action_noise': 1.0,         # Noise added during episodes played
    'action_noise_decay': 0.999,   # Decay of action noise
    'action_noise_min': 0.15,     # minimum value of action noise
    'action_clip': 1.0,          # Actions are clipped to +/- action_clip
    'target_action_noise': 0.6,  # Noise added during the critic update step
    'target_noise_clip': 0.3,    # Noise clip for the critic update step
    'number_steps': 1,           # Number of steps for roll-out, currently not used
    'optimize_actor_every': 2,   # Update the actor only every X update steps
    'pretrain_steps': int(10000), # Number of random actions played before training starts
    'actor_settings': actor_settings,
    'critic_settings': critic_settings}

#  several pairs of settings can be evaluated in one run of the program,
#  make sure that all settings are in the "settings" list below
settings = [settings1, settings2]

# Start the environment
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
env = UnityEnvironment(file_name=env_path, seed=2)
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

#  Run the training for various settings
for setting in settings:
    # Create folder to save this run
    save_path = create_save_path(base_save_path)
    # create new agent
    agent = MAT3DAgent(env, brain, brain_name, device, setting)

    # Setting up lists and dequeues for logging results
    rewards = []
    c_a_loss = []
    c_b_loss = []
    a_loss = []
    avg_rewards = []
    reward_mean = deque(maxlen=100)
    episode = 0
    step = 0

    actor_loss_queue = deque(maxlen=10)
    critic_a_loss_queue = deque(maxlen=10)
    critic_b_loss_queue = deque(maxlen=10)

    # Take random steps before training starts
    agent.pretrain()

    #Start training the agent
    startTime = time.time()
    while True:
        # Taking a step in the environment and save results in the ReplayBuffer
        reward, std_reward = agent.take_step()
        step = step + 1

        for _ in range(5):
            # Improve the policies
            actor_loss, critic_a_loss, critic_b_loss = agent.learn()
            # Log results
            if actor_loss != -1:
                actor_loss_queue.append(actor_loss)
                critic_a_loss_queue.append(critic_a_loss)
                critic_b_loss_queue.append(critic_b_loss)

        if reward != - 1:
            # If the episode is over log results and print status messages
            reward_mean.append(reward)
            rewards.append(reward)
            a_loss.append(np.mean(actor_loss_queue))
            c_a_loss.append(np.mean(critic_a_loss_queue))
            c_b_loss.append(np.mean(critic_b_loss_queue))

            avg_rewards.append(np.mean(reward_mean))
            episodeTime = time.time() - startTime
            startTime = time.time()
            print("\rEps: " + str(episode) + " rew: " + str(reward) + " std: " + str(std_reward) + " AvgRew:" + str(
                np.mean(reward_mean)) + " noise: " + str(agent.action_noise) +  " crcloss: " + str((np.mean(critic_a_loss_queue) + np.mean(critic_b_loss_queue))/2) + " actor loss: " + str(np.mean(actor_loss_queue)) + " tim: " + str(episodeTime), end="")

            episode += 1

            # Save results every 50 episodes
            if episode%50 == 0:
                save_results(save_path, agent, setting, rewards, avg_rewards, a_loss, c_a_loss, c_b_loss)
            # After 3000 episodes, or if the mean reward is above 1.00, save results and stop training
            if episode == 3000 or np.mean(reward_mean) > 1.00:
                save_results(save_path, agent, setting, rewards, avg_rewards, a_loss, c_a_loss, c_b_loss)
                break




