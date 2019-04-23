import numpy as np
import torch
import time

from unityagents import UnityEnvironment
from agent_td3 import MAT3DAgent

env_path = './data/Tennis_Linux/Tennis.x86'
actor0_strategy_path = "./results/run001/actor_0_net.pt"
actor1_strategy_path = "./results/run001/actor_1_net.pt"
number_episodes = 20
max_time = 999

actor_settings = dict(layer1_size=400, layer2_size=300, out_noise=3e-3)
critic_settings = dict(layer1_size=400, layer2_size=300, out_noise=3e-3)

setting = {
    'batch_size': 128,           # Number of experience samples per training step
    'buffer_size': int(3e6),     # Max number of samples in the replay memory
    'gamma': 0.99,               # Reward decay factor
    'tau': 1e-3,                 # Update rate for the slow update of the target networks
    'lr_actor': 5e-4,            # Actor learning rate
    'lr_critic': 5e-4,           # Critic learning rate
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
env = UnityEnvironment(file_name=env_path, seed=1)
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

agent = MAT3DAgent(env, brain, brain_name, device, setting)

agent.agents[0].load_nets(actor0_strategy_path)
agent.agents[0].set_action_noise(0.0)

agent.agents[1].load_nets(actor1_strategy_path)
agent.agents[1].set_action_noise(0.0)


for _ in range(number_episodes):
    env_info = env.reset(train_mode=False)[brain_name]
    while True:
        states = env_info.vector_observations
        actions = np.array([agent.agents[i].get_action(states[i, :]) for i in range(agent.number_agents)])
        actions_all = actions.flatten()
        env_info = env.step(actions_all)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        time.sleep(0.03)
        if np.any(dones):
            episode_rewards = []
            break
env.close()