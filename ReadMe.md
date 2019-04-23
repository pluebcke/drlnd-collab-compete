# drlnd-collab-compete
My implementation of the Collaboration and Competition Project in the scope of Udacity's Deep Reinforcement Learning Nanodegree

The task was to train two agents to play tennis together using the PyTorch package (https://pytorch.org/).
A description of the results can be found in the Report.ipynb jupyter notebook.

# Installation
In order to run the training script you'll need a a working Python environment with a couple of different non-standard libraries.
One simple way to get everything set-up correctly is:
- Install Python, for example using Anaconda from https://www.anaconda.com/
- Follow the instructions from the Deep Reinforcement Learning Repository: https://github.com/udacity/deep-reinforcement-learning
to install the base packages needed.
- You'll also need to download a pre-compiled Unity Environment, links to versions for different operating systems can be found under "Getting started" at
https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet
While I only tested this approach on Ubuntu Linux, it should work on Windows or Mac OS in the same way.

# Description of the Tennis Environment
The goal is to train two agents to play tennis, e.g., keep a tennis ball in the air as long as possible. 
An animation of the task together with a detailed description of the environment can be found at https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet.

According to Udacity's [https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet](description), the agents have an observation space with 8 dimensions, describing
the position and location of the ball and racket. Since three consecutive observations are stacked together (in order to include some dynamics) the observation space of each agent has actually 24 dimensions.
The action space consists of two continuos actions that control the movement in the vertical and horizontal direction.
Each time an agent hits a ball and it passes the net, it recives a reward of +0.1, when the ball hits the ground the agent on that side receives a negative reward of -0.1.
In order to maximize the score, the agents need to learn to serve the ball in a way, that the agent on the other side can return it again. At the end of each episode each agent gets a cumulative reward, and we assign the maximum of these two scores as the total score for this episode.
The task is solved, when the average reward over 100 episodes is above 0.5. 


# Running the Agent
To run the agent you just need to run the "main.py" file. You might need to update the path of the Unity environment in line 11.
The settings are given starting from line 18, the comments behind each line give a brief description of the parameters.
