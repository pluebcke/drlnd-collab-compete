from collections import deque
import random

import numpy as np
import torch

class ReplayMemory:
    """ The replay memory stores tuples of state, action, reward, next action and a flag if the episode is done
        The sample method takes batch_size as an input argument and returns this many samples from the buffer
    """
    def __init__(self, device, maxLen):
        self.data = deque(maxlen=maxLen)
        self.device = device
        self.range = np.arange(0, maxLen, 1)
        return

    def add(self, sample):
        self.data.append(sample)
        return

    def sample_batch(self, batch_size):
        experiences = random.sample(self.data, k=batch_size)
        return experiences

    def number_samples(self):
        return len(self.data)
