import random
from collections import deque

class ReplayBuffer:

    def __init__(self, 
                 maxsize: int):
        self.__maxsize = maxsize
        self.__buffer = deque(maxlen=maxsize)

    @property
    def maxsize(self):
        return self.__maxsize

    @property
    def replay_size(self):
        return len(self.__buffer)

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.__buffer))
        return batch_size, random.sample(self.__buffer, batch_size)
    
    def add(self, tup):
        self.__buffer.append(tup)