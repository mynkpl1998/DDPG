import random
from collections import deque
from typing import NamedTuple, Iterator
import gymnasium as gym
from collections import namedtuple

class ReplayBuffer:

    def __init__(self, 
                 maxsize: int):
        self.__maxsize = maxsize
        self.__buffer = deque(maxlen=maxsize)
        self.__Transition = namedtuple('Transition', ['state', 'action', 'next_state', 'reward', 'done', 'returns'])
    
    @property
    def maxsize(self):
        return self.__maxsize

    @property
    def replay_size(self):
        return len(self.__buffer)

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.__buffer))
        return batch_size, random.sample(self.__buffer, batch_size)
    
    def add_epsiode(self,
                    transitions: Iterator[NamedTuple]):
        for transition in transitions:
            self.__buffer.append(transition)
        
    def add(self, 
            transition: NamedTuple):
        self.__buffer.append(transition)


if __name__ == "__main__":
    env = gym.make("Pendulum-v1")

    Transition = ReplayBuffer.Transition()

    state, info = env.reset()
    done = False
    episode_data = []
    while not done:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            done = True
        
        transition = Transition(state,
                                action,
                                next_state,
                                reward,
                                done,
                                None)
        episode_data.append(transition)

    buffer.add(episode_data, 0)
    
        
       
        