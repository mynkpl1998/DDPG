import torch
import torch.nn as nn
from gymnasium.spaces import Box

class BaseModel(nn.Module):

    def __init__(self,
                 observation_type: Box,
                 action_type: Box):
        super(BaseModel, self).__init__()
    
    @property
    def actor(self,):
        raise NotImplementedError()

    @property
    def critic(self):
        raise NotImplementedError()