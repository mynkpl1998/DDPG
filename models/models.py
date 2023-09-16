import torch
import torch.nn as nn
import numpy as np
from models.base import BaseModel
from gymnasium.spaces import Box

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class SimpleCritic(BaseModel):

    def __init__(self,
                 observation_type: Box,
                 action_type: Box,
                 hidden_size: int,
                 **kwargs):
        
        super(SimpleCritic, self).__init__(observation_type=observation_type,
                                           action_type=action_type)
        
        self.__observation_dims = observation_type.shape[0]
        self.__action_dims = action_type.shape[0]

        self.fc1 = nn.Linear(out_features=hidden_size, in_features=self.__observation_dims + self.__action_dims)
        self.fc2 = nn.Linear(out_features=hidden_size, in_features=hidden_size)
        self.fc3 = nn.Linear(out_features=1, in_features=hidden_size)
        self.activation = nn.ReLU()
        self.init_weights(init_w=3e-3)
        
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    @property
    def observation_dims(self):
        return self.__observation_dims

    @property
    def action_dims(self):
        return self.__action_dims
    
    def forward(self,
                states: torch.FloatTensor,
                actions: torch.FloatTensor):
        
        x = torch.cat((states, actions), 1)
        x = self.fc3(self.activation(self.fc2(self.activation(self.fc1(x)))))
        return x

class SimpleActor(BaseModel):

    def __init__(self,
                 observation_type: Box,
                 action_type: Box,
                 hidden_size: int,
                 **kwargs):
        
        super(SimpleActor, self).__init__(observation_type,
                                          action_type)

        self.__observation_dims = observation_type.shape[0]
        self.__action_dims = action_type.shape[0]

        # Action Lower Bound
        self._action_lower_bound = torch.FloatTensor(action_type.low)
        self._action_upper_bound = torch.FloatTensor(action_type.high)
        assert torch.equal(-self._action_lower_bound, self._action_upper_bound)
        
        self.fc1 = nn.Linear(out_features=hidden_size, in_features=observation_type.shape[0])
        self.fc2 = nn.Linear(out_features=hidden_size, in_features=hidden_size)
        self.fc3 = nn.Linear(out_features=action_type.shape[0], in_features=hidden_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.activation = nn.ReLU()
        self.init_weights(init_w=3e-3)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)        
    
    @property
    def observation_dims(self):
        return self.__observation_dims

    @property
    def action_dims(self):
        return self.__action_dims
    
    def forward(self,
                states: torch.FloatTensor):
        
        if self._action_upper_bound.device != states.device:
            self._action_upper_bound = self._action_upper_bound.device()
        
        x = states.view(-1, self.__observation_dims)
        x = self.fc3(self.activation(self.fc2(self.activation(self.fc1(x)))))
        # Let the model learn the interpolation function from [-1, 1] to [action_high, action_low]
        x = self.tanh(x) * self._action_upper_bound
        return x



if __name__ == "__main__":

    # Test data
    obs_dims = 256
    action_dims = 4
    hidden_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Testing with {} device".format(device))

    random_obs = torch.rand(200, obs_dims).to(device)
    random_act = torch.rand(200, action_dims).to(device)
    

    # Test Critic
    critic = SimpleCritic(observation_dims=obs_dims,
                          action_dims=action_dims,
                          hidden_size=hidden_size,
                          activation='relu').to(device)

    out = critic(random_obs, random_act)
    

    # Test actor
    actor = SimpleActor(observation_dims=obs_dims,
                        action_dims=action_dims,
                        hidden_size=hidden_size,
                        activation='tanh').to(device)
    out = actor(random_obs)

