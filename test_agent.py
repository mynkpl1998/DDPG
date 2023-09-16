import torch
import gymnasium as gym
from models.models import SimpleCritic, SimpleActor

if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    obs_space = env.observation_space
    act_space = env.action_space
    
    print(obs_space.shape[0])
    m = SimpleCritic(observation_type=obs_space,
                     action_type=act_space,
                     hidden_size=256)
    

    a = SimpleActor(observation_type=obs_space,
                    action_type=act_space,
                    hidden_size=256)
    
    obs, info = env.reset()
    obs = torch.from_numpy(obs)
    a.forward(obs)
    
    