import torch
import random
import datetime
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from gymnasium import Env
import torch.optim as optim
import torch.nn.functional as F


from replay import ReplayBuffer
from models import Critic, Actor
from torch.utils.tensorboard import SummaryWriter

DDPG_DEFAULT_DICT = {
    'seed': 1,
    'replay-size': 50000,
    'polyak': 0.995,
    'hidden-size': 256,
    'activation': 'relu',
    'mini-batch-size': 256,
    'update-frequency-iters': 500,
    'num-update-iters': 10,
    'gamma': 0.9,
    'critic-lr':1e-5,
    'actor-lr': 1e-5,
    'num-training-steps': 10e6,
    'exploration-noise-scale-start': 0.5,
    'exploration-noise-scale-final': 0.001,
    'num-test-episodes': 30,
    'training-warmup-iters': 10000,
    'max-gradient-norm': 1.0
}

class DDPG:

    def __init__(self,
                 env: Env,
                 hyperparameters_dict: dict):
        
        self.__env = env
        self.__hyperparameters = hyperparameters_dict

        observation_dims = self.__env.observation_space.shape[0]
        action_dims = self.__env.action_space.shape[0]
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set the seed of the pseudo-random generators
        # (python, numpy, pytorch, gym, action_space)
        # Seed python RNG
        random.seed(self.hyperparameters['seed'])
        # Seed numpy RNG
        np.random.seed(self.hyperparameters['seed'])
        # seed the RNG for all devices (both CPU and CUDA)
        torch.manual_seed(self.hyperparameters['seed'])

    
        # Experience Replay
        self.__exp_replay = ReplayBuffer(maxsize=
                                         self.__hyperparameters['replay-size'])

        # Critic Networks

        self.__critic = Critic(observation_dims=observation_dims,
                                action_dims=action_dims,
                                hidden_size=self.__hyperparameters['hidden-size'],
                                activation=self.__hyperparameters['activation']).to(self.device)
        
        self.__critic_tar = Critic(observation_dims=observation_dims,
                                    action_dims=action_dims,
                                    hidden_size=self.__hyperparameters['hidden-size'],
                                    activation=self.__hyperparameters['activation']).to(self.device)

        self.__critic_optimizer = optim.Adam(self.critic.parameters(),
                                             lr=self.hyperparameters['critic-lr'])
        
        # Actor Networks
        self.__actor = Actor(observation_dims=observation_dims,
                              action_dims=action_dims,
                              hidden_size=self.__hyperparameters['hidden-size'],
                              activation=self.__hyperparameters['activation']).to(self.device)

        self.__actor_tar = Actor(observation_dims=observation_dims,
                                 action_dims=action_dims,
                                 hidden_size=self.__hyperparameters['hidden-size'],
                                 activation=self.__hyperparameters['activation']).to(self.device)

        self.__actor_optimizer = optim.Adam(self.actor.parameters(),
                                            lr=self.hyperparameters['actor-lr'])

        # Copy weights
        self.critic.load_state_dict(self.__critic_tar.state_dict())
        self.actor.load_state_dict(self.__actor_tar.state_dict())


    @property
    def critic(self):
        return self.__critic
    
    @property
    def critic_optimizer(self):
        return self.__critic_optimizer

    @property
    def actor_optimizer(self):
        return self.__actor_optimizer

    @property
    def actor(self):
        return self.__actor
    
    @property
    def device(self):
        return self.__device

    @property
    def hyperparameters(self):
        return self.__hyperparameters
    
    def get_action(self,
                   states: torch.FloatTensor, 
                   noise: float=0.0 ):
        
        # Get the actions prediction from the actor network
        actions_torch = None
        with torch.no_grad():
            self.actor.eval()
            actions_torch = self.actor(states)
            self.actor.train()
        
        actions = actions_torch.numpy()
        
        # Add noise
        if noise > 0.0:
            actions += np.random.normal(scale=noise, size=actions_torch.size())
        
        # Clip the actions value to the max and min allowed.
        actions = np.clip(actions,
                          a_min=self.__env.action_space.low,
                          a_max=self.__env.action_space.high)
        return actions
            
    
    def critic_soft_update(self,
                           polyak:float):
        for param, target_param in zip(self.critic.parameters(), self.__critic_tar.parameters()):
            target_param.data.copy_( polyak * target_param.data + (1 - polyak) * param.data )
    
    def critic_hard_update(self,
                           polyak:float):
        for param, target_param in zip(self.critic.parameters(), self.__critic_tar.parameters()):
            target_param.data.copy_(param.data)  
        
    def actor_soft_update(self,
                          polyak: float):
        for param, target_param in zip(self.actor.parameters(), self.__actor_tar.parameters()):
            target_param.data.copy_( polyak * target_param.data + (1 - polyak) * param.data )

    def actor_hard_update(self,
                          polyak: float):
        for param, target_param in zip(self.actor.parameters(), self.__actor_tar.parameters()):
            target_param.data.copy_( param.data )
        
    def train_step(self, 
                   batch_size: int):
        
        # Sample a batch of transitions from the replay
        num_samples, samples = self.__exp_replay.sample(batch_size)

        if num_samples > 0:
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []
            for sample in samples:
                s, a, r, n_s, d = sample
                states.append(s)
                actions.append(a)
                rewards.append(r)
                next_states.append(n_s)
                dones.append(d)

            states = torch.Tensor(np.array(states)).to(self.device)
            actions = torch.Tensor(np.array(actions)).to(self.device)
            rewards = torch.Tensor(np.array(rewards)).to(self.device)
            next_states = torch.Tensor(np.array(next_states)).to(self.device)
            dones = torch.Tensor(1 - np.array(dones)).to(self.device)

            # Update critic network
            Q_s = None
            with torch.no_grad():
                self.critic.eval()
                Q_s = self.critic(next_states, self.actor(next_states))
                self.critic.train()
            target = rewards + self.hyperparameters['gamma'] * dones * Q_s.squeeze(dim=1)
            Q = self.critic(states, actions).squeeze(dim=1)
            critic_loss = F.mse_loss(Q, target)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            # Clip the gradients
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.hyperparameters['max-gradient-norm'])
            self.critic_optimizer.step()

            # Update Actor Network
            actor_loss = -self.critic(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # Clip the gradients
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.hyperparameters['max-gradient-norm'])
            self.actor_optimizer.step()

            # Update target network weight
            self.critic_soft_update(polyak=self.hyperparameters['polyak'])
            self.actor_soft_update(polyak=self.hyperparameters['polyak'])
            
            return critic_loss.item(), actor_loss.item()
    
    def evaluate(self):

        episodic_cum_reward = []
        
        for _ in range(self.hyperparameters['num-test-episodes']):

            state, info = env.reset()    
            done = False
            cum_reward = 0

            while not done:

                # Get an action to execute
                action = self.get_action(torch.from_numpy(state).to(self.device),
                                        noise=0.0)
            
                # Perform the action in the environment
                next_state, reward, terminated, truncated, info = self.__env.step(action[0])
                cum_reward += reward

                if terminated or truncated:
                    episodic_cum_reward.append(cum_reward)
                    done = True
                
                state = next_state
        
        episodic_cum_reward = np.array(episodic_cum_reward)
        return np.mean(episodic_cum_reward)

    def learn(self):

        # Create a writer object for logging
        curr_timestamp = datetime.datetime.now()
        writer = SummaryWriter("runs/ddpg/" + self.__env.unwrapped.spec.id + "/" + curr_timestamp.strftime("%d/%m/%Y %H:%M:%S"))

        # Initialize learning specific parameters here
        total_steps_count = 0
        curr_noise_scale = self.hyperparameters['exploration-noise-scale-start']
        episode_cum_reward = 0

        assert self.hyperparameters['exploration-noise-scale-start'] > self.hyperparameters['exploration-noise-scale-final']
        noise_decay_rate = (self.hyperparameters['exploration-noise-scale-start'] - self.hyperparameters['exploration-noise-scale-final'])/self.hyperparameters['num-training-steps']

        state, info = env.reset()
        state = np.expand_dims(state, axis=0)
        
        while total_steps_count < self.hyperparameters['num-training-steps']:
            
            total_steps_count += 1
            
            # Get an action to execute
            action = self.get_action(torch.from_numpy(state).to(self.device),
                                     noise=curr_noise_scale)
            
            # Perform the action in the environment
            next_state, reward, terminated, truncated, info = self.__env.step(action[0])
            next_state = np.expand_dims(next_state, axis=0)

            # Save the sample in the experience replay
            self.__exp_replay.add((state[0], action[0], reward, next_state[0], terminated))

            if terminated or truncated:

                writer.add_scalar("reward/epsiode_cum_reward", episode_cum_reward, total_steps_count)
                writer.add_scalar("exploration/noise_scale", curr_noise_scale, total_steps_count)

                episode_cum_reward = 0
                state,info = env.reset()
                state = np.expand_dims(state, axis=0)
            else:
                state = next_state

            episode_cum_reward += reward
            curr_noise_scale -= noise_decay_rate

            # Update Actor and Critic
            if total_steps_count % self.hyperparameters['update-frequency-iters'] == 0 and \
                self.hyperparameters['training-warmup-iters'] < total_steps_count:
                critic_losses = []
                actor_losses = []
                for _ in range(self.hyperparameters['num-update-iters']):
                    critic_loss, actor_loss = self.train_step(self.hyperparameters['mini-batch-size'])
                    critic_losses.append(critic_loss)
                    actor_losses.append(actor_loss)

                writer.add_scalar("loss/critic", np.array(critic_losses).mean(), total_steps_count)
                writer.add_scalar("loss/actor", np.array(actor_losses).mean(), total_steps_count)



if __name__ == "__main__":

    env = gym.make("Pendulum-v1")
    #env = gym.make("BipedalWalker-v3", hardcore=True)
    agent = DDPG(env,
                 DDPG_DEFAULT_DICT)
    
    agent.learn()
