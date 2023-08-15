import torch
import random
import datetime
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from gymnasium import Env
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple

from typing import Literal
from replay import ReplayBuffer
from models import Critic, Actor
from torch.utils.tensorboard import SummaryWriter

DDPG_DEFAULT_DICT = {
    'env': gym.make("Pendulum-v1"),
    'seed': 258,
    'replay_size': int(1e5),
    'polyak': 0.995,
    'actor_critic_hidden_size': 256,
    'activation': 'relu',
    'update_batch_size': 256,
    'update_frequency': 500,
    'update_iterations': 1,
    'gamma': 0.9,
    'critic_lr':1e-4,
    'actor_lr': 1e-3,
    'critic_loss': 'hubber',
    'num_training_episodes': int(50e3),
    'exploration_noise_scale': 0.1,
    'warm_up_iters': 5000,
#    'exploration-noise-scale-final': 0.001,
#    'num_test_episodes': 30,
#    'training_warmup_iters': 10000,
    'max_gradient_norm': 0.5
}

class DDPG:

    def __init__(self,
                 env: Env,
                 seed: int,
                 replay_size: int,
                 polyak: float,
                 actor_critic_hidden_size: int,
                 critic_loss: Literal['mse', 'hubber'],
                 activation: Literal['tanh', 'relu'],
                 update_batch_size: int,
                 gamma: float,
                 update_frequency: int,
                 update_iterations: int,
                 critic_lr: float,
                 actor_lr: float,
                 max_gradient_norm: float,
                 exploration_noise_scale: float,
                 num_training_episodes: int,
                 warm_up_iters: int):
        
        self.__env = env
        self.__seed = seed
        self.__polyak = polyak
        self.__actor_critic_hidden_size = actor_critic_hidden_size
        self.__activation = activation
        self.__update_batch_size = update_batch_size
        self.__gamma = gamma
        self.__update_frequency = update_frequency
        self.__update_iterations = update_iterations
        self.__critic_lr = critic_lr
        self.__actor_lr = actor_lr
        self.__max_gradient_norm = max_gradient_norm
        self.__exploration_noise_scale = exploration_noise_scale
        self.__n_training_episodes = num_training_episodes
        self.__warm_up_iters = warm_up_iters
        self.__critic_loss_fn = critic_loss

        observation_dims = self.env.observation_space.shape[0]
        action_dims = self.env.action_space.shape[0]
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set the seed of the pseudo-random generators
        # (python, numpy, pytorch, gym, action_space)
        # Seed python RNG
        random.seed(seed)
        # Seed numpy RNG
        np.random.seed(seed)
        # seed the RNG for all devices (both CPU and CUDA)
        torch.manual_seed(seed)
        # Seed the env
        env.reset(seed=seed)
        # Seed the action sampler of the environment
        env.action_space.seed(seed)

    
        # Experience Replay
        self.__exp_replay = ReplayBuffer(maxsize=replay_size)

        # Critic Networks

        self.__critic = Critic(observation_dims=observation_dims,
                                action_dims=action_dims,
                                hidden_size=actor_critic_hidden_size,
                                activation=activation).to(self.device)
        
        self.__critic_tar = Critic(observation_dims=observation_dims,
                                    action_dims=action_dims,
                                    hidden_size=actor_critic_hidden_size,
                                    activation=activation).to(self.device)

        self.__critic_optimizer = optim.Adam(self.critic.parameters(),
                                             lr=critic_lr)
        
        # Actor Networks
        self.__actor = Actor(observation_dims=observation_dims,
                              action_dims=action_dims,
                              hidden_size=actor_critic_hidden_size,
                              activation=activation).to(self.device)

        self.__actor_tar = Actor(observation_dims=observation_dims,
                                 action_dims=action_dims,
                                 hidden_size=actor_critic_hidden_size,
                                 activation=activation).to(self.device)

        self.__actor_optimizer = optim.Adam(self.actor.parameters(),
                                            lr=actor_lr)

        # Copy weights
        self.critic.load_state_dict(self.__critic_tar.state_dict())
        self.actor.load_state_dict(self.__actor_tar.state_dict())


    @property
    def env(self):
        return self.__env

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
    def gamma(self):
        return self.__gamma
    
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
            returns = []
            for sample in samples:
                states.append(sample.state)
                actions.append(sample.action)
                rewards.append(sample.reward)
                next_states.append(sample.next_state)
                dones.append(sample.terminated)
                returns.append(sample.returns)

            states = torch.Tensor(np.array(states)).to(self.device)
            actions = torch.Tensor(np.array(actions)).to(self.device)
            rewards = torch.Tensor(np.array(rewards)).to(self.device)
            next_states = torch.Tensor(np.array(next_states)).to(self.device)
            dones = torch.Tensor(1 - np.array(dones)).to(self.device)
            returns = torch.Tensor(np.array(returns)).to(self.device)

            # ------------------ Update Critic Network -------------------- #

            # Calculate target 
            Q_s = None
            with torch.no_grad():
                self.critic.eval()
                Q_s = self.critic(next_states, self.actor(next_states))
                self.critic.train()
            target = rewards + self.__gamma * dones * Q_s.squeeze(dim=1)
            Q = self.critic(states, actions).squeeze(dim=1)
            
            if self.__critic_loss_fn == 'mse':
                critic_loss = F.mse_loss(Q, target)
            else:
                critic_loss = F.huber_loss(Q, target, delta=1.0)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            # Clip the gradients
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.__max_gradient_norm)
            # Update gradients
            self.critic_optimizer.step()

            # ------------------ Update Actor Network -------------------- #
            actor_loss = -self.critic(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # Clip the gradients
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.__max_gradient_norm)
            self.actor_optimizer.step()

            # Update target network weight
            self.critic_soft_update(polyak=self.__polyak)
            self.actor_soft_update(polyak=self.__polyak)
            
            returns_err = None
            with torch.no_grad():
                returns_err = (returns - Q).abs().mean()
            return critic_loss.item(), actor_loss.item(), returns_err.item(), returns.mean().item()
    
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

    def learn2(self,
               experiment_name: str ):
        
        # Create a writer object for logging
        curr_timestamp = datetime.datetime.now()
        writer = SummaryWriter("runs/ddpg/" + self.__env.unwrapped.spec.id + "/" + experiment_name)

        # Log hyper-parameters
        #writer.add_hparams({'critic_loss_fn': self.__critic_loss_fn})

        # Initialize variables
        total_steps_count = 0
        exploration_noise_scale = self.__exploration_noise_scale
        Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'terminated', 'returns'])

        for episode in range(0, self.__n_training_episodes):            
            state, info = self.env.reset()
            episode_sum_reward = 0
            done = False
            epsiode_transitions = []

            while not done:

                total_steps_count += 1

                # Get an action to execute
                action = self.get_action(torch.from_numpy(state).to(self.device), 
                                         noise=exploration_noise_scale)
                
                # Perform the action in the environment
                next_state, reward, terminated, truncated, info = self.__env.step(action[0])
                episode_sum_reward += reward

                t = (state, action[0], reward, next_state, terminated)
                
                epsiode_transitions.append(t)

                if truncated or terminated:
                    done = True
                
                state = next_state

                # Update actor and critic
                if total_steps_count % self.__update_frequency \
                    and total_steps_count > self.__warm_up_iters:
                    critic_losses = []
                    actor_losses = []
                    returns_errs = []
                    returns_hist = []
                    for _ in range(self.__update_iterations):
                        critic_loss, actor_loss, returns_err, returns = self.train_step(batch_size=self.__update_batch_size)
                        critic_losses.append(critic_loss)
                        actor_losses.append(actor_loss)
                        returns_errs.append(returns_err)
                        returns_hist.append(returns)

                    critic_losses = np.array(critic_losses)
                    actor_losses = np.array(actor_losses)
                    returns_errs = np.array(returns_errs)
                    returns_hist = np.array(returns_hist)
                    writer.add_scalar("loss/critic", critic_losses.mean(), total_steps_count)
                    writer.add_scalar("loss/actor", actor_losses.mean(), total_steps_count)
                    writer.add_scalar("returns/estimation_err", returns_errs.mean(), total_steps_count)
                    writer.add_scalar("returns/avg_returns", returns_hist.mean(), total_steps_count)
                   

            # Calculate Returns and store in replay buffer
            returns = 0
            buffer_transitions = []
            for item_idx, item in enumerate(reversed(epsiode_transitions)):
                returns = item[2] + self.gamma * returns
                t = Transition(state=item[0],
                               action=item[1],
                               reward=item[2],
                               next_state=item[3],
                               terminated=item[4],
                               returns=returns)
                buffer_transitions.append(t)
            
            buffer_transitions.reverse()
            
            # Add the episode to the replay buffer
            self.__exp_replay.add_epsiode(buffer_transitions)
            # Log current replay size
            writer.add_scalar("replay/size", self.__exp_replay.replay_size, episode+1)
            

            # Log metrics
            writer.add_scalar("reward/episode_sum_reward", episode_sum_reward, episode+1)
            writer.add_scalar("exploration/noise_scale", exploration_noise_scale, episode+1)            
            print("Episode: {} Cum Reward: {:.2f} Total Steps: {}.".format(episode+1, episode_sum_reward ,total_steps_count))
            


    def learn(self,
              experiment_name: str,):

        # Create a writer object for logging
        curr_timestamp = datetime.datetime.now()
        writer = SummaryWriter("runs/ddpg/" + self.__env.unwrapped.spec.id + "/" + experiment_name)

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

    agent = DDPG(**DDPG_DEFAULT_DICT)
    
    agent.learn2(experiment_name="reduce_grad_norm_0.5_50k_episodes")
