import optuna
from typing import Literal, Dict, Any, Optional

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

from replay import ReplayBuffer
from models import Critic, Actor
import wandb

from utils import TrialEvaluationCallback

from agents.base import BaseAgent


DDPG_DEFAULT_PARAMS = {
    'env_id': "Pendulum-v1",
    'seed': 258,
    'replay_size': int(4e5),
    'polyak': 0.9995,
    'actor_critic_hidden_size': 256,
    'activation': 'relu',
    'update_batch_size': 256,
    'update_frequency': int(100),
    'update_iterations': 5,
    'gamma': 0.9,
    'n_step': 10,
    'critic_lr':1e-4,
    'actor_lr': 1e-3,
    'critic_loss': 'hubber',
    'num_training_episodes': int(5000),
    'exploration_noise_scale': 0.1,
    'warm_up_iters': 10000,
    'max_gradient_norm': 0.5,
    'num_test_episodes': 10,
    'evaluation_freq_episodes': 10,
    'normalize_observations': True,
    'enable_wandb_logging': True,
    'logger_title': 'test_logger'
}

def sample_ddpg_params(op_trial: optuna.Trial) -> Dict[str, Any]:

    """Sampler for DDPG hyperparameters."""
    replay_size = op_trial.suggest_int("replay_size", int(10e3), int(10e5))
    polyak = 1 - op_trial.suggest_float("polyak", 0.00001, 0.1),
    actor_critic_hidden_size = op_trial.suggest_int("actor_critic_hidden_size", 32, 512)
    activation = op_trial.suggest_categorical("activation", ['relu', 'tanh'])
    update_batch_size = op_trial.suggest_int('update_batch_size', 64, 256)
    update_frequency = op_trial.suggest_int('update_frequency', 1, int(20e3))
    update_iterations = op_trial.suggest_int('update_iterations', 1, 20)
    gamma = 1 - op_trial.suggest_float("gamma", 0.00001, 0.1)
    critic_lr = op_trial.suggest_float("critic_lr", 1e-6, 1e-3)
    actor_lr = op_trial.suggest_float("actor_lr", 1e-6, 1e-3)
    critic_loss = op_trial.suggest_categorical("critic_loss", ['hubber', 'mse'])
    exploration_noise_scale = op_trial.suggest_float("exploration_noise_scale", 0.01, 0.3)
    warm_up_iters = op_trial.suggest_int("warm_up_iters", int(1e3), int(10e3))
    max_gradient_norm = op_trial.suggest_float("max_gradient_norm", 0.5, 5.0)

    return {
        'replay_size': replay_size,
        'polyak': polyak,
        'actor_critic_hidden_size': actor_critic_hidden_size,
        'activation': activation,
        'update_batch_size': update_batch_size,
        'update_frequency': update_frequency,
        'update_iterations': update_iterations,
        'gamma': gamma,
        'critic_lr': critic_lr,
        'actor_lr': actor_lr,
        'critic_loss': critic_loss,
        'exploration_noise_scale': exploration_noise_scale,
        'warm_up_iters': warm_up_iters,
        'max_gradient_norm': max_gradient_norm
    }

class DDPG(BaseAgent):

    def __init__(self,
                 env_id: str,
                 seed: int,
                 replay_size: int,
                 polyak: float,
                 actor_critic_hidden_size: int,
                 critic_loss: Literal['mse', 'hubber'],
                 activation: Literal['tanh', 'relu'],
                 update_batch_size: int,
                 gamma: float,
                 n_step: int,
                 update_frequency: int,
                 update_iterations: int,
                 critic_lr: float,
                 actor_lr: float,
                 max_gradient_norm: float,
                 exploration_noise_scale: float,
                 num_training_episodes: int,
                 warm_up_iters: int,
                 num_test_episodes: int,
                 evaluation_freq_episodes: int,
                 normalize_observations: bool,
                 enable_wandb_logging: bool, 
                 logger_title: str | None = None):
        
        super().__init__(env_id, 
                         seed, 
                         gamma, 
                         n_step, 
                         num_training_episodes, 
                         num_test_episodes,
                         evaluation_freq_episodes,
                         normalize_observations,
                         enable_wandb_logging,
                         logger_title)

        # Hyper_parameters much have hparam in the variable name.
        self.__hparam_polyak = polyak
        self.__hparam_actor_critic_hidden_size = actor_critic_hidden_size
        self.__hparam_activation = activation
        self.__hparam_update_batch_size = update_batch_size
        self.__hparam_update_frequency = update_frequency
        self.__hparam_update_iterations = update_iterations
        self.__hparam_critic_lr = critic_lr
        self.__hparam_actor_lr = actor_lr
        self.__hparam_max_gradient_norm = max_gradient_norm
        self.__hparam_exploration_noise_scale = exploration_noise_scale
        self.__hparam_warm_up_iters = warm_up_iters
        self.__hparam_critic_loss_fn = critic_loss
        

        # Update the hyper-parameters in wandb config dict
        if self.is_wandb_logging_enabled:
            hyper_params = self.get_hyper_parameters()
            for param in hyper_params:
                if param not in self.writer.config.keys():
                    self.writer.config[param] = hyper_params[param]
        
        # Register metrics to log in wandb
        if self.is_wandb_logging_enabled:
            self.set_wandb_logging_metrics()

        # Experience Replay
        self.__exp_replay = ReplayBuffer(maxsize=replay_size)

        # Critic Networks

        self.__critic = Critic(observation_dims=self.env.observation_space.shape[0],
                                action_dims=self.env.action_space.shape[0],
                                hidden_size=actor_critic_hidden_size,
                                activation=activation).to(self.device)
        
        self.__critic_target = Critic(observation_dims=self.env.observation_space.shape[0],
                                       action_dims=self.env.action_space.shape[0],
                                       hidden_size=actor_critic_hidden_size,
                                       activation=activation).to(self.device)

        self.__critic_optimizer = optim.Adam(self.critic.parameters(),
                                             lr=critic_lr)
        
        # Actor Networks
        self.__actor = Actor(observation_dims=self.env.observation_space.shape[0],
                              action_dims=self.env.action_space.shape[0],
                              hidden_size=actor_critic_hidden_size,
                              activation=activation).to(self.device)

        self.__actor_target = Actor(observation_dims=self.env.observation_space.shape[0],
                                     action_dims=self.env.action_space.shape[0],
                                     hidden_size=actor_critic_hidden_size,
                                     activation=activation).to(self.device)

        self.__actor_optimizer = optim.Adam(self.actor.parameters(),
                                            lr=actor_lr)

        # Initialize target and primary weights to same values.
        self.critic.load_state_dict(self.critic_target.state_dict())
        self.actor.load_state_dict(self.actor_target.state_dict())
        
    @property
    def critic(self):
        return self.__critic
    
    @property
    def critic_target(self):
        return self.__critic_target
    
    @property
    def actor(self):
        return self.__actor

    @property
    def actor_target(self):
        return self.__actor_target
    
    @property
    def critic_optimizer(self):
        return self.__critic_optimizer

    @property
    def actor_optimizer(self):
        return self.__actor_optimizer
    
    def critic_soft_update(self,
                           polyak:float):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_( polyak * target_param.data + (1 - polyak) * param.data )
    
    def critic_hard_update(self,
                           polyak:float):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(param.data)  
        
    def actor_soft_update(self,
                          polyak: float):
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_( polyak * target_param.data + (1 - polyak) * param.data )

    def actor_hard_update(self,
                          polyak: float):
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_( param.data )
    
    def set_wandb_logging_metrics(self) -> None:
        
        super().set_wandb_logging_metrics()
        # define which metrics will be plotted against it
        wandb.define_metric("replay/size", step_metric="episode")
        wandb.define_metric("loss/actor", step_metric="step")
        wandb.define_metric("loss/critic", step_metric="step")
        wandb.define_metric("returns/avg_err_in_estimate", step_metric="step")
        wandb.define_metric("returns/avg_returns", step_metric="step")

    def learn_episode_callback(self, episode: int, cum_reward: float, episode_length: int, n_step_transition_tuple: list) -> None:
        
        # Add the episode to the replay buffer
        self.__exp_replay.add_epsiode(n_step_transition_tuple)
            
        # log the current replay size
        if self.is_wandb_logging_enabled:
            # Log current replay size
                self.writer.log({
                    "replay/size": self.__exp_replay.replay_size
                }, commit=False)
    
    def __train_step(self, batch_size: int):

        critic_losses = []
        actor_losses = []
        returns_errs = []
        returns_hist = []

        for _ in range(0, self.__hparam_update_iterations):

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
                    rewards.append(sample.n_step_reward)
                    next_states.append(sample.n_step_next_state)
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
                self.critic_target.eval()
                self.actor_target.eval()
                with torch.no_grad():
                    Q_s = self.critic_target(next_states, self.actor_target(next_states))
                self.critic_target.train()
                self.actor_target.train()

                target = rewards + (self.gamma**self.n_step) * dones * Q_s.squeeze(dim=1)
                Q = self.critic(states, actions).squeeze(dim=1)

                if self.__hparam_critic_loss_fn == 'mse':
                    critic_loss = F.mse_loss(Q, target)
                else:
                    critic_loss = F.huber_loss(Q, target, delta=1.0)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                # Clip the gradients
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.__hparam_max_gradient_norm)
                # Update gradients
                self.critic_optimizer.step()

                # ------------------ Update Actor Network -------------------- #
                actor_loss = -self.critic(states, self.actor(states)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                # Clip the gradients
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.__hparam_max_gradient_norm)
                # Update gradients
                self.actor_optimizer.step()
                
                # Update target network weight
                self.critic_soft_update(polyak=self.__hparam_polyak)
                self.actor_soft_update(polyak=self.__hparam_polyak)

                returns_err = None
                with torch.no_grad():
                    returns_err = (returns - Q).abs().mean()
                
                
                critic_losses.append(critic_loss.item())
                actor_losses.append(actor_loss.item())
                returns_errs.append(returns_err.item())
                returns_hist.append(returns.mean().item())
        
        critic_losses = np.array(critic_losses)
        actor_losses = np.array(actor_losses)
        returns_errs = np.array(returns_errs)
        returns_hist = np.array(returns_hist)
        
        return critic_loss.mean(), actor_losses.mean(), returns_errs.mean(), returns_hist.mean() 


    def learn_step_callback(self, 
                              step: int,
                              transition_tuple: tuple) -> None:
        """Step callback. Called at every step.
        """
        if step % self.__hparam_update_frequency == 0 \
            and step > self.__hparam_warm_up_iters:
            critic_loss, actor_loss, err_in_est, avg_returns = self.__train_step(batch_size=self.__hparam_update_batch_size)

            if self.is_wandb_logging_enabled:
                self.writer.log({
                    "loss/critic": critic_loss,
                    "loss/actor": actor_loss,
                    "returns/avg_err_in_estimate": err_in_est,
                    "returns/avg_returns": avg_returns
                }, commit=False)

    def get_action(self,
                   state: np.array,
                   mode: Literal['train', 'eval'] = 'train') -> np.array:
        
        # Get the actions prediction from the actor network
        actions_torch = None
        
        with torch.no_grad():
            self.actor.eval()
            actions_torch = self.actor(torch.from_numpy(state).to(self.device))
            self.actor.train()
        actions = actions_torch.cpu().numpy()
        
        # Add noise if we are in training mode only
        if mode == 'train' \
            and self.__hparam_exploration_noise_scale > 0.0:
            actions += np.random.normal(scale=self.__hparam_exploration_noise_scale,
                                         size=actions.shape)
        
        # Clip the actions value to the max and min allowed.
        actions = np.clip(actions,
                          a_min=self.env.action_space.low,
                          a_max=self.env.action_space.high)
        return actions
    