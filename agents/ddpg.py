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


DDPG_DEFAULT_PARAMS = {
    'env_id': "Pendulum-v1",
    'seed': 258,
    'replay_size': int(4e5),
    'polyak': 0.9995,
    'actor_critic_hidden_size': 256,
    'activation': 'relu',
    'update_batch_size': 256,
    'update_frequency': 10,
    'update_iterations': 1,
    'gamma': 0.9,
    'n_step': 3,
    'critic_lr':1e-4,
    'actor_lr': 1e-3,
    'critic_loss': 'hubber',
    'num_training_episodes': int(20e3),
    'exploration_noise_scale': 0.1,
    'warm_up_iters': 5000,
    'max_gradient_norm': 0.5,
    'num_test_episodes': 10,
    'evaluation_freq_episodes': 10,
    'normalize_observations': True,
    'enable_wandb_logging': True
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

class DDPG:

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
                 enable_wandb_logging: bool):
        

        self.__env_str = env_id
        self.__env = gym.make(self.__env_str)
        # Hyper_parameters much have hparam in the variable name.
        self.__hparam_seed = seed
        self.__hparam_polyak = polyak
        self.__hparam_actor_critic_hidden_size = actor_critic_hidden_size
        self.__hparam_activation = activation
        self.__hparam_update_batch_size = update_batch_size
        self.__hparam_gamma = gamma
        self.__hparam_update_frequency = update_frequency
        self.__hparam_update_iterations = update_iterations
        self.__hparam_n_step = n_step
        self.__hparam_critic_lr = critic_lr
        self.__hparam_actor_lr = actor_lr
        self.__hparam_max_gradient_norm = max_gradient_norm
        self.__hparam_exploration_noise_scale = exploration_noise_scale
        self.__hparam_n_training_episodes = num_training_episodes
        self.__hparam_warm_up_iters = warm_up_iters
        self.__hparam_critic_loss_fn = critic_loss
        self.__hparam_n_test_episodes = num_test_episodes
        self.__hparam_evaluation_freq_episodes = evaluation_freq_episodes
        self.__hparam_normalize_observation = normalize_observations
        self.__enable_wandb_logging = enable_wandb_logging

        observation_dims = self.env.observation_space.shape[0]
        action_dims = self.env.action_space.shape[0]
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_mean_test_reward = 0

        # Set the seed of the pseudo-random generators
        # (python, numpy, pytorch, gym, action_space)
        # Seed python RNG
        random.seed(seed)
        # Seed numpy RNG
        np.random.seed(seed)
        # seed the RNG for all devices (both CPU and CUDA)
        torch.manual_seed(seed)
        # Seed the env
        self.env.reset(seed=seed)
        # Seed the action sampler of the environment
        self.env.action_space.seed(seed)

    
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
    def critic_target(self):
        return self.__critic_tar
    
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
    def actor_target(self):
        return self.__actor_tar
    
    @property
    def device(self):
        return self.__device
    
    @property
    def gamma(self):
        return self.__hparam_gamma
    
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
    
    def normalize_observation(self, obs: np.array):
        low = self.env.observation_space.low
        high = self.env.observation_space.high
        scale_factor = high - low
        normalized_obs = np.divide(obs - low, scale_factor)
        assert normalized_obs.max() <= 1.0
        assert normalized_obs.min() >= 0.0
        return normalized_obs
        
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

            target = rewards + (self.__hparam_gamma**self.__hparam_n_step) * dones * Q_s.squeeze(dim=1)
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
            return critic_loss.item(), actor_loss.item(), returns_err.item(), returns.mean().item()
    
    def evaluate(self):

        episodic_cum_reward = []
        test_episode = gym.make(self.__env_str)
        for _ in range(self.__hparam_n_test_episodes):
            state, info = test_episode.reset()
            if self.__hparam_normalize_observation:
                state = self.normalize_observation(state)
            done = False
            cum_reward = 0

            while not done:

                # Get an action to execute
                action = self.get_action(torch.from_numpy(state).to(self.device),
                                         noise=0.0)
            
                # Perform the action in the environment
                next_state, reward, terminated, truncated, info = test_episode.step(action[0])
                if self.__hparam_normalize_observation:
                    next_state = self.normalize_observation(next_state)
                cum_reward += reward

                if terminated or truncated:
                    episodic_cum_reward.append(cum_reward)
                    done = True
                state = next_state
        
        episodic_cum_reward = np.array(episodic_cum_reward)
        return np.mean(episodic_cum_reward)

    def get_hyper_parameters(self):
        hparams = {}
        for name, value in self.__dict__.items():
            if "hparam" in name:
                param = name.replace(self.__class__.__name__, "").replace("hparam", "").replace("__", "")
                hparams[param] = value
        return hparams
    
    def __calculate_n_step_returns(self, 
                                   episode: list[tuple],
                                   n_step: int,
                                   gamma: float):
        """Calculates the n-step gamma discounted returns for each time step of
        the epsiode.
        This function also returns cumulative returns for each time step.

        Args:
            episode (list[tuple]): Tuple of transitions.
        """
        Transition = namedtuple('Transition', ['state', 
                                               'action', 
                                               'n_step_reward', 
                                               'n_step_next_state', 
                                               'terminated', 
                                               'returns'])
        

        assert n_step > 0, "n-step must be > 1."
        assert gamma > 0 and gamma <= 1 , "gamma must be between (0, 1]."

        cum_returns = 0
        n_step_reward = 0
        reverse_idx = 0
        last_element_index = len(episode) - 1
        buffer_transitions = []

        for item_idx, item in enumerate(reversed(episode)):
            reverse_idx += 1
            cum_returns = item[2] + gamma * cum_returns
            n_step_reward = item[2] + gamma * n_step_reward

            next_state = None
            if reverse_idx > n_step:
                n_step_reward -= (gamma**n_step) * episode[last_element_index][2]
                last_element_index -= 1
                next_state = episode[last_element_index][3]
                done = False              
            else:
                next_state = episode[-1][3]
                done = episode[-1][4]

            t = Transition(state=item[0],
                           action=item[1],
                           n_step_reward=n_step_reward,
                           n_step_next_state=next_state,
                           terminated=done,
                           returns=cum_returns)
            buffer_transitions.append(t)
        
        buffer_transitions.reverse()
        return buffer_transitions


    def learn(self,
              eval_callback: Optional[TrialEvaluationCallback] = None):
        
        if self.__enable_wandb_logging:
            # Create a writer object for logging
            todays_date = datetime.date.today()
            logger_title = "ddpg-" + self.__env.unwrapped.spec.id + "-" + str(todays_date).replace(":","-")
            writer = wandb.init(project=logger_title,
                                config=self.get_hyper_parameters())

        # Initialize variables
        total_steps_count = 0
        exploration_noise_scale = self.__hparam_exploration_noise_scale
        

        for episode in tqdm(range(0, self.__hparam_n_training_episodes)):            
            state, info = self.env.reset()
            if self.__hparam_normalize_observation:
                state = self.normalize_observation(state)
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
                if self.__hparam_normalize_observation:
                    next_state = self.normalize_observation(next_state)
                episode_sum_reward += reward

                t = (state, action[0], reward, next_state, terminated)
                
                epsiode_transitions.append(t)

                if truncated or terminated:    
                    done = True
                
                state = next_state

                # Update actor and critic
                if total_steps_count % self.__hparam_update_frequency == 0 \
                    and total_steps_count > self.__hparam_warm_up_iters:

                    critic_losses = []
                    actor_losses = []
                    returns_errs = []
                    returns_hist = []
                    for _ in range(self.__hparam_update_iterations):
                        critic_loss, actor_loss, returns_err, returns = self.train_step(batch_size=self.__hparam_update_batch_size)
                        critic_losses.append(critic_loss)
                        actor_losses.append(actor_loss)
                        returns_errs.append(returns_err)
                        returns_hist.append(returns)

                    critic_losses = np.array(critic_losses)
                    actor_losses = np.array(actor_losses)
                    returns_errs = np.array(returns_errs)
                    returns_hist = np.array(returns_hist)
                    
                    if self.__enable_wandb_logging:
                        writer.log({
                            "loss/critic": critic_losses.mean(),
                            "loss/actor": actor_losses.mean(),
                            "returns/estimation_err": returns_errs.mean(),
                            "returns/avg_returns": returns_hist.mean()
                        }, commit=False)

                    if not self.critic.training:
                        print("Critic is not in training mode")
                    if not self.actor.training:
                        print("Actor not in training mode")

                    # NOTE: DONOT use training env in evaluation.
                    # As the training env might be in different state

            
            # Calculate Returns and store in replay buffer
            
            buffer_transitions = self.__calculate_n_step_returns(epsiode_transitions,
                                                                  n_step=self.__hparam_n_step,
                                                                  gamma=self.gamma)

            # Add the episode to the replay buffer
            self.__exp_replay.add_epsiode(buffer_transitions)
            
            if self.__enable_wandb_logging:
                # Log current replay size
                writer.log({
                    "replay/size": self.__exp_replay.replay_size
                }, commit=False)
            
            # Evaluate agent performance
            
            if (episode + 1) % self.__hparam_evaluation_freq_episodes == 0:
                self.last_mean_test_reward = self.evaluate()
                
                if eval_callback is not None:
                    eval_callback.step(self.last_mean_test_reward)
                
                if self.__enable_wandb_logging:
                    writer.log({
                        "reward/mean_test_reward": self.last_mean_test_reward
                    }, commit=False)
                
            if self.__enable_wandb_logging:
                # Log metrics
                writer.log({"reward/episode_sum_reward": episode_sum_reward,
                            "exploration/noise_scale": exploration_noise_scale}, commit=True)
            
            #print("Episode: {} Train Cum Reward: {:.2f} Last Mean Test Cum Reward: {:.2f} Total Steps: {}.".format(episode+1, episode_sum_reward , self.last_mean_test_reward, total_steps_count))

        if self.__enable_wandb_logging:
            # Close writer
            writer.finish()