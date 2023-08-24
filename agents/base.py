import torch
import random
import datetime
import numpy as np
from tqdm import tqdm
import gymnasium as gym
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple

import wandb
from typing import Literal, Dict, Any, Optional

BASE_AGENT_DEFAULT_PARAMS = {
    'env_id': "BipedalWalker-v3",
    'seed': 258,
    'gamma': 0.9,
    'n_step': 1,
    'num_training_episodes': int(20e3),
    'num_test_episodes': 10,
    'evaluation_freq_episodes': 10,
    'normalize_observations': True,
    'enable_wandb_logging': False,
    'logger_title': 'test_logger'
}

class BaseAgent:

    def __init__(self,
                 env_id: str,
                 seed: int,
                 gamma: float,
                 n_step: int,
                 num_training_episodes: int,
                 num_test_episodes: int,
                 evaluation_freq_episodes: int,
                 normalize_observations: bool,
                 enable_wandb_logging: bool,
                 logger_title: Optional[str] = None):
        
        self.__env_str = env_id
        self.__env = gym.make(self.__env_str)
        # Hyper_parameters much have hparam in the variable name.
        self.__hparam_seed = seed
        self.__hparam_gamma = gamma
        self.__hparam_n_step = n_step
        self.__hparam_normalize_observation = normalize_observations
        self.__hparam_n_test_episodes = num_test_episodes
        self.__enable_wandb_logging = enable_wandb_logging
        self.__hparam_n_training_episodes = num_training_episodes
        self.__hparam_evaluation_freq_episodes = evaluation_freq_episodes
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set the seed of the pseudo-random generators
        # (python, numpy, pytorch, gym, action_space)
        # Seed python RNG
        random.seed(self.__hparam_seed)
        # Seed numpy RNG
        np.random.seed(self.__hparam_seed)
        # seed the RNG for all devices (both CPU and CUDA)
        torch.manual_seed(self.__hparam_seed)
        # Seed the env
        self.env.reset(seed=self.__hparam_seed)
        # Seed the action sampler of the environment
        self.env.action_space.seed(self.__hparam_seed)

        # Set up wandb Logging
        
        if self.is_wandb_logging_enabled:
            
            if logger_title is None:
                raise ValueError("Invalid logger title None.")
            
            todays_date = datetime.datetime.now()
            self.__wandb_writer = None
            run_name = self.__class__.__name__ + " " + self.__env.unwrapped.spec.id + "-" + str(todays_date).replace(":","-")
            
            self.__wandb_writer = wandb.init(project=logger_title,
                                             config=self.get_hyper_parameters(),
                                             name=run_name)

    def __del__(self):
        """
        if self.is_wandb_logging_enabled \
            and self.writer is not None:
            # Close writer
            wandb.finish()
        """
        pass

    @property
    def env_id(self) -> str:
        return self.__env_str
    
    @property
    def env(self) -> gym.Env:
        return self.__env
    
    @property
    def device(self):
        return self.__device
    
    @property
    def writer(self):
        return self.__wandb_writer
    
    @property
    def gamma(self):
        return self.__hparam_gamma
    
    @property
    def n_step(self):
        return self.__hparam_n_step

    @property
    def is_wandb_logging_enabled(self,) -> bool:
        return self.__enable_wandb_logging
    
    def get_hyper_parameters(self):
        hparams = {}
        for name, value in self.__dict__.items():
            if "hparam" in name:
                for base in self.__class__.__bases__:
                    name = name.replace(base.__name__, "")
                param = name.replace(self.__class__.__name__, "").replace("hparam", "").replace("__", "")
                hparams[param] = value
        return hparams

    def normalize_observation(self, obs: np.array):
        low = self.env.observation_space.low
        high = self.env.observation_space.high
        scale_factor = high - low
        normalized_obs = np.divide(obs - low, scale_factor)
        
        if normalized_obs.max() > 1.0 \
            and np.abs(1 - normalized_obs.max()) > 0.1:
            raise RuntimeError("Detected Normalized vector having value greater than one.")

        if normalized_obs.min() < 0.0 \
            and np.abs(normalized_obs.min()) > 0.1:
            raise RuntimeError("Detected Normalized vector having value less than zero.")
        return normalized_obs
    
    def get_action(self,
                   state: np.array,
                   mode: Literal['train', 'eval'] = 'train') -> np.array:
        raise NotImplementedError
    
    def set_wandb_logging_metrics(self) -> None:
        """Defines how the logging metrics are plotted.
        """
        
        # define our custom x axis metrics
        wandb.define_metric("episode")
        wandb.define_metric("step")

        # define which metrics will be plotted against it
        wandb.define_metric("reward/*", step_metric="episode")
        wandb.define_metric("episode_length/*", step_metric="episode")

    
    def learn_step_callback(self, 
                            step: int,
                            transition_tuple: tuple) -> None:
        """Step callback. Called on every step.
        """
        pass

    def learn_episode_callback(self,
                                episode: int,
                                cum_reward: float,
                                episode_length: int,
                                n_step_transition_tuple: list[namedtuple]) -> None:
        """Epsiode callback. Called after every epsiode.
        """
        pass

    def learn_evaluate_callback(self) -> (float, float):
        """Evaluation callback. Called at every evaluation step.
        """
        eval_episode_reward = []
        eval_episode_length = []
        test_env = gym.make(self.__env_str)
        
        for _ in range(self.__hparam_n_test_episodes):
            
            state, info = test_env.reset()
            if self.__hparam_normalize_observation:
                state = self.normalize_observation(state)

            done = False
            cum_reward = 0
            ep_length = 0

            while not done:

                ep_length += 1

                # Get an action to execute
                action = self.get_action(state,
                                         mode='eval')
                
                # Perform the action in the environment
                next_state, reward, terminated, truncated, info = test_env.step(action[0])
                if self.__hparam_normalize_observation:
                    next_state = self.normalize_observation(next_state)
                cum_reward += reward

                if terminated or truncated:
                    eval_episode_reward.append(cum_reward)
                    eval_episode_length.append(ep_length)
                    done = True
                state = next_state
        
        eval_episode_reward = np.array(eval_episode_reward)
        eval_episode_length = np.array(eval_episode_length)
        return np.mean(eval_episode_reward), np.mean(eval_episode_length)

    def __calculate_n_step_returns(self, 
                                   episode: list[tuple],
                                   n_step: int,
                                   gamma: float):
        """Calculates the n-step gamma discounted returns for each time step of
        the epsiode.
        This function also returns cumulative returns for each time step.

        Args:
            episode (list[tuple]): Tuple of transitions.
            n_step (int): number of future steps to consider.
            gamma (float)[0-1]: Discount factor
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


    def learn(self):
        
        # Initialize variables
        total_steps_count = 0

        for episode in tqdm(range(0, self.__hparam_n_training_episodes)):
            
            state, info = self.env.reset()
            if self.__hparam_normalize_observation:
                state = self.normalize_observation(state)
            
            episode_sum_reward = 0
            episode_length = 0
            done = False
            epsiode_transitions = []

            while not done:
                
                total_steps_count += 1
                episode_length += 1

                # Log the current step count
                if self.is_wandb_logging_enabled:
                    self.writer.log({
                        "step": total_steps_count
                    }, commit=False)

                # Get an action to execute
                action = self.get_action(state,
                                         mode='train')

                # Perform the action in the environment
                next_state, reward, terminated, truncated, info = self.env.step(action[0])
                episode_sum_reward += reward

                if self.__hparam_normalize_observation:
                    next_state = self.normalize_observation(next_state)

                # Transition tuple
                t = (state, action[0], reward, next_state, terminated)


                self.learn_step_callback(step=total_steps_count,
                                           transition_tuple=t)
                epsiode_transitions.append(t)

                if truncated or terminated:  
                    done = True
                
                state = next_state
        
            #print("Episode: {} Train Cum Reward: {:.2f} Last Mean Test Cum Reward: Total Steps: {}.".format(episode+1, episode_sum_reward, total_steps_count))
            
            # Calculate n-step returns from the transitions
            buffer_transitions = self.__calculate_n_step_returns(episode=epsiode_transitions,
                                                                 n_step=self.__hparam_n_step,
                                                                 gamma=self.__hparam_gamma)
            self.learn_episode_callback(episode + 1, 
                                          episode_sum_reward,
                                          episode_length,
                                          buffer_transitions)
            
            # Evaluate agent performance
            if (episode + 1) % self.__hparam_evaluation_freq_episodes == 0:
                eval_mean_reward, eval_mean_ep_length = self.learn_evaluate_callback()
                
                if self.is_wandb_logging_enabled:
                    self.writer.log({
                        "reward/eval": eval_mean_reward,
                        "episode_length/eval": eval_mean_ep_length
                    }, commit=False)

            if self.is_wandb_logging_enabled:
                self.writer.log({"reward/train": episode_sum_reward,
                                 "episode_length/train": episode_length,
                                 "epsiode": episode}, commit=True)