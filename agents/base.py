import os
import torch
import random
import importlib
import datetime
import numpy as np
from tqdm import tqdm
import gymnasium as gym
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple

import wandb
from typing import Literal, Dict, Any, Optional, NamedTuple

BASE_AGENT_DEFAULT_PARAMS = {
    'env_id': "BipedalWalker-v3",
    'seed': 258,
    'gamma': 0.9,
    'n_step': 1,
    'num_training_episodes': int(20e3),
    'num_test_episodes': 10,
    'warm_up_iters': 10000,
    'evaluation_freq_episodes': 10,
    'normalize_observations': True,
    'enable_wandb_logging': False,
    'logger_title': 'test_logger',
    'exploration_noise_type': 'NormalNoise',
    'exploration_noise_params': {'NormalNoise': {'mu': 0.0, 'sigma': 0.3}}
}

class BaseAgent:

    def __init__(self,
                 env_id: str,
                 seed: int,
                 gamma: float,
                 n_step: int,
                 num_training_episodes: int,
                 num_test_episodes: int,
                 warm_up_iters: int,
                 evaluation_freq_episodes: int,
                 normalize_observations: bool,
                 enable_wandb_logging: bool,
                 exploration_noise_type: Literal['NormalNoise'],
                 exploration_noise_params: dict,
                 logger_title: Optional[str] = None,
                 ):
        # Hyper_parameters much have hparam in the variable name.
        self._hparam_seed = seed
        self.__env_str = env_id
        self.__env = gym.make(self.__env_str)
        
        # Set the seed of the pseudo-random generators
        # (python, numpy, pytorch, gym, action_space)
        # Seed python RNG
        random.seed(self._hparam_seed)
        # Seed numpy RNG
        np.random.seed(self._hparam_seed)
        # seed the RNG for all devices (both CPU and CUDA)
        torch.manual_seed(self._hparam_seed)
        # Seed the env
        self.env.reset(seed=self._hparam_seed)
        # Seed the action sampler of the environment
        self.env.action_space.seed(self._hparam_seed)

        self._hparam_gamma = gamma
        self._hparam_n_step = n_step
        self._hparam_normalize_observations = normalize_observations
        self._hparam_num_test_episodes = num_test_episodes
        self._enable_wandb_logging = enable_wandb_logging
        self._hparam_num_training_episodes = num_training_episodes
        self._hparam_evaluation_freq_episodes = evaluation_freq_episodes
        self._hparam_warm_up_iters = warm_up_iters
        self._hparam_exploration_noise_type = exploration_noise_type
        noise_params = exploration_noise_params[exploration_noise_type]
        for param in noise_params:
            setattr(self, '_hparam_exploration_noise_' + param, noise_params[param])
        module = importlib.import_module("utils.noise")
        self._action_noise = getattr(module, exploration_noise_type)(**noise_params)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._max_mean_test_reward = -float("inf")

        # Set up wandb Logging
        if self.is_wandb_logging_enabled:
            
            if logger_title is None:
                raise ValueError("Invalid logger title None.")
            
            todays_date = datetime.datetime.now()
            self._wandb_writer = None
            run_name = self.__class__.__name__ + " " + self.__env.unwrapped.spec.id + "-" + str(todays_date).replace(":","-")
            
            self._wandb_writer = wandb.init(project=logger_title,
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
        return self._device
    
    @property
    def writer(self):
        return self._wandb_writer
    
    @property
    def gamma(self):
        return self._hparam_gamma
    
    @property
    def n_step(self):
        return self._hparam_n_step

    @property
    def is_wandb_logging_enabled(self,) -> bool:
        return self._enable_wandb_logging

    @property
    def max_mean_test_reward(self) -> float:
        return self._max_mean_test_reward
    
    @property
    def warm_up_iters(self):
        return self._hparam_warm_up_iters


    def get_agent_arguments(self,
                            local_dict: dict,
                            default_dict: dict):
        args_dict = {}
        for key in local_dict:
            if key in default_dict.keys():
                args_dict[key] = local_dict[key]
        return args_dict

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
        
        """
        if normalized_obs.max() > 1.0 \
            and np.abs(1 - normalized_obs.max()) > 0.1:
            raise RuntimeError("Detected Normalized vector having value greater than one.")

        if normalized_obs.min() < 0.0 \
            and np.abs(normalized_obs.min()) > 0.1:
            raise RuntimeError("Detected Normalized vector having value less than zero.")
        """
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

    def learn_start_callback(self,):
        """Learn Start callback. Called at the start of learn function once.
        """
        pass
    
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
                                n_step_transition_tuple: list) -> None:
        """Epsiode callback. Called after every epsiode.
        """
        pass

    def learn_start_episode_callback(self,
                                     episode):
        """Start of Episode callback. Called at the start of every episode.
        """
        pass

    def learn_evaluate_callback(self,
                                num_episodes: int,
                                render: bool = False) -> (float, float):
        """Evaluation callback. Called at every evaluation step.
        """
        eval_episode_reward = []
        eval_episode_length = []
        if render:
            test_env = gym.make(self.__env_str, render_mode="human")
        else:
            test_env = gym.make(self.__env_str)
        
        for _ in range(num_episodes):
            
            self.learn_start_episode_callback(_+1)

            state, info = test_env.reset()
            state = state.astype(np.float32)
            if self._hparam_normalize_observations:
                state = self.normalize_observation(state)

            done = False
            cum_reward = 0
            ep_length = 0

            while not done:

                if render:
                    test_env.render()
                
                ep_length += 1

                # Get an action to execute
                action = self.get_action(state,
                                         mode='eval')
                
                # Perform the action in the environment
                next_state, reward, terminated, truncated, info = test_env.step(action[0])
                next_state = next_state.astype(np.float32)
                if self._hparam_normalize_observations:
                    next_state = self.normalize_observation(next_state)
                cum_reward += reward

                if terminated or truncated:
                    eval_episode_reward.append(cum_reward)
                    eval_episode_length.append(ep_length)
                    done = True

                    if render:
                        print("Episode: {}, Cum Reward {}, Episode Length: {}".format(_ + 1,
                                                                                      cum_reward,
                                                                                      ep_length))

                state = next_state
        
        eval_episode_reward = np.array(eval_episode_reward)
        eval_episode_length = np.array(eval_episode_length)
        return np.mean(eval_episode_reward), np.mean(eval_episode_length)

    def __calculate_n_step_returns(self, 
                                   episode: list,
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

        self.learn_start_callback()
        
        # Initialize variables
        total_steps_count = 0

        for episode in tqdm(range(0, self._hparam_num_training_episodes)):
            
            state, info = self.env.reset()
            state = state.astype(np.float32)
            if self._hparam_normalize_observations:
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
                                         mode="train")

                # Perform the action in the environment
                next_state, reward, terminated, truncated, info = self.env.step(action[0])
                next_state = next_state.astype(np.float32)
                episode_sum_reward += reward

                if self._hparam_normalize_observations:
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
                                                                 n_step=self._hparam_n_step,
                                                                 gamma=self._hparam_gamma)
            self.learn_episode_callback(episode + 1, 
                                          episode_sum_reward,
                                          episode_length,
                                          buffer_transitions)
            
            # Evaluate agent performance
            if (episode + 1) % self._hparam_evaluation_freq_episodes == 0:
                eval_mean_reward, eval_mean_ep_length = self.learn_evaluate_callback(self._hparam_num_test_episodes)
                
                if self.is_wandb_logging_enabled:
                    self.writer.log({
                        "reward/eval": eval_mean_reward,
                        "episode_length/eval": eval_mean_ep_length
                    }, commit=False)
                
                # Save the model checkpoint
                if eval_mean_reward > self.max_mean_test_reward:
                    
                    self._max_mean_test_reward = eval_mean_reward
                    if self.is_wandb_logging_enabled:
                        prefix = "checkpoints/{}/{}/".format(self.__class__.__name__, wandb.run._run_id)
                    else:
                        prefix = "checkpoints/{}/".format(self.__class__.__name__)
                    
                    if not os.path.exists(prefix):
                        os.makedirs(prefix)
                    
                    check_point_name = self.env_id + "_" + self.__class__.__name__ + "_{}_episode_".format(episode+1) + "{:.2f}_mean_reward_checkpoint.pkt".format(eval_mean_reward)
                    self.save_checkpoint(prefix + check_point_name)
                    
                    if self.is_wandb_logging_enabled:
                        self.log_artifact(name=check_point_name,
                                          filepath=prefix + check_point_name,
                                          type="model",
                                          metadata={"mean_test_reward": eval_mean_reward})

            if self.is_wandb_logging_enabled:
                self.writer.log({"reward/train": episode_sum_reward,
                                 "episode_length/train": episode_length,
                                 "epsiode": episode+1}, commit=True)
    
    def log_artifact(self,
                     name: str,
                     filepath: str, 
                     type: str,
                     metadata: dict):
        
        artifact = wandb.Artifact(name=name, 
                                  type=type, 
                                  metadata=metadata)
        artifact.add_file(local_path=filepath)
        wandb.run.log_artifact(artifact)

    def load_checkpoint(self, path: str):
        """Method to load the state of the trainer.

        Args:
            path (str): Path of the checkpoint to load.
        """
        raise NotImplementedError()

    def save_checkpoint(self, path: str):
        """Method to save the state of the trainer.

        Args:
            path (str): Path to save the checkpoint.
        """
        raise NotImplementedError()