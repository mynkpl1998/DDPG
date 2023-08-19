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

from typing import Literal, Dict, Any, Optional
from replay import ReplayBuffer
from models import Critic, Actor
import wandb

import optuna
from optuna import trial
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

DDPG_DEFAULT_DICT = {
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
    'critic_lr':1e-4,
    'actor_lr': 1e-3,
    'critic_loss': 'hubber',
    'num_training_episodes': int(20e3),
    'exploration_noise_scale': 0.1,
    'warm_up_iters': 5000,
    'max_gradient_norm': 0.5,
    'num_test_episodes': 10,
    'evaluation_freq_episodes': 10,
    'normalize_observations': True
}

class TrialEvluationCallback:

    def __init__(self,
                 op_trial: optuna.Trial):
        
        self.__op_trial = op_trial
        self.__eval_index = 0
        self.__is_pruned = False
        self.__last_mean_reward = 0
    
    @property
    def op_trial(self):
        return self.__op_trial
    
    @property
    def eval_index(self):
        return self.__eval_index
    
    @property
    def is_pruned(self):
        return self.__is_pruned
    
    @property
    def last_mean_reward(self):
        return self.__last_mean_reward
    
    def step(self,
             eval_value):
        
        # Report the result to optuna
        self.__eval_index += 1
        self.op_trial.report(eval_value, self.eval_index)
        self.__last_mean_reward = eval_value

        # Prune trial if needed
        if self.op_trial.should_prune():
            self.__is_pruned = True
            raise optuna.exceptions.TrialPruned()

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
                 normalize_observations: bool):
        

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
            
            self.critic_target.eval()
            self.actor_target.eval()
            with torch.no_grad():
                Q_s = self.critic_target(next_states, self.actor_target(next_states))
            self.critic_target.train()
            self.actor_target.train()

            target = rewards + self.__hparam_gamma * dones * Q_s.squeeze(dim=1)
            Q = self.critic(states, actions).squeeze(dim=1)
            
            if self.__hparam_critic_loss_fn == 'mse':
                critic_loss = F.mse_loss(Q, target)
            else:
                critic_loss = F.huber_loss(Q, target, delta=1.0)
            
            """
            if critic_loss.item() > 300:
                print("Returns: ", returns.mean(), "Rewards", rewards.mean(), rewards.size(), "Q: ", Q.mean(), "Target mean:", target.mean(), "Loss: ", critic_loss.item())
            """

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

    def learn(self,
              eval_callback: Optional[TrialEvluationCallback] = None):
        
        # Create a writer object for logging
        curr_timestamp = datetime.datetime.now()
        logger_title = "ddpga-" + self.__env.unwrapped.spec.id + "-" + str(curr_timestamp).replace(":","-")
        writer = wandb.init(project=logger_title,
                            config=self.get_hyper_parameters())

        # Initialize variables
        total_steps_count = 0
        exploration_noise_scale = self.__hparam_exploration_noise_scale
        Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'terminated', 'returns'])

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
            writer.log({
                "replay/size": self.__exp_replay.replay_size
            }, commit=False)
            
            # Evaluate agent performance
            
            if (episode + 1) % self.__hparam_evaluation_freq_episodes == 0:
                self.last_mean_test_reward = self.evaluate()
                
                if eval_callback is not None:
                    eval_callback.step(self.last_mean_test_reward)
                
                writer.log({
                    "reward/mean_test_reward": self.last_mean_test_reward
                }, commit=False)
            

            # Log metrics
            writer.log({"reward/episode_sum_reward": episode_sum_reward,
                        "exploration/noise_scale": exploration_noise_scale}, commit=True)
            
            #print("Episode: {} Train Cum Reward: {:.2f} Last Mean Test Cum Reward: {:.2f} Total Steps: {}.".format(episode+1, episode_sum_reward , self.last_mean_test_reward, total_steps_count))
        # Close writer
        writer.finish()
                

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

def objective(op_trial: optuna.Trial):

    kawags = DDPG_DEFAULT_DICT.copy()
    
    # Sample Hyper-parameters for the trial
    kawags.update(sample_ddpg_params(op_trial))


    # Create the DDPG learning and start training
    agent = DDPG(**DDPG_DEFAULT_DICT)
    
    # Create a callback function
    eval_callback = TrialEvluationCallback(op_trial)

    # Start Learning
    agent.learn(eval_callback=eval_callback)

    return eval_callback.last_mean_reward



if __name__ == "__main__":
    
    N_STARTUP_TRIALS = 5
    N_EVALUATIONS = 2
    N_TRIALS = 30
    TIMEOUT_MINS = 15

    # Set pytorch num threads to 1 for faster training.
    torch.set_num_threads(1)

    # Intialize sampling and pruning algorithms 
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS)

    # Create a study
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_MINS*60)