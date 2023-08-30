from typing import Literal
from agents.base import BaseAgent
from buffers.replay import ReplayBuffer
from models.models import Actor, Critic
import torch.optim as optim
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from typing import Optional

TD3_DEFAULT_PARAMS = {
    'env_id': "BipedalWalker-v3",
    'seed': 258,
    'replay_size': int(1e6),
    'polyak': 0.9995,
    'actor_critic_hidden_size': 256,
    'activation': 'relu',
    'update_batch_size': 256,
    'update_frequency': int(2000),
    'update_iterations': 5,
    'policy_delay': 2,
    'gamma': 0.95,
    'n_step': 3,
    'critic_lr':1e-4,
    'actor_lr': 1e-4,
    'critic_loss_fn': 'hubber',
    'num_training_episodes': int(50000),
    'exploration_noise_scale': 0.2,
    'target_noise': 0.2,
    'target_noise_clip': 0.5,
    'warm_up_iters': 1000,
    'max_gradient_norm': 1.0,
    'num_test_episodes': 20,
    'evaluation_freq_episodes': 200,
    'normalize_observations': True,
    'enable_wandb_logging': False,
    'logger_title': 'test_logger'
}

class TD3(BaseAgent):
    
    def __init__(self, env_id: str, 
                 seed: int, 
                 replay_size: int, 
                 polyak: float, 
                 actor_critic_hidden_size: int, 
                 critic_loss_fn: Literal['mse', 'hubber'], 
                 activation: Literal['tanh', 'relu'], 
                 update_batch_size: int, 
                 gamma: float, 
                 n_step: int, 
                 update_frequency: int, 
                 update_iterations: int,
                 policy_delay: int,
                 critic_lr: float, 
                 actor_lr: float, 
                 max_gradient_norm: float, 
                 exploration_noise_scale: float,
                 target_noise: float,
                 target_noise_clip: float,
                 num_training_episodes: int, 
                 warm_up_iters: int, 
                 num_test_episodes: int, 
                 evaluation_freq_episodes: int, 
                 normalize_observations: bool, 
                 enable_wandb_logging: bool, 
                 logger_title: Optional[str] = None):
        
        super().__init__(env_id, 
                         seed, 
                         gamma, 
                         n_step, 
                         num_training_episodes, 
                         num_test_episodes,
                         warm_up_iters,
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
        self.__hparam_policy_delay = policy_delay
        self.__hparam_critic_lr = critic_lr
        self.__hparam_actor_lr = actor_lr
        self.__hparam_max_gradient_norm = max_gradient_norm
        self.__hparam_exploration_noise_scale = exploration_noise_scale
        self.__hparam_target_noise = target_noise
        self.__hparam_target_noise_clip = target_noise_clip
        self.__hparam_critic_loss_fn = critic_loss_fn
        
        if self.__hparam_policy_delay > self.__hparam_update_iterations:
            raise ValueError("Policy Delay must be < Update iterations")

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
        self.__critic_first, self.__critic_first_target, self.__critic_first_optimizer = self.__build_critic(observation_dims=self.env.observation_space.shape[0],
                                                                                                             action_dims=self.env.action_space.shape[0],
                                                                                                             hidden_size=actor_critic_hidden_size,
                                                                                                             activation=activation,
                                                                                                             device=self.device,
                                                                                                             critic_lr=self.__hparam_critic_lr)
        
        self.__critic_second, self.__critic_second_target, self.__critic_second_optimizer = self.__build_critic(observation_dims=self.env.observation_space.shape[0],
                                                                                                                action_dims=self.env.action_space.shape[0],
                                                                                                                hidden_size=actor_critic_hidden_size,
                                                                                                                activation=activation,
                                                                                                                device=self.device,
                                                                                                                critic_lr=self.__hparam_critic_lr)

        # Actor Network
        self.__actor, self.__actor_target, self.__actor_optimizer = self.__build_actor(observation_dims=self.env.observation_space.shape[0],
                                                                                       action_dims=self.env.action_space.shape[0],
                                                                                       hidden_size=actor_critic_hidden_size,
                                                                                       activation=activation,
                                                                                       device=self.device,
                                                                                       actor_lr=self.__hparam_actor_lr)
    
    @property
    def critic_first(self):
        return self.__critic_first

    @property
    def critic_first_target(self):
        return self.__critic_first_target

    @property
    def critic_second(self):
        return self.__critic_second

    @property
    def critic_second_target(self):
        return self.__critic_second_target
    
    @property
    def critic_optimizer_first(self):
        return self.__critic_first_optimizer

    @property
    def critic_optimizer_second(self):
        return self.__critic_second_optimizer

    @property
    def actor_optimizer(self):
        return self.__actor_optimizer

    @property
    def actor(self):
        return self.__actor
    
    @property
    def actor_target(self):
        return self.__actor_target
    
    def __build_critic(self, 
                       observation_dims: int,
                       action_dims: int,
                       hidden_size: int,
                       activation: str,
                       device: str,
                       critic_lr: float):
        
        critic       = Critic(observation_dims,
                              action_dims,
                              hidden_size,
                              activation).to(device)

        critic_targ  = Critic(observation_dims,
                              action_dims,
                              hidden_size,
                              activation).to(device)
        
        optimizer = optim.Adam(critic.parameters(),
                               lr=critic_lr)

        critic.load_state_dict(critic_targ.state_dict())
        return critic, critic_targ, optimizer


    def __build_actor(self, 
                      observation_dims: int,
                      action_dims: int,
                      hidden_size: int,
                      activation: str,
                      device: str,
                      actor_lr: float):
    
        actor       = Actor(observation_dims,
                             action_dims,
                             hidden_size,
                             activation).to(device)

        actor_targ  = Actor(observation_dims,
                             action_dims,
                             hidden_size,
                             activation).to(device)

        optimizer = optim.Adam(actor.parameters(),
                               lr=actor_lr)
        
        actor.load_state_dict(actor_targ.state_dict())
        return actor, actor_targ, optimizer
    
    
    def first_critic_soft_update(self,
                                 polyak:float):        
        for param, target_param in zip(self.critic_first.parameters(), self.critic_first_target.parameters()):
            target_param.data.copy_( polyak * target_param.data + (1 - polyak) * param.data )

    def second_critic_soft_update(self,
                                  polyak:float):        
        for param, target_param in zip(self.critic_second.parameters(), self.critic_second_target.parameters()):
            target_param.data.copy_( polyak * target_param.data + (1 - polyak) * param.data )
    
    def actor_soft_update(self,
                          polyak: float):
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_( polyak * target_param.data + (1 - polyak) * param.data )
    
    def set_wandb_logging_metrics(self) -> None:            

        super().set_wandb_logging_metrics()
        # define which metrics will be plotted against it
        wandb.define_metric("replay/size", step_metric="episode")
        wandb.define_metric("loss/actor", step_metric="step")
        wandb.define_metric("loss/critic_first", step_metric="step")
        wandb.define_metric("loss/critic_second", step_metric="step")
        wandb.define_metric("returns/estimated_first", step_metric="step")
        wandb.define_metric("returns/estimated_second", step_metric="step")
        wandb.define_metric("returns/true_returns", step_metric="step")
    
    def learn_episode_callback(self, episode: int, cum_reward: float, episode_length: int, n_step_transition_tuple: list) -> None:
        
        # Add the episode to the replay buffer
        self.__exp_replay.add_epsiode(n_step_transition_tuple)
            
        # log the current replay size
        if self.is_wandb_logging_enabled:
            # Log current replay size
                self.writer.log({
                    "replay/size": self.__exp_replay.replay_size
                }, commit=False)
        
    def learn_start_callback(self):
        print("Action space size: ", self.env.action_space.shape)
        self._action_noise.reset(self.env.action_space.shape)

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
    
    def __train_step(self, batch_size: int):

        critic_first_losses = []
        critic_second_losses = []
        returns_estimated_first = []
        returns_estimated_second = []
        returns_true = []
        actor_losses = []

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

                # Get next state action
                actions_next = None
                self.actor_target.eval()
                with torch.no_grad():
                    actions_next = self.actor_target(next_states).detach()
                self.actor_target.train()
                target_noise = torch.normal(mean=0.0, 
                                            std=self.__hparam_target_noise,
                                            size=actions_next.size()).clip(min=-self.__hparam_target_noise_clip,
                                                                           max=self.__hparam_target_noise_clip)
                target_noise = target_noise.to(self.device)
                action_space_min_torch = torch.FloatTensor(self.env.action_space.low).to(self.device)
                action_space_max_torch = torch.FloatTensor(self.env.action_space.high).to(self.device)
                actions_next = (actions_next + target_noise).clip(min=action_space_min_torch,
                                                                  max=action_space_max_torch)
                
                # Get Q_s
                self.critic_first_target.eval()
                self.critic_second_target.eval()
                with torch.no_grad():
                    q_next_first = self.critic_first_target(next_states, actions_next)
                    q_next_second = self.critic_second_target(next_states, actions_next)
                self.critic_first_target.train()
                self.critic_second_target.train()

                q_next = torch.minimum(q_next_first, q_next_second)
                target = rewards + (self.gamma**self.n_step) * dones * q_next.squeeze(dim=1)
                
                Q_first = self.critic_first(states, actions).squeeze(dim=1)
                Q_second = self.critic_second(states, actions).squeeze(dim=1)

                if self.__hparam_critic_loss_fn == 'mse':
                    critic_loss_first = F.mse_loss(Q_first, target)
                else:
                    critic_loss_first = F.huber_loss(Q_first, target, delta=1.0)

                if self.__hparam_critic_loss_fn == 'mse':
                    critic_loss_second = F.mse_loss(Q_second, target)
                else:
                    critic_loss_second = F.huber_loss(Q_second, target, delta=1.0)

                self.critic_optimizer_first.zero_grad()
                critic_loss_first.backward()
                # Clip the gradients
                torch.nn.utils.clip_grad_norm_(self.critic_first.parameters(), self.__hparam_max_gradient_norm)
                # Update gradients
                self.critic_optimizer_first.step()

                self.critic_optimizer_second.zero_grad()
                critic_loss_second.backward()
                # Clip the gradients
                torch.nn.utils.clip_grad_norm_(self.critic_second.parameters(), self.__hparam_max_gradient_norm)
                # Update gradients
                self.critic_optimizer_second.step()

                critic_first_losses.append(critic_loss_first.item())
                critic_second_losses.append(critic_loss_second.item())
                returns_estimated_first.append(Q_first.mean().item())
                returns_estimated_second.append(Q_second.mean().item())
                returns_true.append(returns.mean().item())

                # ------------------ Update Actor Network -------------------- #
                if (_ + 1) % self.__hparam_policy_delay == 0:

                    # Update actor
                    actor_loss = -self.critic_first(states, self.actor(states)).mean()
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    # Clip the gradients
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.__hparam_max_gradient_norm)
                    # Update gradients
                    self.actor_optimizer.step()

                    actor_losses.append(actor_loss.item())

                    # Update target network weights
                    self.first_critic_soft_update(polyak=self.__hparam_polyak)
                    self.second_critic_soft_update(polyak=self.__hparam_polyak)
                    self.actor_soft_update(polyak=self.__hparam_polyak)

        critic_first_losses = np.array(critic_first_losses)
        critic_second_losses = np.array(critic_second_losses)
        returns_estimated_first = np.array(returns_estimated_first)
        returns_estimated_second = np.array(returns_estimated_second)
        returns_true = np.array(returns_true)
        actor_losses = np.array(actor_losses)
        
        return critic_first_losses.mean(), critic_second_losses.mean(), returns_estimated_first.mean(), returns_estimated_second.mean(), actor_losses.mean(), returns_true.mean()
    
    def learn_step_callback(self, 
                              step: int,
                              transition_tuple: tuple) -> None:
        """Step callback. Called at every step.
        """
        if step % self.__hparam_update_frequency == 0 \
            and step > self.warm_up_iters:
            critic_loss_first, critic_loss_second, returns_est_first, returns_est_second, actor_loss, returns_true = self.__train_step(batch_size=self.__hparam_update_batch_size)

            if self.is_wandb_logging_enabled:
                self.writer.log({
                    "loss/critic_first": critic_loss_first,
                    "loss/critic_second": critic_loss_second,
                    "loss/actor": actor_loss,
                    "returns/estimated_first": returns_est_first,
                    "returns/estimated_second": returns_est_second,
                    "returns/true_returns": returns_true
                }, commit=False)

    def load_checkpoint(self, path: str):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__device = device
        state = torch.load(path, map_location=self.device)
        self.critic_first.load_state_dict(state["critic_first"])
        self.critic_optimizer_first.load_state_dict(state["first_critic_optimizer"])
        self.critic_second.load_state_dict(state["critic_second"])
        self.critic_optimizer_second.load_state_dict(state["second_critic_optimizer"])
        self.actor.load_state_dict(state["actor"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        hyper_params = state["hyper_params"]
        print("Loaded checkpoint: {}".format(path))


    def save_checkpoint(self, path: str):
        torch.save({
            "critic_first": self.critic_first.state_dict(),
            "critic_second": self.critic_second.state_dict(),
            "actor": self.actor.state_dict(),
            "first_critic_optimizer": self.critic_optimizer_first.state_dict(),
            "second_critic_optimizer": self.critic_optimizer_second.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "hyper_params": self.get_hyper_parameters(),
            "algo": "TD3",
            "env_id": self.env_id
        }, path)