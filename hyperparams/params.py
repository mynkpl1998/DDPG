PARAMS = {
    "Pendulum-v1": 
    { 
        "DDPG": 
        {   'env_id': "Pendulum-v1",
            'render': False,
            'seed': 258,
            'replay_size': int(1e6),
            'polyak': 0.995,
            'update_batch_size': 256,
            'update_frequency': int(1),
            'update_iterations': 1,
            'gamma': 0.9,
            'n_step': 5,
            'num_training_episodes': int(5000),
            'warm_up_iters': 10000,
            'max_gradient_norm': 1.0,
            'num_test_episodes': 20,
            'evaluation_freq_episodes': 20,
            'normalize_observations': True,
            'enable_wandb_logging': True,
            'logger_title': 'noise_logger',
            'actor': 'SimpleActor',
            'actor_params': {'SimpleActor': {'hidden_size': 256, 'lr': 1e-3}},
            'critic': 'SimpleCritic',
            'critic_params': {'SimpleCritic': {'hidden_size': 256, 'lr': 1e-3, 'loss_fn': 'hubber'}},
            'exploration_noise_type': 'NormalNoise',
            'exploration_noise_params': {'NormalNoise': {'mu': 0.0, 'sigma': 0.1},
                                         'OUNoise': {'mu': 0.0, 'sigma': 0.2, 'theta': 0.15}}
        },

        "TD3": 
        {   'env_id': "Pendulum-v1",
            'seed': 258,
            'replay_size': int(1e6),
            'polyak': 0.9995,
            'actor_critic_hidden_size': 256,
            'activation': 'relu',
            'update_batch_size': 256,
            'update_frequency': int(10),
            'update_iterations': 2,
            'policy_delay': 1,
            'gamma': 0.95,
            'n_step': 1,
            'critic_lr':1e-4,
            'actor_lr': 1e-3,
            'critic_loss_fn': 'hubber',
            'num_training_episodes': int(5000),
            'target_noise': 0.2,
            'target_noise_clip': 0.5,
            'warm_up_iters': 10000,
            'max_gradient_norm': 0.5,
            'num_test_episodes': 20,
            'evaluation_freq_episodes': 20,
            'normalize_observations': True,
            'enable_wandb_logging': False,
            'logger_title': 'noise_logger',
            'exploration_noise_type': 'NormalNoise',
            'exploration_noise_params': {'NormalNoise': {'mu': 0.0, 'sigma': 0.1}}
        },
    },

    "BipedalWalker-v3": 
    {
        "DDPG": 
        {
            'env_id': "BipedalWalker-v3",
            'seed': 258,
            'replay_size': int(1e6),
            'polyak': 0.995,
            'actor_critic_hidden_size': 256,
            'activation': 'relu',
            'update_batch_size': 256,
            'update_frequency': int(1000),
            'update_iterations': 4,
            'gamma': 0.98,
            'n_step': 3,
            'critic_lr':1e-4,
            'actor_lr': 1e-4,
            'critic_loss_fn': 'hubber',
            'num_training_episodes': int(50000),
            'exploration_noise_scale': 0.01,
            'warm_up_iters': 10000,
            'max_gradient_norm': 1.0,
            'num_test_episodes': 20,
            'evaluation_freq_episodes': 100,
            'normalize_observations': True,
            'enable_wandb_logging': True,
            'logger_title': 'test_logger'
        },
        
        "TD3": 
        {   'env_id': "BipedalWalker-v3",
            'seed': 258,
            'replay_size': int(1e6),
            'polyak': 0.995,
            'actor_critic_hidden_size': 256,
            'activation': 'relu',
            'update_batch_size': 256,
            'update_frequency': int(100),
            'update_iterations': 4,
            'policy_delay': 2,
            'gamma': 0.95,
            'n_step': 3,
            'critic_lr':1e-4,
            'actor_lr': 1e-4,
            'critic_loss_fn': 'hubber',
            'num_training_episodes': int(50000),
            'exploration_noise_scale': 0.1,
            'target_noise': 0.2,
            'target_noise_clip': 0.5,
            'warm_up_iters': 10000,
            'max_gradient_norm': 0.5,
            'num_test_episodes': 20,
            'evaluation_freq_episodes': 20,
            'normalize_observations': True,
            'enable_wandb_logging': True,
            'logger_title': 'td3_test_logger',
        },
    },

    "HalfCheetah-v4":
    {
        "DDPG":
        {
            'env_id': "HalfCheetah-v4",
            'seed': 258,
            'replay_size': int(1e6),
            'polyak': 0.995,
            'update_batch_size': 256,
            'update_frequency': int(2),
            'update_iterations': 1,
            'gamma': 0.98,
            'n_step': 5,
            'num_training_episodes': int(10000),
            'warm_up_iters': 20000,
            'max_gradient_norm': 1.0,
            'num_test_episodes': 20,
            'evaluation_freq_episodes': 100,
            'normalize_observations': False,
            'enable_wandb_logging': True,
            'logger_title': 'noise_logger',
            'actor': 'SimpleActor',
            'actor_params': {'SimpleActor': {'hidden_size': 256, 'lr': 1e-3}},
            'critic': 'SimpleCritic',
            'critic_params': {'SimpleCritic': {'hidden_size': 256, 'lr': 1e-3, 'loss_fn': 'hubber'}},
            'exploration_noise_type': 'NormalNoise',
            'exploration_noise_params': {'NormalNoise': {'mu': 0.0, 'sigma': 0.2},
                                         'OUNoise': {'mu': 0.0, 'sigma': 0.2, 'theta': 0.15}}            
        }
    },

    "sumo/v2i-v0":
    {
        "DDPG":
        {
            'env_id': "sumo/v2i-v0",
            'render': False,
            'seed': 258,
            'replay_size': int(1e6),
            'polyak': 0.995,
            'update_batch_size': 256,
            'update_frequency': int(2),
            'update_iterations': 1,
            'gamma': 0.98,
            'n_step': 5,
            'num_training_episodes': int(10000),
            'warm_up_iters': 20000,
            'max_gradient_norm': 1.0,
            'num_test_episodes': 20,
            'evaluation_freq_episodes': 100,
            'normalize_observations': False,
            'enable_wandb_logging': True,
            'logger_title': 'sumo_v2x',
            'actor': 'SimpleActor',
            'actor_params': {'SimpleActor': {'hidden_size': 256, 'lr': 1e-3}},
            'critic': 'SimpleCritic',
            'critic_params': {'SimpleCritic': {'hidden_size': 256, 'lr': 1e-3, 'loss_fn': 'hubber'}},
            'exploration_noise_type': 'NormalNoise',
            'exploration_noise_params': {'NormalNoise': {'mu': 0.0, 'sigma': 0.2},
                                         'OUNoise': {'mu': 0.0, 'sigma': 0.2, 'theta': 0.15}}            
        }
    }
}