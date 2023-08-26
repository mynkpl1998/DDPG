PARAMS = {
    "Pendulum-v1": 
    { 
        "DDPG": 
        {   'env_id': "Pendulum-v1",
            'seed': 258,
            'replay_size': int(1e6),
            'polyak': 0.9995,
            'actor_critic_hidden_size': 256,
            'activation': 'relu',
            'update_batch_size': 256,
            'update_frequency': int(2),
            'update_iterations': 1,
            'gamma': 0.9,
            'n_step': 1,
            'critic_lr':1e-4,
            'actor_lr': 1e-3,
            'critic_loss_fn': 'hubber',
            'num_training_episodes': int(5000),
            'exploration_noise_scale': 0.1,
            'warm_up_iters': 10000,
            'max_gradient_norm': 0.5,
            'num_test_episodes': 20,
            'evaluation_freq_episodes': 20,
            'normalize_observations': True,
            'enable_wandb_logging': True,
            'logger_title': 'test_logger' 
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
    }
}