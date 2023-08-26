import torch
import argparse
from agents.ddpg import DDPG, DDPG_DEFAULT_PARAMS

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint to load.")
    args = parser.parse_args()

    state = torch.load(args.checkpoint)
    algo = state["algo"]
    hyperparams = state['hyper_params']
    if algo == "DDPG":
        
        params = DDPG_DEFAULT_PARAMS
        params['enable_wandb_logging'] = False
        for param in hyperparams:
            params[param] = hyperparams[param]

        params["env_id"] = state["env_id"]
        
        agent = DDPG(**params)
        agent.load_checkpoint(args.checkpoint)
        agent.learn_evaluate_callback(num_episodes=10, render=True)

    else:
        raise NotImplementedError()
