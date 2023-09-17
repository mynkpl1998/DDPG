import torch
import argparse
from agents.ddpg import DDPG, DDPG_DEFAULT_PARAMS
from agents.td3 import TD3, TD3_DEFAULT_PARAMS

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint to load.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device)
    algo = state["algo"]
    hyperparams = state['hyper_params']
    
    if algo == "DDPG":
        
        hyperparams['enable_wandb_logging'] = False
        hyperparams['render'] = True
        agent = DDPG(**hyperparams)
        agent.load_checkpoint(args.checkpoint)
        agent.learn_evaluate_callback(num_episodes=10, render=True)

    elif algo == "TD3":
        hyperparams['enable_wandb_logging'] = False
        hyperparams['render'] = True
        agent = TD3(**hyperparams)
        agent.load_checkpoint(args.checkpoint)
        agent.learn_evaluate_callback(num_episodes=10, render=True)
        
    else:
        raise NotImplementedError()
