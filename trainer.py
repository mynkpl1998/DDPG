import torch
import optuna
import argparse
import importlib
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from agents.ddpg import DDPG, DDPG_DEFAULT_PARAMS, sample_ddpg_params
from agents.td3 import TD3, TD3_DEFAULT_PARAMS
from agents.base import BASE_AGENT_DEFAULT_PARAMS

from hyperparams.params import PARAMS

from utils.optuna_callbacks import TrialEvaluationCallback

                
def objective(op_trial: optuna.Trial):

    kawags = DDPG_DEFAULT_PARAMS.copy()
    
    # Sample Hyper-parameters for the trial
    kawags.update(sample_ddpg_params(op_trial))


    # Create the DDPG learning and start training
    agent = DDPG(**kawags)
    
    # Create a callback function
    eval_callback = TrialEvaluationCallback(op_trial)

    # Start Learning
    agent.learn(eval_callback=eval_callback)

    return eval_callback.last_mean_reward



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, required=True, help="Environment ID")
    parser.add_argument("--algo", type=str, required=True, choices=["DDPG", "TD3"], help="Algorithm to use for training.")
    args = parser.parse_args()

    tune = False

    if tune:

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
    
    else:

        if args.env_id in PARAMS and args.algo in PARAMS[args.env_id].keys():
            PARAM_DICT = PARAMS[args.env_id][args.algo]
            
        else:
            print("Existing Hyperparams for {} for {} not found. Using default values.".format(args.env_id,
                                                                                               args.algo))
            if args.algo == "DDPG":
                PARAM_DICT = DDPG_DEFAULT_PARAMS
            elif args.algo == "TD3":    
                PARAM_DICT = TD3_DEFAULT_PARAMS
        
        if args.algo == "DDPG":
            agent = DDPG(**PARAM_DICT)
        elif args.algo == "TD3":
            agent = TD3(**PARAM_DICT)
        
        # Start learning
        agent.learn()
