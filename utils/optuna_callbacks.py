import optuna

class TrialEvaluationCallback:

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