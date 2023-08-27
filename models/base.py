class BaseModel:

    def __init__(self,
                 observation_dims: int,
                 action_dims: int,):
        
        raise NotImplementedError()
    
    @property
    def actor(self,):
        raise NotImplementedError()

    @property
    def critic(self):
        raise NotImplementedError()
    
    