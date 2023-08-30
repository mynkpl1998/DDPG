import numpy as np

class Noise:

    def __init__(self):
        pass
    
    def reset(self,
              size: int):
        self._size = size

    def sample(self):
        raise NotImplementedError()


class NormalNoise(Noise):

    def __init__(self,
                 mu: float,
                 sigma: float):
        super().__init__()
        self.__sigma = sigma
        self.__mu = mu

    def reset(self, 
              size: int):
        super().reset(size=size)
    
    def sample(self):
        return np.random.normal(self.__mu,
                                self.__sigma,
                                size=self._size)

if __name__ == "__main__":

    obj = NormalNoise(mu=0.0, sigma=1.0)
    obj.reset(size=10)
    print(obj.sample())



