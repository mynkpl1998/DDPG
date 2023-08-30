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

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        #self.mu = mu * np.ones(size)
        self.mu = mu
        self.theta = theta
        self.sigma = sigma

    def reset(self,
              size: int):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu * np.ones(size)
        self.size = size

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu * np.ones(self.size) - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state

if __name__ == "__main__":

    obj = NormalNoise(mu=0.0, sigma=1.0)
    obj.reset(size=10)
    print(obj.sample())

    obj = OUNoise(mu=0.0, sigma=1.0, theta=0.15)
    obj.reset(size=10)
    print(obj.sample())



