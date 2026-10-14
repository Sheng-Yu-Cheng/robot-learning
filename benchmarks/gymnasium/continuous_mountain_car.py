import gymnasium as gym

class ContinuousMountainCar(gym.Wrapper):
    def __init__(self, verbose=False):
        env = gym.make("MountainCarContinuous-v0")
        super().__init__(env)

        self.state_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]
        self.action_low = self.action_space.low
        self.action_high = self.action_space.high

        self.verbose = verbose

        if self.verbose:
            print(
                f"MountainCarContinuous-v0 | "
                f"state_dim={self.state_dim} | "
                f"action_dim={self.action_dim} | "
                f"action_range=[{self.action_low}, {self.action_high}]"
            )