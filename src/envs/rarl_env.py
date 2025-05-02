import gym

#Classe de l'environnement modifiable par l'antagoniste
class AdversarialCartPole(gym.Env):
    def __init__(self, gravity=9.8, pole_mass=0.1):
        super().__init__()
        self.gravity = gravity
        self.pole_mass = pole_mass
        self.base_env = gym.make("CartPole-v1")
        self.env = self._make_custom_env()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def _make_custom_env(self):
        env = gym.make("CartPole-v1")
        env.env.gravity = self.gravity
        env.env.masspole = self.pole_mass
        env.env.total_mass = env.env.masspole + env.env.masscart
        env.env.polemass_length = env.env.masspole * env.env.length
        return env

    def reset(self):
        self.env.close()
        self.env = self._make_custom_env()
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()





