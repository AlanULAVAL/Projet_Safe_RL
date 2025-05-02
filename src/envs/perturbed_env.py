from gym.envs.classic_control.cartpole import CartPoleEnv

#On va créer l'environnement à perturbation en modifiant des paramètres du chariot, du poteau...
class PerturbedCartPoleEnv(CartPoleEnv) :
    def __init__(self, gravity, masspole):
        super().__init__()
        self.gravity = gravity
        self.masspole = masspole
        self.total_mass = self.masspole + self.masscart
        self.polemass_length = self.masspole * self.length