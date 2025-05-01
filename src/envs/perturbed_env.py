from gym.envs.classic_control.cartpole import CartPoleEnv

#On va créer l'environnement à perturbation en modifiant des paramètres du chariot, du poteau...
class PerturbedCartPoleEnv(CartPoleEnv) :
    def __init__(self):
        super().__init__()
        self.gravity = 30.0
        self.masspole = 2.0