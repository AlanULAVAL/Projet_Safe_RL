import numpy as np
from stable_baselines3 import PPO
from envs.rarl_env import AdversarialCartPole
from stable_baselines3.common.vec_env import DummyVecEnv

def evaluer_agent(model, env, n_episodes=5):
    all_rewards = []
    for _ in range(n_episodes):
        current_state = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(current_state)
            next_state, reward, done, _ = env.step(action)
            current_state = next_state
            total_reward += reward[0]
        all_rewards.append(total_reward)
    return np.mean(all_rewards)

def train_rarl_simple(
    protagonist_steps=10000,
    adversary_steps=70,
    iterations=50,
):
    protagonist = PPO("MlpPolicy", DummyVecEnv([lambda: AdversarialCartPole()]), verbose=0)

    best_gravity = 15
    best_masspole = 0.3

    for it in range(iterations):
        print(f"\nItération {it+1}/{iterations}")

        # Intervalles d'entraînement
        gravity_range = (15, 50)
        masspole_range = (0.3, 5)

        # Entraînement du protagoniste
        env = DummyVecEnv([lambda: AdversarialCartPole(gravity=best_gravity, pole_mass=best_masspole)])
        protagonist.set_env(env)
        protagonist.learn(total_timesteps=protagonist_steps)

        best_gravity, best_masspole = None, None
        worst_reward = float("inf")

        # Recherche de l'environnement le plus difficile
        for _ in range(adversary_steps):
            gravity = np.random.uniform(*gravity_range)
            masspole = np.random.uniform(*masspole_range)

            temp_env = DummyVecEnv([lambda: AdversarialCartPole(gravity, masspole)])
            protagonist.set_env(temp_env)
            mean_reward = evaluer_agent(protagonist, temp_env, n_episodes=3)

            if mean_reward < worst_reward:
                worst_reward = mean_reward
                best_gravity = gravity
                best_masspole = masspole

        print(f"Adversaire choisit : gravité = {best_gravity:.2f}, masse = {best_masspole:.2f}, reward = {worst_reward:.2f}")

    protagonist.save("models/protagonist_rarl_simple")
    return protagonist

if __name__ == "__main__":
    train_rarl_simple()
