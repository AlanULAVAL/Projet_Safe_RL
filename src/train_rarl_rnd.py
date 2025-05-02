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

def train_rarl(
    protagonist_steps=10000,
    adversary_steps=70,
    iterations=50,
):
    protagonist = PPO("MlpPolicy", DummyVecEnv([lambda: AdversarialCartPole()]), verbose=0)
    
    best_gravity = 15
    best_masspole = 1.3

    mid_it = 20

    reward_history = []
    max_no_progress = 5
    reward_tol = 1.5

    for it in range(iterations):
        print(f"\nItération {it+1}/{iterations}")

        #La difficulté de l'environnement augmente progressivement de manière quadratique
        if it<=20:
            scaled_gravity_range = (15 + (40-15) * (it/mid_it)**2, 40 + (80-40) * (it/mid_it)**2)
            scaled_masspole_range = (1.3 + (5.0 - 1.3) * (it/mid_it)**2, 4.0 + (10.0-4.0) * (it/mid_it)**2)

        env = DummyVecEnv([lambda: AdversarialCartPole(gravity=best_gravity, pole_mass=best_masspole)])
        protagonist.set_env(env)
        protagonist.learn(total_timesteps=protagonist_steps)

        best_gravity, best_masspole = None, None
        worst_reward = float("inf")

        #On augmente la reward minimale nécessaire au fur et à mesure des itérations
        if it < mid_it:
            reward_threshold = 10 + (it / mid_it) * 100
        else:
            reward_threshold = 110

        for _ in range(adversary_steps):
            gravity = np.random.uniform(*scaled_gravity_range)
            masspole = np.random.uniform(*scaled_masspole_range)

            temp_env = DummyVecEnv([lambda: AdversarialCartPole(gravity, masspole)])
            protagonist.set_env(temp_env)
            mean_reward = evaluer_agent(protagonist, temp_env, n_episodes=3)
            
            #On choisi le pire environnement selon le critère d'une reward minimale exigée
            if mean_reward < worst_reward and mean_reward > reward_threshold:
                worst_reward = mean_reward
                best_gravity = gravity
                best_masspole = masspole

        print(f"Adversaire choisi l'environnement: gravité = {best_gravity:.2f}, poids du poteau ={best_masspole:.2f} avec reward = {worst_reward:.2f}")
        
        #On stop l'entraînement si les reward stagnent
        reward_history.append(worst_reward)
        if len(reward_history) > max_no_progress:
            recent = reward_history[-max_no_progress:]
            max_recent = max(recent)
            min_recent = min(recent)
            if max_recent - min_recent < reward_tol:
                no_progress_count += 1
            else:
                no_progress_count = 0

            if no_progress_count >= 3:
                print("Early stopping: stagnation détectée.")
                break

    protagonist.save("models/protagonist_rarl")

    return protagonist

if __name__ == "__main__":
    train_rarl()