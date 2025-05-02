from stable_baselines3 import PPO
from envs.perturbed_env import PerturbedCartPoleEnv
from gym.wrappers import TimeLimit
import numpy as np
import matplotlib.pyplot as plt

def tester_agent(path, env):
    model = PPO.load(path, env)
    score = 0
    done = False
    current_state = env.reset()
    while not done:
        action, _ = model.predict(current_state)
        new_state, reward, done, _ = env.step(action)
        current_state = new_state
        score += reward
    return score

if __name__ == "__main__":
    path_agent = "models/agent_ppo.zip"
    path_protagonist = "models/protagonist_rarl4.zip"

    gravity = np.array([9.8, 19, 28, 37, 46, 55, 64, 71, 79])
    masspole = np.array([0.1, 1.3, 2.6, 3.9, 5.2, 6.5, 7.8, 9.1, 10.4])

    size = gravity.shape[0]
    nb_itération = 10

    performance_a = np.zeros((size, size))
    performance_p = np.zeros((size, size))
    for i, grav in enumerate(gravity):
        for j, mass in enumerate(masspole):
            env = TimeLimit(PerturbedCartPoleEnv(grav, mass), max_episode_steps=500)
            final_scores_a = np.zeros(nb_itération)
            final_scores_p = np.zeros(nb_itération)
            for k in range(nb_itération):
                final_scores_a[k] = tester_agent(path_agent, env)
                final_scores_p[k] = tester_agent(path_protagonist, env)
            mean_a = np.mean(final_scores_a)
            mean_p = np.mean(final_scores_p)
            performance_a[i, j] = mean_a
            performance_p[i, j] = mean_p
        print(f"Progression : {int(100*(i+1)/size)}%")
    plt.imshow(performance_a, extent=[masspole.min(), masspole.max(), gravity.min(), gravity.max()],
            origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Performance')
    plt.xlabel('Masspole')
    plt.ylabel('Gravity')
    plt.title('Performance selon gravity et masspole pour agent simple')
    plt.show()

    plt.imshow(performance_p, extent=[masspole.min(), masspole.max(), gravity.min(), gravity.max()],
            origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Performance')
    plt.xlabel('Masspole')
    plt.ylabel('Gravity')
    plt.title('Performance selon gravity et masspole pour agent safe')
    plt.show()
    