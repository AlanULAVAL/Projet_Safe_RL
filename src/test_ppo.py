import gym
from stable_baselines3 import PPO
from envs.perturbed_env import PerturbedCartPoleEnv
from gym.wrappers import TimeLimit
import numpy as np

def tester_agent():
    # env = gym.make("CartPole-v1")
    env = TimeLimit(PerturbedCartPoleEnv(), max_episode_steps=500)
    model = PPO.load("models/agent_ppo.zip", env)
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
    final_scores = []
    for _ in range(10):
        final_scores.append(tester_agent())
    mean = np.mean(final_scores)
    std = np.std(final_scores)
    print(f"Le score final est en moyenne de {mean} ± {std}")