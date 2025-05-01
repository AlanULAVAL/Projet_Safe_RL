import gym
from stable_baselines3 import PPO


def agent_ppo():
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(100_000)
    model.save("models/agent_ppo")

    return model

if __name__ == "__main__":
    agent_ppo()