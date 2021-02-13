import gym
from utils import print_step
import time
import os


class RandomAgent:

    def __init__(self, env):
        self.env = env

    def random_inference(self):
        steps = 0
        penalties, reward = 0, 0
        frames = []  # for animation
        cum_reward = 0

        done = False
        self.env.reset()
        # one game round of randomness
        while not done:
            action = self.env.action_space.sample()
            state, reward, done, info = self.env.step(action)

            if reward == -10:
                penalties += 1

            steps += 1
            cum_reward += reward
            self.env.render()
            print_step(steps, state, action, reward, cum_reward)
            time.sleep(0.25)
            os.system('clear')


def demo():
    env = gym.make("Taxi-v3")
    random_agent = RandomAgent(env=env)
    random_agent.random_inference()


demo()
