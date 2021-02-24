import gym
import time
import os
from utils import print_step


class RandomAgent:
    def __init__(self, env):
        self.env = env

    def random_inference(self):
        steps = 0
        total_reward = 0

        done = False
        self.env.reset()
        # one game round of randomness
        while not done:
            action = self.env.action_space.sample()
            state, reward, done, info = self.env.step(action)

            steps += 1
            total_reward += reward
            self.env.render()
            print_step(steps, state, action, reward, total_reward)
            time.sleep(0.25)
            os.system('clear')


def demo():
    env = gym.make("Taxi-v3")
    random_agent = RandomAgent(env=env)
    random_agent.random_inference()


demo()
