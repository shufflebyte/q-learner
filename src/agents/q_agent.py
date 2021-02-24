import os
import numpy as np
import random
from utils import print_step
import time


class QAgent:
    def __init__(self, env, alpha, gamma, epsilon, q_table=[]):
        if len(q_table) == 0:
            self.q_table = np.zeros([env.observation_space.n, env.action_space.n])
        else:
            self.q_table = q_table

        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def q_learning(self):
        state = self.env.reset()
        steps, reward, = 0, 0
        done = False
        total_reward = 0

        while not done:
            if random.uniform(0, 1) < self.epsilon:
                action = self.env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(self.q_table[state])  # Exploit learned values

            next_state, reward, done, info = self.env.step(action)

            old_value = self.q_table[state, action]
            next_max = np.max(self.q_table[next_state])

            new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
            self.q_table[state, action] = new_value

            state = next_state
            steps += 1
            total_reward += reward
        return steps, total_reward

    def validation(self, sleep_timer=0):
        state = self.env.reset()
        steps, cum_reward = 0, 0
        done = False

        while not done:
            action = np.argmax(self.q_table[state])
            state, reward, done, info = self.env.step(action)
            if sleep_timer > 0:
                os.system('clear')
                self.env.render()
                print_step(steps, state, action, reward, cum_reward)
                time.sleep(sleep_timer)

            cum_reward += reward
            steps += 1

        return steps, cum_reward