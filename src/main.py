from agents.q_agent import QAgent
import gym
import sys
import os
import utils
import time
import numpy as np
import pickle

decay = True
load_from_file = False


def main(argv):
    if len(argv) > 1:
        env_name = argv[1]
    else:
        env_name = "Taxi-v3"

    env = gym.make(env_name)

    start_alpha = 0.1
    start_gamma = 0.6
    start_epsilon = 0.1
    decay_rate = 0.0001

    if load_from_file:
        file = open('q_table', 'rb')
        loaded_q_table = pickle.load(file)
        q_agent = QAgent(env, alpha=start_alpha, gamma=start_gamma, epsilon=start_epsilon, q_table=loaded_q_table)
    else:
        q_agent = QAgent(env, alpha=start_alpha, gamma=start_gamma, epsilon=start_epsilon)

    if not load_from_file:
        # metrics
        total_rewards = []  # every 100 epochs
        total_steps = []    # every 100 epochs

        epochs = 15000
        for i in range(0, epochs):
            steps, cum_reward = q_agent.q_learning()
            if i % 100 == 0 or i == 0:
                steps, total_reward = q_agent.validation()
                total_rewards.append(cum_reward)
                total_steps.append(steps)
                print(i, epochs, cum_reward)
                if decay:
                    # change alpha, beta, gamma
                    # https://www.youtube.com/watch?v=QzulmoOg2JE
                    new_alpha = 1 / (1 + decay_rate * i) * start_alpha
                    # linear increase of gamma
                    new_gamma = (1 - start_gamma) / epochs * i + start_gamma
                    q_agent.gamma = new_gamma
                    new_epsilon = 1 / (1 + decay_rate * i) * start_epsilon
                    q_agent.alpha = new_alpha
                    q_agent.epsilon = new_epsilon
                    print(new_alpha, new_gamma, new_epsilon)

        # pickle files
        file = open('q_table', 'wb')
        # dump information to that file
        pickle.dump(q_agent.q_table, file)
         # close the file
        file.close()

    # show graphs
        x = np.arange(0, epochs, 100)
        utils.plot(x, y=total_rewards, x_label='epochs of training', y_label='cummulative reward')
        utils.plot(x, y=total_steps, x_label='epochs of training', y_label='steps needed')

    # end of not loading_from_file
    # after training show agent 5 times
    input("Press Enter to continue...")
    os.system('clear')
    print(q_agent.q_table)
    for i in range(0, 5):
        steps, cum_reward = q_agent.validation(sleep_timer=0.25)
        print("needed steps", steps, "cum_reward", cum_reward)
        time.sleep(1)
    print('finished')


if __name__ == '__main__':
    main(sys.argv[1:])
