import os
from time import sleep
import seaborn as sns # for data visualization
import pandas as pd # for data analysis
import matplotlib.pyplot as plt # for data visualization

def print_step(steps, state, action, reward, cum_reward):
    print('step:', steps)
    print('state:', state)
    print('action:', action)
    print('reward for this action:', reward)
    print('cumulative reward:', cum_reward)

def plot(x,y, x_label='t', y_label='y'):
    # x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # y = [36.6, 37, 37.7, 39, 40.1, 43, 43.4, 45, 45.6, 40.1, 44, 45, 46.8, 47, 47.8]
    # x_label = 't'
    # y_label = 'reward'

    # create dataframe using two list days and temperature
    temp_df = pd.DataFrame({x_label: x, y_label: y})

    # Draw line plot
    sns.lineplot(x=x_label, y=y_label, data=temp_df, )
    plt.show()  # to show graph