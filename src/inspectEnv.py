import gym

env = gym.make("Taxi-v3")
#env.render()

print("action space", env.action_space)
print("observation space", env.observation_space)

print("example action sample", env.action_space.sample())
print("example observation sample", env.observation_space.sample())

#env.reset()
# (taxi row, taxi column, passenger index, destination index)
#state = env.encode(0, 0, 0, 0)
#print("State:", state)
env.render()
state = env.s
decoded = list(env.decode(state)) # reversed iterator must be casted to list.. lul
print("State", state)
print("decoded", decoded)

#env.s = state

print(env.s, state)
# {action: [(probability, nextstate, reward, done)]}

# 5x5 positions
# 4 destinations
# 4+1 passenger locations
# => 5x5 pos * 4 dest * 5 locs = 500 => 500 states

# passengers want to drive from y -> r

print("Q-Table for State", state)
print("A [prob, next, R, done]")
for i in range(len(env.P[state])):
    print(i, env.P[state][i])


# every moevement -1
# pickup / drop off wrong -10
# drop off on the right loc +20
print("complete Q-Table")
#for i in range(len(env.P)):
#    print(env.P[i])