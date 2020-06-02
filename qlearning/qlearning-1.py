#from https://www.youtube.com/watch?v=yMk_XtIEzH8
#goal: push the cart up the yellow flag area
"""
idea of QL (Q-learning):
    build q table that reacts with (position, velocity)
    combinations. init with random values.
    agent (robot cart) explores the space and gets
    better at climbing up the hill...
    then it uses q function to update the q-table,
    until it is great.
problem of QL:
    notices how the raw state parameters have 8 digits
    of precision after decimal. this blows memory up!
solution:
    discretize states! (naturally)
"""

import numpy as np
import gym

debug=True

env=gym.make("MountainCar-v0")
env.reset()

#note this information (i.e. q-table's space) is
# sometimes not known, while action space is known.
if debug:
    print(f"observation space: {env.observation_space.low} " \
          f"- {env.observation_space.high} " \
          f", size: {env.action_space.n}")

#this is not hard coded for a real RL agent...
#OS is observation space
DISCRETE_OS_SIZE=[20] * len(env.observation_space.high)
discrete_os_win_size=(env.observation_space.high-env.observation_space.low)/DISCRETE_OS_SIZE

#in practice one should tinker these variables
#here q_table's dimension is 20 by 20 by 3, storing a q-value for each scenario.
q_table=np.random.uniform(low=-2,high=0,size=(DISCRETE_OS_SIZE + [env.action_space.n]))
if debug:
    print(q_table.shape)

done=False

while not done:
    action=2 #push cart to right
    #state = (position, velocity)
    #reward is -1 until you reach flag, and by then will be 0.
    new_state, reward, done, _ = env.step(action)
    if debug:
        print(reward,new_state)
    env.step(action)
    env.render()

#after done
env.close()