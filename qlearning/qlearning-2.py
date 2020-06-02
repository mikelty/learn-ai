"""
one completion / reward is needed for sensible future action
in q-learning.
"""
import numpy as np
import gym

debug=False
class State():
    POSITION=0
    VELOCITY=1

env=gym.make("MountainCar-v0")

#anything from 0 to 1.
LEARNING_RATE=0.1
#how important do we find future actions?
#some sorta markov chain-like geometric decay...
DISCOUNT=0.95
#this is not that long as long as we don't render all of them XD
EPISODES=25000

#print metrics every 2000 episodes
SHOW_EVERY=2000

DISCRETE_OS_SIZE=[20] * len(env.observation_space.high)
discrete_os_win_size=(env.observation_space.high-env.observation_space.low)/DISCRETE_OS_SIZE

#0 to 1, exploration constant (perhaps like k in simulated anneling?)
epsilon=0.5
START_EPSILON_DECAYING=1
END_EPSILON_DECAYING=EPISODES//2
epsion_decay_value=epsilon/(END_EPSILON_DECAYING-START_EPSILON_DECAYING)

q_table=np.random.uniform(low=-2,high=0,size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    """

    :param state:
    :return:
    """
    discrete_state=(state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

"""
discrete_state = get_discrete_state(env.reset())
if debug:
    print(discrete_state)
    #which action to take
    print(np.argmax(q_table[discrete_state]))
"""

done=False

for episode in range(EPISODES):
    render=episode % SHOW_EVERY == 0
    if episode % 100 == 0:
        print(episode)
    discrete_state = get_discrete_state(env.reset())
    done=False
    while not done:
        if np.random.random()>epsilon:
            action=np.argmax(q_table[discrete_state])
        else:
            action=np.random.randint(0,env.action_space.n)
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            #q formula.
            #basically, finds a point set by learning rate
            #on the line (current_q, future_q * discout + rewards)
            new_q = (1 - LEARNING_RATE) * current_q \
                        + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            #update discrete state after taking the step
            q_table[discrete_state+(action,)]=new_q
        elif new_state[State.POSITION] >= env.goal_position:
            #reward for completing the task
            print(f"first success  on episode {episode}")
            env.close()
            exit(0)
            q_table[discrete_state+(action,)]=0
        discrete_state = new_discrete_state
    if END_EPSILON_DECAYING>=episode>=START_EPSILON_DECAYING:
        epsilon-=epsion_decay_value

env.close()
