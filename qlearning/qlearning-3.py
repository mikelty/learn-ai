"""
"""
import numpy as np
import gym
import matplotlib.pyplot as plt

debug=False
class State():
    POSITION=0
    VELOCITY=1

env=gym.make("MountainCar-v0")

LEARNING_RATE=0.1
DISCOUNT=0.95
EPISODES=2000
SHOW_EVERY=500

DISCRETE_OS_SIZE=[20] * len(env.observation_space.high)
discrete_os_win_size=(env.observation_space.high-env.observation_space.low)/DISCRETE_OS_SIZE

epsilon=0.5
START_EPSILON_DECAYING=1
END_EPSILON_DECAYING=EPISODES//2
epsion_decay_value=epsilon/(END_EPSILON_DECAYING-START_EPSILON_DECAYING)

q_table=np.random.uniform(low=-2,high=0,size=(DISCRETE_OS_SIZE + [env.action_space.n]))

#tracking structures
ep_rewards=[]
#for each episode, what's the average, worst, or best agent
#sometimes one prefers a decent bound for worst model
aggr_ep_rewards={'ep':[],'avg':[],'min':[],'max':[]}

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
    episode_reward=0
    render=episode % SHOW_EVERY == 0
    discrete_state = get_discrete_state(env.reset())
    done=False
    while not done:
        if np.random.random()>epsilon:
            action=np.argmax(q_table[discrete_state])
        else:
            action=np.random.randint(0,env.action_space.n)
        new_state, reward, done, _ = env.step(action)
        episode_reward+=reward
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1 - LEARNING_RATE) * current_q \
                    + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state+(action,)]=new_q
        elif new_state[State.POSITION] >= env.goal_position:
            q_table[discrete_state+(action,)]=0
        discrete_state = new_discrete_state
    if END_EPSILON_DECAYING>=episode>=START_EPSILON_DECAYING:
        epsilon-=epsion_decay_value
    ep_rewards.append(episode_reward)
    if render:
#        np.save(f"qtables\\{episode}-qtable.npy",q_table)
        average_reward=sum(ep_rewards[-SHOW_EVERY:])/SHOW_EVERY
        min_reward=min(ep_rewards[-SHOW_EVERY:])
        max_reward=max(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min_reward)
        aggr_ep_rewards['max'].append(max_reward)

        print(f"Episode: {episode}; Average: {average_reward}, Worst: {min_reward}; Best: {max_reward}")

env.close()

plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['avg'],label="avg")
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['min'],label="min")
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['max'],label="max")
plt.legend(loc=4)
plt.show()
