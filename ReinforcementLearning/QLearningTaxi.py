import random
import gym
import time
import numpy as np

env = gym.make("Taxi-v3")


q_table = np.zeros([env.observation_space.n, env.action_space.n])

alpha = 0.1
gamma = 0.6
epsilon = 0.1
MaxEpisodes = 100000

def greedy_policy(state,epsilon):
    if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()  # Explore action space
    else:
        action = np.argmax(q_table[state])  # Exploit learned values

    return action

for i in range(1, MaxEpisodes):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False

    r = 0

    while not done:

        action = greedy_policy(state,epsilon)

        next_state, reward, done, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        r = r+reward

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

    # if i % 100 == 0:
    #     print(f"Episode: {i}")

    print("Episode ",i," reward:", reward)

print("Training finished.\n")

time.sleep(2)

total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0

    done = False
    print("episode:", i)
    while not done:
        env.render()

        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")


