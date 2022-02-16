import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tqdm import tqdm
"""
The cart pole has passive revolute joint bwteen cart and pole
Action space in [0,1] where 0 = push cart to left, 1 to tight (with fixed amount of force of 10 N)
Observations: - Crat position; - Crat Velocity; - Pole angle ; - Pole Angular Velocity

An episode ends if cart position not in (-2.4 2.4); pole angle not in (-12 12) deg; epsiode length > 500
"""


seed = 42
gamma = 0.99 #discount factor past rewards
max_steps_per_episode = 1000

env = gym.make("CartPole-v0")

env.seed(seed)
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

"""
#---- Actor Critic Implementation
The Actor takes as input the sate of the environmtnt (observations) and returns a probability
for each action taken

The Critic takes observations as inputs and return estimaton of total future rewards

"""

num_inputs = 4 #The observations
num_actions = 2 #0,1
num_hidden = 128 #hidden layers


class ActorCritic():
    def __init__(self,input_dim,num_hidden, output_size):
        super(ActorCritic,self).__init__()

        inputs = tf.keras.layers.Input(shape=(num_inputs,))
        hidden = tf.keras.layers.Dense(num_hidden, activation="relu")(inputs)
        action = tf.keras.layers.Dense(num_actions, activation="softmax")(hidden)
        critic = tf.keras.layers.Dense(1)(hidden)

        self.model = tf.keras.Model(inputs=inputs,outputs=[action,critic])

#----Train netwroks
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.huber_loss = tf.keras.losses.Huber()

    def update(self,history):
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                self.huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def predict(self,state):
        return self.model(state)

action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

opt_actions = []

model = ActorCritic(num_inputs,num_hidden, num_actions)

while True:  # Run until solved
    state = env.reset()
    episode_reward = 0
    with tf.GradientTape() as tape:
        for timestep,tqdm_iter in zip(range(1, max_steps_per_episode),tqdm(range(1,max_steps_per_episode-1))):
            env.render(); #Adding this line would show the attempts
            # of the agent in a pop up window.

            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0) #reshpes to (1,-1)

            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = model.predict(state)
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            # returns a single value for the actions. The possible actions are [0,1]
            #  given their probabilities, the function returns a random value weighted by their probability
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(tf.math.log(action_probs[0, action]))


            # Apply the sampled action in our environment
            state, reward, done, _ = env.step(action)
            rewards_history.append(reward)
            episode_reward += reward

            if done:
                break

        running_reward = episode_reward

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]: #loop list of rewards from end (mst recent reward)
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        model.update(history)

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

        # Log details
        episode_count += 1
        if episode_count % 10 == 0:
            template = "running reward: {:.2f} at episode {}"
            print(template.format(running_reward, episode_count))

        if running_reward > 195:  # Condition to consider the task solved
            print("Solved at episode {}!".format(episode_count))
            print("running reward!", running_reward)

            break

