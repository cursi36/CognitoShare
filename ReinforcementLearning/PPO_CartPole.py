import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
import scipy.signal
import time


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    discounted_rewards = []
    sum_rewards = 0
    for reward in x[::-1]:
        sum_rewards = reward + discount * sum_rewards
        discounted_rewards.append(sum_rewards)

    discounted_rewards.reverse()
    return np.array(discounted_rewards)
    # return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
    # Buffer for storing trajectories
    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        # Buffer initialization
        self.observation_buffer = np.zeros(
            (size, observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        #Compute TD error
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        #Compute the discounted advantage
        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )

        # Compute the discounted rewards
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )


class ActorCritic():
    def __init__(self,num_states,hidden_sizes,num_actions):

        self.actor = self.GetNet(num_states,hidden_sizes,num_actions)
        self.critic = self.GetNet(num_states, hidden_sizes, 1)

        self.critic_loss = tf.keras.losses.MeanSquaredError()

        policy_learning_rate = 3e-4
        value_function_learning_rate = 1e-3

        # Initialize the policy and the value function optimizers
        self.policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
        self.value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)

    def GetNet(self,input_dim,hidden_sizes,output_dim):
        input = keras.Input(shape=input_dim, dtype=tf.float32)
        x = input
        for size in hidden_sizes:
            x = layers.Dense(units=size, activation=tf.tanh)(x)
        output = layers.Dense(units=output_dim)(x)
        return keras.Model(inputs=input, outputs=output)

    # Train the value function by regression on mean-squared error
    @tf.function
    def train_value_function(self,observation_buffer, return_buffer):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            # value_loss = tf.reduce_mean((return_buffer - self.critic(observation_buffer)) ** 2)
            value_loss = self.critic_loss(return_buffer,self.critic(observation_buffer))
        value_grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_grads, self.critic.trainable_variables))

    # Train the policy by maxizing the PPO-Clip objective
    @tf.function
    def train_policy(self,
            observation_buffer, action_buffer, logprobability_buffer, advantage_buffer,clip_ratio
    ):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            logits = self.actor(observation_buffer)
            ratio = tf.exp(
                self.logprobabilities(logits, action_buffer)
                - logprobability_buffer
            )
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + clip_ratio) * advantage_buffer,
                (1 - clip_ratio) * advantage_buffer,
            )

            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantage_buffer, min_advantage)
            )
        policy_grads = tape.gradient(policy_loss, self.actor.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.actor.trainable_variables))

        logits = self.actor(observation_buffer)
        kl = tf.reduce_mean(
            logprobability_buffer
            - self.logprobabilities(logits, action_buffer)
        )
        kl = tf.reduce_sum(kl)
        return kl

    # Sample action from actor
    @tf.function
    def sample_action(self,observation):
        logits = self.actor(observation)
        # samples = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 5)
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
        return logits, action

    def logprobabilities(self,logits, a):
        # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
        logprobabilities_all = tf.nn.log_softmax(logits)
        #same but won't work with eager tensor
        # act = a[0].numpy()
        # return logprobabilities_all[0, act]

        logprobability = tf.reduce_sum(
        tf.one_hot(a, num_actions) * logprobabilities_all, axis=1
        )
        return logprobability

# Hyperparameters of the PPO algorithm
steps_per_epoch = 4000
epochs = 30
gamma = 0.99
clip_ratio = 0.2
train_policy_iterations = 80
train_value_iterations = 80
lam = 0.97
target_kl = 0.01
hidden_sizes = [64, 64]


# Initialize the environment and get the dimensionality of the
# observation space and the number of possible actions
env = gym.make("CartPole-v0")
observation_dimensions = env.observation_space.shape[0]
num_actions = env.action_space.n

# Initialize the buffer
buffer = Buffer(observation_dimensions, steps_per_epoch)

# Initialize the actor and the critic as keras models
ACNet = ActorCritic(observation_dimensions,hidden_sizes,num_actions)

# Initialize the observation, episode return and episode length
state, episode_return, episode_length = env.reset(), 0, 0

# Iterate over the number of epochs
for epoch in range(epochs):
    # Initialize the sum of the returns, lengths and number of episodes for each epoch
    sum_return = 0
    sum_length = 0
    num_episodes = 0

    # Iterate over the steps of each epoch
    for t in range(steps_per_epoch):
        env.render()

        # Get the logits, action, and take one step in the environment
        state = state.reshape(1, -1)
        logits, action = ACNet.sample_action(state)
        state_new, reward, done, _ = env.step(action[0].numpy())
        episode_return += reward
        episode_length += 1

        # Get the value and log-probability of the action
        value_t = ACNet.critic(state)
        logprobability_t = ACNet.logprobabilities(logits, action)

        # Store obs, act, rew, v_t, logp_pi_t
        buffer.store(state, action, reward, value_t, logprobability_t)

        # Update the observation
        state = state_new

        # Finish trajectory if reached to a terminal state
        if done or (t == steps_per_epoch - 1):
            last_value = 0 if done else ACNet.critic(state.reshape(1, -1))
            buffer.finish_trajectory(last_value)
            sum_return += episode_return
            sum_length += episode_length
            num_episodes += 1
            state, episode_return, episode_length = env.reset(), 0, 0

    # Get values from the buffer
    (
        state_buffer,
        action_buffer,
        advantage_buffer,
        return_buffer,
        logprobability_buffer,
    ) = buffer.get()

    # Update the policy and implement early stopping using KL divergence
    for _ in range(train_policy_iterations):
        kl = ACNet.train_policy(
            state_buffer, action_buffer, logprobability_buffer, advantage_buffer,clip_ratio
        )
        if kl > 1.5 * target_kl:
            # Early Stopping
            break

    # Update the value function
    for _ in range(train_value_iterations):
        ACNet.train_value_function(state_buffer, return_buffer)

    # Print mean return and length for each epoch
    print(
        f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
    )