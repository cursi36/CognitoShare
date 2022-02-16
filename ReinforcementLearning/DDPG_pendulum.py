import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

problem = "Pendulum-v1"
env = gym.make(problem)

num_states = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1




class ActorCritic():
    def __init__(self,num_states,num_actions,hidden_actor=[256,256]):

        self.actor = self.getActor(num_states,hidden_actor)
        self.critic = self.getCritic(num_states, num_actions)

        self.target_actor = self.getActor(num_states,hidden_actor)
        self.target_critic = self.getCritic(num_states, num_actions)

        self.setTargetWeights()

        self.critic_loss = tf.keras.losses.MeanSquaredError()

        critic_lr = 0.002
        actor_lr = 0.001

        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

    def setTargetWeights(self):
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

    def getActor(self,num_states,hidden_sizes=[256,256]):
        inputs = layers.Input(shape=(num_states,))

        x = inputs
        for n in hidden_sizes:
            x = layers.Dense(n, activation="relu")(x)

        outputs = layers.Dense(1, activation="tanh")(x)
        # Our upper bound is 2.0 for Pendulum.
        outputs = outputs * upper_bound
        model = tf.keras.Model(inputs, outputs)

        return model

    def getCritic(self,num_states,num_actions):

        # State as input
        state_input = layers.Input(shape=(num_states))
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=(num_actions))
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    @tf.function
    def update_target(self, tau):
        #update weights of target actor
        for (a, b) in zip(self.target_actor.variables, self.actor.variables):
            a.assign(b * tau + a * (1 - tau)) #assign value to reference
        #update weights of critic
        for (a, b) in zip(self.target_critic.variables, self.critic.variables):
            a.assign(b * tau + a * (1 - tau)) #assign value to reference

    def GetActionPolicy(self,state):
        #it is a deteministic policy
        action = self.actor(state)
        action = action[0].numpy()

        # We make sure action is within bounds
        legal_action = np.clip(action, lower_bound, upper_bound)

        return legal_action

    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        #update critic
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic([state_batch, action_batch], training=True)
            # critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
            critic_loss = self.critic_loss(y,critic_value)

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables)
        )

        #update actor
        with tf.GradientTape() as tape:
            actions = self.actor(state_batch, training=True)
            critic_value = self.critic([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self,buffer):
        # Get sampling range
        record_range = min(buffer.buffer_counter, buffer.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, buffer.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(buffer.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(buffer.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(buffer.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(buffer.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


ACNet = ActorCritic(num_states,num_actions,hidden_actor=[256,256])

total_episodes = 100
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

buffer = Buffer(50000, 64)

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

# Takes about 4 min to train
for ep in range(total_episodes):

    prev_state = env.reset()
    episodic_reward = 0

    while True:
        # Uncomment this to see the Actor in action
        # But not in a python notebook.
        env.render()

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = ACNet.GetActionPolicy(tf_prev_state)
        # Recieve state and reward from environment.
        state, reward, done, info = env.step(action)

        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        ACNet.learn(buffer)
        ACNet.update_target(tau)

        # End this episode when `done` is True
        if done:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list)
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()
