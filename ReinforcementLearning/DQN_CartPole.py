import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random

env = gym.envs.make("CartPole-v1")


class DQNet():
        def __init__(self, input_dim, h_sizes, output_dim):
            super(DQNet, self).__init__()

            inputs = tf.keras.layers.Input(shape=input_dim)
            x = inputs
            for k in range(len(h_sizes)):
                x = tf.keras.layers.Dense(h_sizes[k], activation="relu")(x)

            output = tf.keras.layers.Dense(output_dim)(x)

            self.model = tf.keras.Model(inputs=inputs, outputs=output)

            self.loss = tf.keras.losses.MeanSquaredError()
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        def update(self,state,y):
            with tf.GradientTape() as tape:

                y_pred = self.predict(state)

                loss_value = self.loss(y_pred,y)

                grads = tape.gradient(loss_value, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        def predict(self,state):
            return self.model(state)

class DQNet_replay(DQNet):

    def replay(self,memory,size,gamma=0.9):
        if len(memory) >= size:
            states = np.empty((0,4))
            targets = np.empty((0,2))

            #take a random set of memory of size=size
            batch = random.sample(memory,size)

            for state, action, next_state,reward,done in batch:
                states = np.vstack((states,state))

                q_values = self.predict(state).numpy()

                if done:
                    q_values[0,action] = reward
                else:
                    q_values_next = self.predict(next_state).numpy()
                    q_values[0, action] = reward + gamma * np.max(q_values_next)

                targets = np.vstack((targets, q_values))

            self.update(states,targets)

class DQNet_double(DQNet):
    def __init__(self, input_dim, h_sizes, output_dim):
        super().__init__(input_dim, h_sizes, output_dim)

        self.target = tf.keras.models.clone_model(self.model)

    def target_predict(self,state):
        return self.target(state)

    def target_update(self):
        self.target.set_weights(self.model.get_weights())

    def replay(self,memory,size,gamma=0.9):
        if len(memory) >= size:
            states = np.empty((0,4))
            targets = np.empty((0,2))

            #take a random set of memory of size=size
            batch = random.sample(memory,size)

            for state, action, next_state,reward,done in batch:
                states = np.vstack((states,state))

                q_values = self.predict(state).numpy()

                if done:
                    q_values[0,action] = reward
                else:
                    #the update values come from target network
                    q_values_next = self.target_predict(next_state).numpy()
                    q_values[0, action] = reward + gamma * np.max(q_values_next)

                targets = np.vstack((targets, q_values))

            self.update(states,targets)


final = []
memory = []

gamma = 0.9
epsilon = 0.1
eps_decay=0.99
MaxEpisodes = 1000
replay_size = 4
n_update = 10

replay = True
double_net = True

# model = DQNet(4,[64, 64*2],2)
# model = DQNet_replay(4,[64, 64*2],2)
model = DQNet_double(4,[64, 64*2],2)

for episode in range(MaxEpisodes):
    if double_net:
        # Update target network every n_update steps
        if episode % n_update == 0:
            model.target_update()
    # if double and soft:
    #     model.target_update()

    # Reset state
    state = env.reset()
    done = False
    total = 0

    state = tf.expand_dims(state, 0)

    while not done:
        env.render()

        # Implement greedy search policy
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = model.predict(state).numpy()
            action = np.argmax(q_values[0,:])

        # Take action and add reward to total
        next_state, reward, done, _ = env.step(action)
        next_state = tf.expand_dims(next_state, 0)

        # Update total and memory
        total += reward
        memory.append((state.numpy(), action, next_state.numpy(), reward, done))

        q_values = model.predict(state).numpy()

        if done:
            if not replay:
                q_values[0,action] = reward
                # Update network weights
                model.update(state, q_values)
            break

        if replay:
            # Update network weights using replay memory
            model.replay(memory, replay_size, gamma)
        else:
            # Update network weights using the last step only
            q_values_next = model.predict(next_state).numpy()
            q_values[0,action] = reward + gamma * np.max(q_values_next)

            model.update(state, q_values)

        state = next_state

        # Update epsilon
        epsilon = max(epsilon * eps_decay, 0.01)
        # final.append(total)
        # plot_res(final, title)

    print("Episode: ",episode," reward: ",total)