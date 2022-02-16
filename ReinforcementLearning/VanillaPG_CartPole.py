import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp

from tqdm import tqdm

class PolicyNet():
    def __init__(self,input_dim,hidden_sizes,out_dim=1,discrete=False):

        inputs = tf.keras.layers.Input(shape=input_dim)  # input dimension
        x = inputs
        for n in hidden_sizes:
            x = tf.keras.layers.Dense(n, activation="relu")(x)

        if not discrete:
            mu = tf.keras.layers.Dense(out_dim, activation="linear")(x)
            sigma = tf.keras.layers.Dense(out_dim, activation="softplus")(x)

            self.model = tf.keras.Model(inputs=inputs, outputs=[mu, sigma])

        else:

            prob = tf.keras.layers.Dense(out_dim, activation="softmax")(x)
            self.model = tf.keras.Model(inputs=inputs, outputs=prob)


        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


    def update(self,state_history,action_history,reward_history,
               discrete,gamma):

        discounted_rewards = []
        sum_rewards = 0

        for reward in reward_history[::-1]:
            sum_rewards = reward + gamma * sum_rewards
            discounted_rewards.append(sum_rewards)

        discounted_rewards.reverse()
        history = zip(state_hist, action_hist, discounted_rewards)


        for state,action,reward in history:
            with tf.GradientTape() as tape:

                state = tf.expand_dims(state, 0)

                if not discrete:
                    mu, sigma = self.model(state)
                    # get gaussian distribution
                    distr = tfp.distributions.Normal(mu, sigma)

                else:
                    prob = self.model(state)
                    distr = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)

                # get probability of action, given gaussian distr
                log_prob = distr.log_prob(action)
                loss_value = -abs(reward)*log_prob

                grads = tape.gradient(loss_value, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


# env = gym.make("Pendulum-v1")
# state_dim = 3 #cos angle, sin angle, angular vel
# out_dim = 1
# discrete=False

env = gym.make("CartPole-v0")
discrete=True
state_dim = 4 #cos angle, sin angle, angular vel
out_dim = 2

hidden_sizes = [64,64]
PNet = PolicyNet(state_dim,hidden_sizes,out_dim=out_dim,discrete=discrete)

MaxEpisodes = 10000
gamma = 0.95

for episode in range(MaxEpisodes):
    state = env.reset()
    done = False
    r = 0
    steps = 0

    reward_hist = []
    state_hist = []
    action_hist = []

    while not done:
        env.render()

        curr_state = state
        state = tf.expand_dims(state, 0)
        if not discrete:
            mu, sigma = PNet.model(state)
            action = np.random.normal(mu[0].numpy(), sigma[0].numpy(), 1)
            action = np.clip(action,-2,2)

        else:
            probs = PNet.model(state).numpy()
            action = np.random.choice(a=[0, 1],size = 1,p = probs[0,:])
            action = action[0]

        action_hist.append(action)
        state_hist.append(curr_state)

        state, reward, done, _ = env.step(action)

        reward_hist.append(reward)

        r = r+reward
        steps = steps+1

    PNet.update(state_hist,action_hist,reward_hist, discrete,gamma)
    print("eisode:", episode," reward:", r)






