import random
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from network import ActorCriticNetwork
from collections import deque
import time


class Agent:
    def __init__(self, alpha=0.0003, gamma=0.99, n_actions=1):
        self.gamma = gamma
        self.n_actions = n_actions
        self.action = None
        self.action_space = [0, 10]
        self.actor_critic = ActorCriticNetwork(n_actions=n_actions)
        self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))

        # Define a replay buffer
        # self.replay_buffer = deque(maxlen=10000)  # Adjust the buffer size as needed

    def choose_action(self, observation):
        # state = tf.convert_to_tensor([observation])
        _, mu, sigma = self.actor_critic(observation)

        # Create a normal distribution with the calculated mean (mu) and standard deviation (sigma)
        action_distribution = tfp.distributions.Normal(loc=mu, scale=sigma)

        # Sample an action from the distribution
        raw_action = action_distribution.sample()
        action = tf.clip_by_value(raw_action, clip_value_min=0.0, clip_value_max=tf.float32.max)

        # Calculate the log probability of the chosen action
        log_prob = action_distribution.log_prob(action)
        self.action = action

        # Store the chosen action in the instance variable
        action = action[0]

        return action.numpy()[0]

    def save_models(self):
        print('... saving models ...')
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor_critic.load_weights(self.actor_critic.checkpoint_file)

    def get_reward(self, ewma_duplicate_count, rtt, srtt, alpha=0.5, beta=0.4, gamma=0.3):
        if rtt is not None:
            reward = -(alpha * float(ewma_duplicate_count) + beta * rtt + gamma * (rtt - srtt))
        else:
            reward = -(alpha * float(ewma_duplicate_count) + gamma * srtt)
        return reward



    def train_from_buffer(self, batch_size, replay_buffer, count_training):
        if len(replay_buffer)%batch_size == 0:
            
            print("Training with buffer ", count_training)
            start_training = time.time()
            batch = random.sample(replay_buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            # states = tf.convert_to_tensor(states, dtype=tf.float32)
            # actions = tf.convert_to_tensor(actions, dtype=tf.int32)
            # rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            # next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
            # dones = tf.convert_to_tensor(dones, dtype=tf.float32)

            with tf.GradientTape() as tape:
                value, mu, sigma = self.actor_critic(states)
                value_, _, _ = self.actor_critic(next_states)
                td_error = rewards + self.gamma * value_ * (1 - dones) - value
                advantage = td_error

                dist = tfp.distributions.Normal(loc=mu, scale=sigma)
                log_prob = dist.log_prob(actions)
                actor_loss = -log_prob * tf.stop_gradient(advantage)
                critic_loss = tf.square(td_error)

                loss = actor_loss + critic_loss

            gradients = tape.gradient(loss, self.actor_critic.trainable_variables)
            self.actor_critic.optimizer.apply_gradients(zip(gradients, self.actor_critic.trainable_variables))
            end_training = time.time()
            print("Training period ", (end_training - start_training)*1000)
            count_training = count_training + 1
            print("Count lerarning increased", count_training)
     

