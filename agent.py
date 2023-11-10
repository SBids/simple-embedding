# Try using different reinforcement learning algorithms, such as Proximal Policy Optimization (PPO) or Deep Q-Networks (DQN), 
# and compare their performance to the Actor-Critic algorithm.

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from network import ActorCriticNetwork

class Agent:
  def __init__(self, alpha=0.0003, gamma=0.99, n_actions=1):
    self.gamma = gamma
    self.n_actions = n_actions
    self.action = None          #keep track of last action
    self.action_space = [0, 10] #random action selection
    self.actor_critic = ActorCriticNetwork(n_actions=n_actions)
    self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))


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

  # learn the policy
  def learn(self, state, reward, state_, done):
    state = tf.convert_to_tensor([state], dtype=tf.float32)

  
    reward = tf.convert_to_tensor([reward], dtype=tf.float32)
    
    done = tf.convert_to_tensor([int(done)], dtype=tf.float32)
    # print("Previous state tensor", state)
    # print("New tensor of state", state_)
      
    '''
    GradientTape is a TensorFlow tool that is used for automatic differentiation. 
    It allows you to record operations that are performed on tensors, and then 
    compute the gradients of any variables with respect to the recorded operations.

    GradientTape is often used to compute the gradients of the policy and value functions 
    with respect to the loss function, which allows the weights of the neural network 
    to be updated in a direction that improves the performance of the agent.
    '''

    with tf.GradientTape() as tape:
      value, mu, sigma = self.actor_critic(state)
      value_, _, _ = self.actor_critic(state_)
     

      td_error = reward + self.gamma * value_ * (1 - done) -value
      advantage = td_error

      dist = tfp.distributions.Normal(loc=mu, scale=sigma)
      log_prob = dist.log_prob(self.action)
      actor_loss = -log_prob * tf.stop_gradient(advantage)
      critic_loss = tf.square(td_error)

      loss = actor_loss + critic_loss
    # Compute gradients and update weights
    gradients = tape.gradient(loss, self.actor_critic.trainable_variables)
    self.actor_critic.optimizer.apply_gradients(zip(gradients, self.actor_critic.trainable_variables))
  




