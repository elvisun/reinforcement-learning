"""
Author: Uriel Sade
Date: July 2nd, 2017
"""
import tensorflow as tf
from dqn.replay_memory import ReplayMemory
import os

class NeuralNet:

    """
    Deep-Q-Network used to approximate the function Q(s, a), similar to the one
    in the original DeepMind paper with experience replay and e-greedy policy.

    Args:
        W: The width of the image passed in as the input layer
        H: The height of the image passed in as the input layer
        N_ACTIONS: number of possible actions. A q-value is predicted for each
                   action in each state
        learning_rate: The optimizer's learning rate
        game_title: name of the game being played
        replay_memory: experience replay memory used for random minibatch
                       optimization
    """
    def __init__(self, W, H, N_ACTIONS, learning_rate, game_title, replay_memory=None):
        self.replay_memory = replay_memory if replay_memory else ReplayMemory()
        self.input_layer = tf.placeholder(shape=[None, W, H, 1], dtype=tf.float32, name="input_layer")
        self.lr = learning_rate
        self.q_values_next = tf.placeholder(shape=[None, N_ACTIONS], dtype=tf.float32, name="q_next")
        self.checkpoint_dir = "../saved_checkpoints/"

        weight_init = tf.truncated_normal_initializer(mean=0, stddev=0.03)
        activation  = tf.nn.relu
        padding     = 'SAME'
        conv2d = tf.layers.conv2d
        dense = tf.layers.dense
        flatten = tf.contrib.layers.flatten

        # 1st conv layer
        NN = conv2d(name="conv1", inputs=self.input_layer,
                    filters=16, kernel_size=3, strides=2,
                    padding=padding, kernel_initializer=weight_init,
                    activation=activation)
        # 2nd conv layer
        NN = conv2d(name="conv2", inputs=NN,
                    filters=32, kernel_size=3, strides=2,
                    padding=padding, kernel_initializer=weight_init,
                    activation=activation)

        # 3rd conv layer
        NN = conv2d(name="conv3", inputs=NN,
                    filters=64, kernel_size=3, strides=1,
                    padding=padding, kernel_initializer=weight_init,
                    activation=activation)

        # Flatten the output of the 3rd convolutional layer to input into fc layer
        NN = flatten(NN)

        # 1st fully connected layer
        NN = dense( name='fc1', inputs=NN, units=1024,
                    kernel_initializer=weight_init,
                    activation=activation)
        # 2nd fc layer
        NN = dense( name='fc2', inputs=NN, units=1024,
                    kernel_initializer=weight_init,
                    activation=activation)
        # Output layer
        NN = dense( name='fc_out', inputs=NN, units=N_ACTIONS,
                    kernel_initializer=weight_init,
                    activation=None)

        self.q_values = NN

        error = tf.square(self.q_values - self.q_values_next)
        self.loss = tf.reduce_mean(tf.reduce_sum(error, axis=1))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        self.sess  = tf.InteractiveSession()
        self.saver = tf.train.Saver()

        try:
            last_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir + game_title + "/")
            self.saver.restore(self.sess, save_path=last_checkpoint)
        except:
            self.sess.run(tf.global_variables_initializer())
    """
    Saves the current session checkpoint
    """
    def save(self):
        self.saver.save(self.sess, save_path=self.checkpoint_dir)

    """
    Closes the TensorFlow session being used
    """
    def close_session(self):
        self.sess.close()

    """
    Q(state, network_params) -> (q_a1, q_a2, ..., q_an)
    Args:
        states: A batch of game states that are inputs to the neural net
    Returns:
        predicted q-value vector for each state. Each vector contains the
        predicted q-values for each action
    """
    def Q(self, states):
        return self.sess.run(self.q_values, feed_dict={self.input_layer:states})

    """
    Optimize on the equation:
    [reward + GAMMA * max_a'(Q(s', a')) - Q(s, a)]^2
    Args: ...
    """
    def optimize(self, args=...):
        #TODO
        ...
