"""
rnn.py
---

Defines the RNN Tensorflow Graph.

"""

import itertools
import os

import numpy as np
import tensorflow as tf

import config
import data

# Define global Tensorflow variables.
FLAGS = tf.app.flags.FLAGS

class RNN(object):
  def __init__(self):
    # Input placeholders.
    self.init_state = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.state_size])
    self.x = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.truncated_backprop_length])
    self.y = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.truncated_backprop_length])

    # Weights and biases.
    self.w = tf.Variable(np.random.rand(FLAGS.state_size + 1, FLAGS.state_size), dtype=tf.float32)
    self.b = tf.Variable(np.zeros((1, FLAGS.state_size)), dtype=tf.float32)

    self.w2 = tf.Variable(np.random.rand(FLAGS.state_size, FLAGS.num_classes), dtype=tf.float32)
    self.b2 = tf.Variable(np.zeros((1, FLAGS.num_classes)), dtype=tf.float32)

    # Input/Label Sequences.
    self.input_sequence = tf.unpack(self.x, axis=1)
    self.labels_seq = tf.unpack(self.y, axis=1)

    # Initial state for forward pass.
    self.current_state = self.init_state
    self.states_sequence = []

    # Initialize Loss.
    self._loss()

    # Initialize Tensorboard FileWriters.
    self._tensorboard()

  def _loss(self):
    print "[RNN._loss]"

    self.forward_pass()

    # Logits and Predictions.
    self.logits_seq = [tf.matmul(state, self.w2) + self.b2 for state in self.states_sequence]
    self.predictions = [tf.nn.softmax(logits) for logits in self.logits_seq]

    # Losses.
    self.losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
                   for logits, labels in itertools.izip(self.logits_seq, self.labels_seq)]
    self.total_loss = tf.reduce_mean(self.losses)

    # Training Optimizer.
    self.optimizer = tf.train.AdagradOptimizer(FLAGS.learning_rate).minimize(self.total_loss)

  def _tensorboard(self):
    print "[RNN._tensorboard]"
    # Training Visualization.
    self.loss_summary = tf.summary.scalar('Total Loss', self.total_loss)
    self.loss_writer_train = tf.summary.FileWriter(os.path.join(FLAGS.summaries_dir, 'train'))
    self.loss_writer_valid = tf.summary.FileWriter(os.path.join(FLAGS.summaries_dir, 'valid'))

  def forward_pass(self):
    print "[RNN.forward_pass]"
    for inp in self.input_sequence:
      current_input = tf.reshape(inp, [FLAGS.batch_size, 1])
      input_state = tf.concat(1, [current_input, self.current_state])

      # Compute linear combination of current input and Wa, and current state and Wb
      next_state = tf.tanh(tf.matmul(input_state, self.w) + self.b)
      self.states_sequence.append(next_state)
      self.current_state = next_state

  def train(self):
    with tf.Session() as sess:
      tf.initialize_all_variables()

      # Training epoch loop.
      for epoch in xrange(FLAGS.num_epochs):
        x, y = data.generateTrainData()

        # REVIEW josephz: Note that this is a standard initial state.
        # In other contexts, this may be the output of some other neural
        # network or preprocessor, such as a convolution or a separate,
        # independent recurrent network.
        curr_state = np.zeros((FLAGS.batch_size, FLAGS.state_size))

        # Training batch loop.
        # REVIEW josephz: This should be parallelized.
        for batch in xrange(FLAGS.batch_size):
          start = batch * FLAGS.truncated_backprop_length
          end = start + FLAGS.truncated_backprop_length

          batchX = x[:, start:end]
          batchY = y[:, start:end]

          total_loss, train_step, curr_state, predictions = sess.run(
            [self.total_loss, self.optimizer, self.current_state, self.predictions],
            feed_dict={
              self.x: batchX, self.y: batchY, self.init_state: curr_state
            }
          )

          # Tensorboard Training visualization.
          self.loss_writer_train.add_summary(total_loss)
          print "[Batch: {:04}] [Train Loss: {:04}]".format(batch, total_loss)

        # Tensorboard Validation visualization.
        if not batch % 10:
          x_v, y_v = data.generateValidData()
          valid_loss, curr_state, predictions = sess.run(
            [self.total_loss, self.current_state, self.predictions],
            feed_dict={
              self.x: x_v, self.y: y_v, self.init_state: curr_state
            }
          )
          self.loss_writer_valid.add_summary(valid_loss)
          print "[Batch: {:04}] [Valid Loss: {:04}]".format(batch, valid_loss)
