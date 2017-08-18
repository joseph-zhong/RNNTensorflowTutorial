"""
rnn.py
---

Defines the RNN Tensorflow Graph.

"""

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
    self.input_sequence = tf.unpack(x, axis=1)
    self.label_sequence = tf.unpack(y, axis=1)

  def forward_pass(self):
    current_state = self.init_state
    states_sequence = []

    for inp in self.input_sequence:
      current_input = tf.reshape(inp, [FLAGS.batch_size, 1])
      input_state = tf.concat(1, [current_input, current_state])

      # Compute linear combination of current input and Wa, and current state and Wb
      next_state = tf.tanh(tf.matmul(input_state, self.w) + self.b)
      states_sequence.append(next_state)
      current_state = next_state

