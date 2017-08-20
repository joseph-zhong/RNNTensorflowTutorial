"""
data.py
---

Data manipulation.

"""

import numpy as np
import tensorflow as tf

import config

# Define global Tensorflow variables.
FLAGS = tf.app.flags.FLAGS

def generateTrainData():
  """ Generates a random binary vector as training data. """
  x = np.array(np.random.choice(2, FLAGS.total_series_length))
  y = np.roll(x, FLAGS.echo_step)
  y[0:FLAGS.echo_step] = 0

  return x.reshape((FLAGS.batch_size, -1)), y.reshape((FLAGS.batch_size, -1))

def generateValidData():
  """ Generates a random binary vector as validation data. """
  x = np.array(np.random.choice(2, FLAGS.total_series_length))
  y = np.roll(x, FLAGS.echo_step)
  y[0:FLAGS.echo_step] = 0

  return x.reshape((FLAGS.batch_size, -1)), y.reshape((FLAGS.batch_size, -1))





