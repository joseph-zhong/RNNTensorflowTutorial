#!/usr/local/bin/python
""" 
train_dnn.py
---

Main script for training.

"""

import tensorflow as tf

import config
import rnn

def main(args):
  print "[main] Starting App."

  print "[main] Initializing RNN."
  Rnn = rnn.RNN()

  print "[main] Training RNN."
  Rnn.train()

  print "[main] Training Done!"

if __name__ == "__main__":
  tf.app.run()

