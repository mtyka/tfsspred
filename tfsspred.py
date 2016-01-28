"""Simple Secondary structure prediction with TensorFlow.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import argparse
import time
import random

import tensorflow.python.platform

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

RESIDUE_ALPHABET = 21
HHM_ALPHABET = 26
DSSP_ALPHABET = 8
NUM_FEATURES = 47
VALIDATION_SIZE = 40000
TEST_SIZE = 40000

SEED = 66478  # Set to None for random seed.
BATCH_SIZE =  512

tf.app.flags.DEFINE_string("dataset", "compacter.npz",
       'Input file contain training, validation and test data.')
tf.app.flags.DEFINE_integer("sequence_window", "15",
       'Size of the sequence input window. Must be odd number for symmetry.')
tf.app.flags.DEFINE_integer("dssp_window", "1",
       'Size of the sequence input window. Must be odd number for symmetry.')
tf.app.flags.DEFINE_integer("num_epochs", "20",
       'Num of epochs to train for.')
tf.app.flags.DEFINE_string('train_dir', '/tmp/',
      """Directory where to write event logs """
      """and checkpoint.""")
FLAGS = tf.app.flags.FLAGS


def success_rate(predictions, labels):
  """Return the error rate based on dense predictions and 1-hot labels."""
  return 100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0]

def success_rate_3state(predictions, labels):
  """Return the error rate based on prediction limited to 3-state E,H,C"""
  ## Collapse states 3-8 to 3
  predictions = np.roll(predictions,-1,axis=1)
  predictions_3state = np.hstack((predictions[:,0:2],np.amax(predictions[:,2:,None],1)))
  labels_3state = labels.copy()
  labels_3state[labels_3state==0]=8
  labels_3state -= 1
  labels_3state[labels_3state>=2]=2
  return 100.0 * np.sum(np.argmax(predictions_3state, 1) == labels_3state) / predictions_3state.shape[0]

def flatten(data):
  """Returns a reshaped tensor such that [a,b,c,...,z] dimensioned tensor becomes
     a [a, b*c*...*z] dimensioned one."""
  data_shape = data.get_shape().as_list()
  return tf.reshape(data, [data_shape[0], np.array(data_shape[1:]).prod()])

def slice_dataset(dataset, slice_param):
  result = {}
  for subset in ["sequence", "dssp", "hhm"]:
    result[subset] = dataset[subset][slice_param]
  return result

def make_windows(dataset, shuffled=False):
  hhm_features = dataset["hhm"].shape[1]
  end = dataset["sequence"].shape[0] - FLAGS.sequence_window
  seq_windows = np.zeros((end, FLAGS.sequence_window))
  hhm_windows = np.zeros((end, FLAGS.sequence_window, hhm_features))
  dssp_windows = np.zeros((end, FLAGS.dssp_window))
  offset = (FLAGS.sequence_window - FLAGS.dssp_window)/2
  seq_count = 0
  seq_pos = range(seq_windows.shape[0])
  if shuffled: random.shuffle(seq_pos)
  for i in seq_pos:
    # Ensure center position is actually a residue
    if dataset["sequence"][i+(FLAGS.sequence_window-1)//2] != 0:
      seq_windows[seq_count] = dataset["sequence"][i:i+FLAGS.sequence_window]
      hhm_windows[seq_count] = dataset["hhm"][i:i+FLAGS.sequence_window]
      dssp_windows[seq_count] = dataset["dssp"][i+offset:i+offset+FLAGS.dssp_window]
      seq_count += 1

  assert seq_windows.shape[1] == FLAGS.sequence_window
  assert hhm_windows.shape[1] == FLAGS.sequence_window
  assert hhm_windows.shape[2] == HHM_ALPHABET
  assert dssp_windows.shape[1] == FLAGS.dssp_window
  print("Created %d examples"%seq_count)
  return seq_windows[0:seq_count],hhm_windows[0:seq_count],dssp_windows[0:seq_count]

def to_one_hot_1D(labels, num_classes):
  num_labels = tf.size(labels)
  r_labels = tf.expand_dims(tf.cast(labels, tf.int32), 1)
  indices = tf.expand_dims(tf.range(0, num_labels, 1), 1)
  concated = tf.concat(1, [indices, r_labels])
  return tf.sparse_to_dense(concated, tf.pack([num_labels, num_classes]), 1.0, 0.0)

def to_one_hot_2D(labels, num_classes):
  labels_shape = labels.get_shape().as_list()
  r_labels = tf.reshape(labels, [-1])
  return tf.reshape(to_one_hot_1D(r_labels, num_classes), [labels_shape[0], labels_shape[1], num_classes])

def merge_sequence_with_hhm(sequence_data_sparse, hhm_data, train=False):
  sequence_data_dense = tf.cast(to_one_hot_2D(sequence_data_sparse, RESIDUE_ALPHABET), tf.float32)
  merged = tf.concat(2, [sequence_data_dense, hhm_data])
  return merged

def main(argv=None):  # pylint: disable=unused-argument
  # Extract it into np arrays.

  dataset = np.load(FLAGS.dataset)
  dataset_size = dataset["sequence"].size - TEST_SIZE - VALIDATION_SIZE
  print("Dataset size:", dataset_size)

  validation_set = slice_dataset(dataset, np.r_[TEST_SIZE:VALIDATION_SIZE+TEST_SIZE])
  validation_sequence, validation_hhm, validation_dssp = make_windows(validation_set)
  assert validation_sequence.shape[0] == validation_hhm.shape[0]
  assert validation_sequence.shape[0] == validation_dssp.shape[0]

  test_set = slice_dataset(dataset, np.r_[:TEST_SIZE])
  test_sequence, test_hhm, test_dssp = make_windows(test_set)
  assert test_sequence.shape[0] == test_hhm.shape[0]
  assert test_sequence.shape[0] == test_dssp.shape[0]

  train_sequence_node = tf.placeholder(tf.int32, shape=(BATCH_SIZE, FLAGS.sequence_window))
  train_hhm_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, FLAGS.sequence_window, HHM_ALPHABET))
  train_data_node = merge_sequence_with_hhm(train_sequence_node, train_hhm_node)
  train_dssp_node = tf.placeholder(tf.int32, shape=(BATCH_SIZE, FLAGS.dssp_window))
  train_labels_node = to_one_hot_2D(train_dssp_node, DSSP_ALPHABET)

  # For the validation and test data, we'll just hold the entire dataset in
  # one constant node.

  validation_sequence_node = tf.constant(validation_sequence, dtype=tf.int32)
  validation_hhm_node = tf.constant(validation_hhm, dtype=tf.float32)
  validation_data_node = merge_sequence_with_hhm(validation_sequence_node, validation_hhm_node)
  test_sequence_node = tf.constant(test_sequence, dtype=tf.int32)
  test_hhm_node = tf.constant(test_hhm, dtype=tf.float32)
  test_data_node = merge_sequence_with_hhm(test_sequence_node, test_hhm_node)

  # The variables below hold all the trainable weights. They are passed an
  # initial value which will be assigned when when we call:
  # {tf.initialize_all_variables().run()}
  conv1_filters = 128
  conv1_weights = tf.Variable(
      tf.truncated_normal([1, 3, NUM_FEATURES, conv1_filters],
                          stddev=0.1,
                          seed=SEED))
  conv1_biases = tf.Variable(tf.zeros([conv1_filters]))
  conv2_filters = 256
  conv2_weights = tf.Variable(
      tf.truncated_normal([1, 3, conv1_filters, conv2_filters],
                          stddev=0.1,
                          seed=SEED))
  conv2_biases = tf.Variable(tf.constant(0.1, shape=[conv2_filters]))
  conv3_filters = 256
  conv3_weights = tf.Variable(
      tf.truncated_normal([1, 3, conv2_filters, conv3_filters],
                          stddev=0.1,
                          seed=SEED))
  conv3_biases = tf.Variable(tf.constant(0.1, shape=[conv3_filters]))
  conv4_filters = 256
  conv4_weights = tf.Variable(
      tf.truncated_normal([1, 3, conv3_filters, conv4_filters],
                          stddev=0.1,
                          seed=SEED))
  conv4_biases = tf.Variable(tf.constant(0.1, shape=[conv4_filters]))

  fc1_features = FLAGS.sequence_window*conv4_filters
  fc1_nodes = 256
  fc1_weights = tf.Variable(
      tf.truncated_normal(
          [fc1_features, fc1_nodes],
          stddev=0.1,
          seed=SEED))
  fc1_biases = tf.Variable(tf.constant(0.1, shape=[fc1_nodes]))

  NUM_LABELS = DSSP_ALPHABET * FLAGS.dssp_window
  fc2_weights = tf.Variable(
      tf.truncated_normal([fc1_nodes, NUM_LABELS],
                          stddev=0.1,
                          seed=SEED))
  fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

  def model_fc(data, train=False):
    """FC layers."""
    hidden = tf.nn.relu(tf.matmul(flatten(data), fc1_weights) + fc1_biases)
    if train:
      hidden = tf.nn.dropout(hidden, 0.8, seed=SEED)
    hidden = tf.matmul(hidden, fc2_weights) + fc2_biases
    return hidden

  def model_conv(data, train=False):
    """Conv layers."""
    # Add a dimension 'y' with width 1 so we can use conv2d
    # (even though we're just doing 1d convolutions)
    data_shape = data.get_shape().as_list()
    data = tf.reshape(data, [data_shape[0], 1,  data_shape[1],  data_shape[2]])
    conv = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    conv = tf.nn.conv2d(relu, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    conv = tf.nn.conv2d(relu, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv3_biases))
    conv = tf.nn.conv2d(relu, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv4_biases))
    return relu

  def model(data, train=False):
    conv = model_conv(data, train)
    logits = model_fc(conv, train)
    return tf.reshape(logits, [-1, FLAGS.dssp_window, DSSP_ALPHABET])

  # Training computation: logits + cross-entropy loss.
  logits = model(train_data_node, True)

  # Do all DSSP_WINDOWS together into one loss (no differential weighting)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        tf.reshape(logits, [BATCH_SIZE * FLAGS.dssp_window, DSSP_ALPHABET]),
        tf.reshape(train_labels_node, [BATCH_SIZE * FLAGS.dssp_window, DSSP_ALPHABET])))

  # L2 regularization for the fully connected parameters.
  regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                  tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
  # Add the regularization term to the loss.
  loss += 5e-4 * regularizers

  # Optimizer: set up a variable that's incremented once per batch and
  # controls the learning rate decay.
  batch = tf.Variable(0)

  # Decay once per epoch, using an exponential schedule starting at 0.01.
  learning_rate = tf.train.exponential_decay(
      0.005,                # Base learning rate.
      batch * BATCH_SIZE,  # Current index into the dataset.
      1000000,             # Decay step.
      0.98,                # Decay rate.
      staircase=True)

  tf.scalar_summary('learning_rate', learning_rate)
  tf.scalar_summary('loss', loss)

  # Use simple momentum for the optimization.
  optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)

  # Predictions for the minibatch, validation set and test set.
  train_prediction = tf.nn.softmax(tf.squeeze(tf.slice(logits, [0, FLAGS.dssp_window//2, 0], [-1,1,-1])))
  # We'll compute them only once in a while by calling their {eval()} method.
  v1 = model(validation_data_node)
  validation_prediction = tf.nn.softmax(tf.squeeze(tf.slice(model(validation_data_node), [0, FLAGS.dssp_window//2, 0], [-1,1,-1])))
  test_prediction = tf.nn.softmax(tf.squeeze(tf.slice(model(test_data_node), [0, FLAGS.dssp_window//2, 0], [-1,1,-1])))

  # Build the summary operation based on the TF collection of Summaries.
  summary_op = tf.merge_all_summaries()

  # Create a local session to run this computation.
  with tf.Session() as s:
    # Run all the initializers to prepare the trainable parameters.
    tf.initialize_all_variables().run()
    print('Initialized!')

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                            graph_def=s.graph_def)

    total_train_examples = dataset_size - TEST_SIZE - VALIDATION_SIZE
    super_batch_size =  total_train_examples//4 + 5
    super_batches = range(0,total_train_examples, super_batch_size)
    mini_epochs =  4

    global_epoch = 0
    while global_epoch <= FLAGS.num_epochs:
      print("SuperEpoch: ", global_epoch)
      for super_offset in super_batches:
        super_batch_end = min(total_train_examples, super_offset+super_batch_size)
        print("SuperOffset: ", super_offset )
        print("SuperBatchSize: ", super_batch_end)

        training_set = slice_dataset(dataset, np.r_[TEST_SIZE+VALIDATION_SIZE+super_offset:TEST_SIZE+VALIDATION_SIZE+super_batch_end])
        training_sequence, training_hhm, training_dssp = make_windows(training_set)
        train_size = training_sequence.shape[0]
        print("TrainSize: ", train_size)

        for step in xrange(mini_epochs * train_size // BATCH_SIZE):
          offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
          feed_dict = {
            train_sequence_node :   training_sequence[offset:(offset + BATCH_SIZE)],
            train_hhm_node :        training_hhm[offset:(offset + BATCH_SIZE)],
            train_dssp_node :       training_dssp[offset:(offset + BATCH_SIZE)]
          }

          start_time = time.time()
          _, l, lr, predictions = s.run(
              [optimizer, loss, learning_rate, train_prediction],
              feed_dict=feed_dict)
          duration = time.time() - start_time

          #if step % 100 == 0:
          #  summary_str = s.run(summary_op)
          #  summary_writer.add_summary(summary_str, step)

          if step % 200 == 0:
            examples_per_sec = ((offset+BATCH_SIZE)%train_size-offset) / duration
            validation_result = validation_prediction.eval()
            print('Epoch %d loss: %.2f lrate: %.6f Validation success: %.1f%% 3state: %.1f%% %.0f ex/sec' % (
              global_epoch,
              l,lr,
              success_rate(validation_result, validation_dssp[:,FLAGS.dssp_window//2]),
              success_rate_3state(validation_result, validation_dssp[:,FLAGS.dssp_window//2]),
              examples_per_sec))
            sys.stdout.flush()
      global_epoch += mini_epochs
    # Finally print the result!
    test_result = test_prediction.eval()
    print('Test error: %.1f%%  %.1f%% ' %(success_rate(test_result, test_dssp[:,FLAGS.dssp_window//2]) ,
                                          success_rate_3state(test_result, test_dssp[:,FLAGS.dssp_window//2])    ))

if __name__ == '__main__':
  tf.app.run()

