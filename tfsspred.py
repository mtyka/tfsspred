"""Simple Secondary structure prediction with TensorFlow.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import argparse

import tensorflow.python.platform

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

SEQUENCE_WINDOW = 21
NUM_FEATURES = 47
LABEL_WINDOW = 3
NUM_LABELS =  8
VALIDATION_SIZE = 1000  
TEST_SIZE = 1000  
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 1000

FLAGS = tf.app.flags.FLAGS

def extract_data(filename):
  print('Extracting', filename)
  f=gzip.open(filename,'rb')
  buf=f.read()
  data = numpy.frombuffer(buf, dtype=numpy.float32)
  print(data.shape)
  data = data.reshape((-1, 21, 47)) 
  print(data.shape)
  return data


def extract_labels(filename, num_images):
  """Extract the labels into a 1-hot matrix [image index, label index]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
  # Convert to dense 1-hot representation.
  return (numpy.arange(NUM_LABELS) == labels[:, None]).astype(numpy.float32)


def success_rate(predictions, labels):
  """Return the error rate based on dense predictions and 1-hot labels."""
  #print(predictions.shape, labels.shape)
  return 100.0 * numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) / predictions.shape[0]

def success_rate_3state(predictions, labels):
  """Return the error rate based on prediction limited to 3-state E,H,C"""
  ## Collapse states 3-8 to 3
  predictions = numpy.roll(predictions,-1,axis=1) 
  labels = numpy.roll(labels,-1,axis=1) 
  predictions_3state = numpy.hstack((predictions[:,0:2],numpy.amax(predictions[:,2:,None],1)))
  labels_3state = numpy.hstack((labels[:,0:2],numpy.amax(labels[:,2:,None],1)))
  #print("3state: ", predictions_3state.shape, labels_3state.shape, predictions_3state.sum(), labels_3state.sum())
  #print("preds: ", predictions_3state[0:10])
  return 100.0 * numpy.sum(numpy.argmax(predictions_3state, 1) == numpy.argmax(labels_3state, 1)) / predictions_3state.shape[0]

def flatten(data):
  """Returns a reshaped tensor such that [a,b,c,...,z] dimensioned tesnor becomes
     a [a, b*c*...*z] dimensioned one."""
  data_shape = data.get_shape().as_list()
  return tf.reshape(data, [data_shape[0], numpy.array(data_shape[1:]).prod()])

def main(argv=None):  # pylint: disable=unused-argument
  parser = argparse.ArgumentParser(
      description='Secondary structure prediction with TensorFlow.',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument('--dataset', type=str, default="full_list_100.examples.npz",
                      help='Input file contain training, validation and test data.')
  args = parser.parse_args()

  # Extract it into numpy arrays.
  
  all_data = numpy.load(args.dataset)
  all_examples = all_data["examples"]
  all_labels = all_data["labels"][:,1] # use middle label for now
  print ("Training data: ", all_examples.shape, " labels: ", all_labels.shape)
  percentages = numpy.sum(all_labels, axis=0)/all_labels.shape[0]*100
  print(percentages.astype(numpy.int32))

  ## FAKE IT OUT:
  #all_examples[:,11,0:8] = all_labels
  
  # Generate a validation set.
  validation_data   = all_examples[:VALIDATION_SIZE, :, :]
  validation_labels = all_labels[:VALIDATION_SIZE]
  train_data = all_examples[VALIDATION_SIZE:-TEST_SIZE:, :, :]
  train_labels = all_labels[VALIDATION_SIZE:-TEST_SIZE:]
  train_size = train_data.shape[0]
  test_data =   all_examples[-TEST_SIZE:, :, :]
  test_labels = all_labels[-TEST_SIZE:]
  num_epochs = NUM_EPOCHS

  # This is where training samples and labels are fed to the graph.
  # These placeholder nodes will be fed a batch of training data at each
  # training step using the {feed_dict} argument to the Run() call below.
  train_data_node = tf.placeholder(
      tf.float32,
      shape=(BATCH_SIZE, SEQUENCE_WINDOW, NUM_FEATURES))
  train_labels_node = tf.placeholder(tf.float32,
                                     shape=(BATCH_SIZE, NUM_LABELS))
  # For the validation and test data, we'll just hold the entire dataset in
  # one constant node.
  validation_data_node = tf.constant(validation_data)
  test_data_node = tf.constant(test_data)

  # The variables below hold all the trainable weights. They are passed an
  # initial value which will be assigned when when we call:
  # {tf.initialize_all_variables().run()}
  conv1_filters = 64
  conv1_weights = tf.Variable(
      tf.truncated_normal([1, 5, NUM_FEATURES, conv1_filters],  # 5x1 filter, depth 32.
                          stddev=0.1,
                          seed=SEED))
  conv1_biases = tf.Variable(tf.zeros([conv1_filters]))
  conv2_filters = 128
  conv2_weights = tf.Variable(
      tf.truncated_normal([1, 5, conv1_filters, conv2_filters],
                          stddev=0.1,
                          seed=SEED))
  conv2_biases = tf.Variable(tf.constant(0.1, shape=[conv2_filters]))

  #fc1_features = NUM_FEATURES
  fc1_nodes = 128
  fc1_weights = tf.Variable(  
      tf.truncated_normal(
          [SEQUENCE_WINDOW*conv2_filters, fc1_nodes],
          stddev=0.1,
          seed=SEED))
  fc1_biases = tf.Variable(tf.constant(0.1, shape=[fc1_nodes]))
  
  fc2_weights = tf.Variable(
      tf.truncated_normal([fc1_nodes, NUM_LABELS],
                          stddev=0.1,
                          seed=SEED))
  fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))


  def model_fc(data, train=False):
    """The Model definition."""
    # FC1 
    hidden = tf.nn.relu(tf.matmul(flatten(data), fc1_weights) + fc1_biases)
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
      hidden = tf.nn.dropout(hidden, 0.2, seed=SEED)
    # FC2
    hidden = tf.matmul(hidden, fc2_weights) + fc2_biases
    return hidden

  def model_conv(data, train=False):
    """The Model definition."""
    # Add a dimension 'y' with width 1 so we can use conv2d 
    # (even though we're just doing 1d convolutions)
    data_shape = data.get_shape().as_list()
    data = tf.reshape(
        data,
        [data_shape[0], 1,  data_shape[1],  data_shape[2]])
    #print(data.get_shape().as_list())
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    conv = tf.nn.conv2d(conv,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    return model_fc(conv, train) 

  def model(data, train=False):
    return model_conv(data, train) 

  # Training computation: logits + cross-entropy loss.
  logits = model(train_data_node, True)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
      logits, train_labels_node))

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
      0.01,                # Base learning rate.
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,          # Decay step.
      0.99,                # Decay rate.
      staircase=True)
  
  # Use simple momentum for the optimization.
  optimizer = tf.train.MomentumOptimizer(learning_rate,
                                         0.9).minimize(loss,
                                                       global_step=batch)

  # Predictions for the minibatch, validation set and test set.
  train_prediction = tf.nn.softmax(logits)
  # We'll compute them only once in a while by calling their {eval()} method.
  validation_prediction = tf.nn.softmax(model(validation_data_node))
  test_prediction = tf.nn.softmax(model(test_data_node))

  # Create a local session to run this computation.
  with tf.Session() as s:
    # Run all the initializers to prepare the trainable parameters.
    tf.initialize_all_variables().run()
    print('Initialized!')
    # Loop through training steps.
    for step in xrange(num_epochs * train_size // BATCH_SIZE):
      # Compute the offset of the current minibatch in the data.
      # Note that we could use better randomization across epochs.
      offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
      batch_data = train_data[offset:(offset + BATCH_SIZE), :, :]
      batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
      # This dictionary maps the batch data (as a numpy array) to the
      # node in the graph is should be fed to.
      feed_dict = {train_data_node: batch_data,
                   train_labels_node: batch_labels}
      # Run the graph and fetch some of the nodes.
      _, l, lr, predictions = s.run(
          [optimizer, loss, learning_rate, train_prediction],
          feed_dict=feed_dict)
      if step % 100 == 0:
        print('Epoch %.2f learning rate: %.6f Validation error: %.1f%% 3state: %.1f%% ' % (
          float(step) * BATCH_SIZE / train_size, 
          lr,
          success_rate(validation_prediction.eval(), validation_labels), 
          success_rate_3state(validation_prediction.eval(), validation_labels)))
        sys.stdout.flush()
    # Finally print the result!
    test_error = success_rate(test_prediction.eval(), test_labels)
    print('Test error: %.1f%%' % test_error)

if __name__ == '__main__':
  tf.app.run()
  
