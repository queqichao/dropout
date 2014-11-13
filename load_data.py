import numpy as np
import scipy.io
import cPickle
import gzip
import os
import sys

import theano
import theano.tensor as T

def min_max_normalize(X):
    min_x = np.min(X, axis=0)
    max_x = np.max(X, axis=0)
    return np.divide(X - min_x, max_x - min_x)

def mean_and_std(X):
  return np.mean(X, axis=0), np.std(X, axis=0)

def mean_sub_and_std_norm(X, mean, std):
  return np.divide(X - mean, std)

def _shared_dataset(data_xy):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                           dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y,
                                           dtype=theano.config.floatX))
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

def load_mnist(path):
    mnist = np.load(path)
    train_set_x = mnist['train_data']
    train_set_y = mnist['train_labels']
    test_set_x = mnist['test_data']
    test_set_y = mnist['test_labels']

    train_set_x, train_set_y = _shared_dataset((train_set_x, train_set_y))
    test_set_x, test_set_y = _shared_dataset((test_set_x, test_set_y))
    valid_set_x, valid_set_y = test_set_x, test_set_y
    
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def load_speech(path, channel_idx=None):
    M = scipy.io.loadmat('../contest/contest_data7.mat')
    train_x = M['train_x']
    train_y = M['train_y']
    dev_x = M['dev_x']
    dev_y = M['dev_y']
    train_mean, train_std = mean_and_std(train_x)
    train_x = mean_sub_and_std_norm(train_x, train_mean, train_std)
    dev_x = mean_sub_and_std_norm(dev_x, train_mean, train_std)
    num_training = train_x.shape[0]
    num_dev = dev_x.shape[0]
    p_training = np.random.permutation(num_training)
    p_dev = np.random.permutation(num_dev)

    if channel_idx is not None:
      train_y = train_y[:, channel_idx]
      dev_y = dev_y[:, channel_idx]

    train_x, train_y = _shared_dataset((train_x[p_training], train_y[p_training]))
    dev_x, dev_y = _shared_dataset((dev_x[p_dev], dev_y[p_dev]))
    M = scipy.io.loadmat('../contest/contest_data_test_425d.mat')
    test_x = M['test_x']
    test_y = M['test_y']
    test_x = mean_sub_and_std_norm(test_x, train_mean, train_std)
    print(test_x)
    num_testing = test_x.shape[0]
    p_testing = np.random.permutation(num_testing)

    if channel_idx is not None:
      test_y = test_y[:, channel_idx]

    test_x, test_y = _shared_dataset((test_x[p_testing], test_y[p_testing]))
    return [(train_x, train_y), (dev_x, dev_y), (test_x, test_y)]

def load_umontreal_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an np.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #np.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def _shared_dataset(data_xy):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX))
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX))
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = _shared_dataset(test_set)
    valid_set_x, valid_set_y = _shared_dataset(valid_set)
    train_set_x, train_set_y = _shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

