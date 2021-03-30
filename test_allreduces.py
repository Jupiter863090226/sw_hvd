import os
os.environ['KMP_WARNINGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import itertools
import sw_horovod.tensorflow as hvd
import numpy as np

def random_uniform(*args, **kwargs):
    if hasattr(tf, 'random') and hasattr(tf.random, 'set_seed'):
        tf.random.set_seed(1234)
        return tf.random.uniform(*args, **kwargs)
    else:
        tf.set_random_seed(1234)
        return tf.random_uniform(*args, **kwargs)

ccl_supported_types = set([tf.int32, tf.int64, tf.float32, tf.float64])
def filter_supported_types(types):
    types = [t for t in types if t in ccl_supported_types]
    return types

def evaluate(tensors):
    with tf.Session() as sess:
        return sess.run(tensors)


hvd.init()
rank = hvd.rank()
size = hvd.size()

#dims = [1]
#dtypes = filter_supported_types([tf.float32])
#root_ranks = list(range(1))
dims = [1, 2, 3, 4]
dtypes = filter_supported_types([tf.int32, tf.int64, tf.float32, tf.float64])
root_ranks = list(range(size))

def test_allreduce_op():
    for dtype, dim, root_rank in itertools.product(dtypes, dims, root_ranks):
        list_size = 32 
        input_list = []
        for i in range(list_size):
            input_list.append(tf.Variable(random_uniform([list_size] * dim, -10, 10, dtype=dtype)))
        before_tensor = tf.identity(input_list[0])
        summed_list = [t*size for t in input_list]
        with tf.control_dependencies([before_tensor, tf.group(summed_list)]):
            output_list = hvd.allreduces(input_list, 0)
            with tf.control_dependencies([tf.group(output_list)]):
                after_tensor = tf.identity(output_list[0])

        max_difference = tf.reduce_max(tf.abs(summed_list[0] - output_list[0]))

        if dtype in [tf.int32, tf.int64]:
            threshold = 0
        elif size < 128:
            threshold = 1e-5
        else:
            threshold = 1e-4
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            before = sess.run(before_tensor)
            after = sess.run(after_tensor)
        
        print("rank: {}, root_rank: {}, dtype: {}, dims: {}, before: {}, after: {}".format(rank, root_rank, dtype, dim, np.sum(before), np.sum(after)))

test_allreduce_op()

def test_allreduce_func():
    for dtype, dim, root_rank in itertools.product(dtypes, dims, root_ranks):
        list_size = 7 
        tensor_list = []
        for i in range(list_size):
            tensor_list.append(tf.Variable(random_uniform([56] * dim, -10, 10, dtype=dtype)))
        summed_list = [tensor * size for tensor in tensor_list]
        with tf.control_dependencies([tf.group(summed_list)]):
            allreduce_list = hvd.allreduces(tensor_list)
        
        max_difference_list = []
        for i in range(list_size):
            max_difference_list.append(tf.reduce_max(tf.abs(summed_list[i] - allreduce_list[i])))
        max_difference = tf.reduce_max(max_difference_list)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if dtype in [tf.int32, tf.int64]:
                threshold = 0
            elif size < 128:
                threshold = 5e-6
            elif size < 1024:
                threshold = 1e-5
            else:
                break
            out1, out2, diff = sess.run([tensor_list[0], allreduce_list[0], max_difference])
            if diff > threshold:
                print("rank: {}, root_rank: {}, dtype: {}, dims: {}, Success_diff: {}, out1: {}, out2: {}".format(rank, root_rank, dtype, dim, diff, out1, out2))
            else:
                print("rank: {}, root_rank: {}, dtype: {}, dims: {}, Fail_diff: {}".format(rank, root_rank, dtype, dim, diff))
        break

#test_allreduce_func()

hvd.finalize()
