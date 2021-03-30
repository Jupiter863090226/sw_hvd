import os
import itertools
import time

os.environ['KMP_WARNINGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import sw_horovod as hvd

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

dims = [1, 2, 3, 4]
dtypes = filter_supported_types([tf.int32, tf.int64, tf.float32, tf.float64])
root_ranks = list(range(size))

def test_allreduce_op():
    data_size = 47
    for dtype, dim, root_rank in itertools.product(dtypes, dims, root_ranks):

        tensor = random_uniform([data_size] * dim, -10, 10, dtype=dtype)
        summed = hvd.allreduce(tensor)
        multiplied = tensor * size
        max_difference = tf.reduce_max(tf.abs(summed - multiplied))

        # Threshold for floating point equality depends on number of
        # ranks, since we're comparing against precise multiplication.
        if dtype in [tf.int32, tf.int64]:
            threshold = 0
        elif size < 128:
            threshold = 1e-5
        else:
            threshold = 1e-4

        diff = evaluate(max_difference)
        if diff <= threshold:
            print("rank: {}, root_rank: {}, dtype: {}, dims: {}, Success".format(rank, root_rank, dtype, dim))
        else:
            print("rank: {}, root_rank: {}, dtype: {}, dims: {}, Fail_diff: {}".format(rank, root_rank, dtype, dim, diff))

#test_allreduce_op()

def test_allreduce_func():
    list_size = 10
    for dtype, dim, root_rank in itertools.product(dtypes, dims, root_ranks):
        tensor_list = []
        for i in range(list_size):
            tensor_list.append(tf.Variable(random_uniform([list_size] * dim, -10, 10, dtype=dtype)))
        before_tensor = tf.identity(tensor_list[0])
        before_list = [t for t in tensor_list]
        with tf.control_dependencies([before_tensor, tf.group(before_list)]):
            allreduce_list = hvd.tf_make_allreduce_fn(name="Test", compression=hvd.Compression.none, buffer_flag=True)(tensor_list)
            with tf.control_dependencies([tf.group(allreduce_list)]):
                after_tensor = tf.identity(tensor_list[0])        
                max_difference_list = []
                for i in range(list_size):
                    max_difference_list.append(tf.reduce_max(tf.abs(before_list[i] - allreduce_list[i])))
                max_difference = tf.reduce_max(max_difference_list)
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if dtype in [tf.int32, tf.int64]:
                threshold = 0
            elif size < 1024:
                threshold = 5e-6
            elif size < 16384:
                threshold = 1e-5
            else:
                threshold = 5e-5

            for i in range(3):
                start_time = time.time()
                before_value = sess.run(before_tensor)
                after_value = sess.run(after_tensor)
                diff = sess.run(max_difference)
                end_time = time.time()
                if rank == 0:
                    if diff < threshold:
                        print("rank: {}, root_rank: {}, dtype: {}, dims: {}, Success_diff: {}".format(rank, root_rank, dtype, dim, diff))
                    else:
                        print("rank: {}, root_rank: {}, dtype: {}, dims: {}, Fail_diff: {}".format(rank, root_rank, dtype, dim, diff))
                        #print("rank: {}, root_rank: {}, dtype: {}, dims: {}, before: {}, after: {}, Success_diff: {}".format(rank, root_rank, dtype, dim,before_value, after_value, diff))

test_allreduce_func()

hvd.finalize()
