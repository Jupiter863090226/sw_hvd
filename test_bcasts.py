import os

os.environ['KMP_WARNINGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import itertools
import sw_horovod.tensorflow as hvd

import time

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
#dims = [2]
#dtypes = filter_supported_types([tf.float32])
#root_ranks = list(range(size))


def test_broadcast_op():
    print("*************broadcast ***********")
    for dtype, dim, root_rank in itertools.product(dtypes, dims, root_ranks):
        tensor = tf.ones([17] * dim) * rank

        root_tensor = tf.ones([17] * dim) * root_rank
        broadcasted_tensor = hvd.broadcasts(tensor, root_rank)
        #res = evaluate(tf.reduce_all(tf.equal(tf.cast(root_tensor, tf.int32), tf.cast(broadcasted_tensor, tf.int32))))
        res = evaluate(tf.reduce_all(tf.equal(root_tensor, broadcasted_tensor)))
        print("rank: %d, dtype: %s, dims: %d, res: %r" % (rank, dtype, dim, res))

#test_broadcast_op()

def test_broadcast_fun():
    for dtype, dim, root_rank in itertools.product(dtypes, dims, root_ranks):
        list_size = 1
        tensor_list = []
        for i in range(list_size):
            tensor_list.append(tf.Variable(tf.ones([list_size] * dim) * rank))
        before_tensor = tf.identity(tensor_list[0])
        with tf.control_dependencies([before_tensor]):
            bcasts_op = hvd.tf_make_broadcast_fn()(root_rank)
            with tf.control_dependencies([bcasts_op]):
                after_tensor = tf.identity(tensor_list[0])

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for i in range(1):
                start_time = time.time()
                before_value = sess.run(before_tensor)
                after_value = sess.run(after_tensor)
                end_time = time.time()
                #if rank == 0:
                print("rank: {}, root_rank: {}, dtype: {}, dims: {}, before: {}, after: {}".format(rank, root_rank, dtype, dim, before_value, after_value))

test_broadcast_fun()

hvd.finalize()
