import tensorflow.compat.v1 as tf

from sw_horovod.common.hvd_base import *
from sw_horovod.common.hvd_tf_ops import *
from sw_horovod.common.compression import Compression

from sw_horovod.tensorflow import tf_make_broadcast_fn
from sw_horovod.tensorflow import tf_make_allreduce_fn

def keras_allreduce_fn(name, compression=Compression.none):
    return tf_make_allreduce_fn(name, compression)

def keras_broadcast_global_variables(root_rank):
    keras_make_broadcast = tf_make_broadcast_fn()
    return keras_make_broadcast(root_rank)


def keras_allreduce(value, name, average=True):
    if average:
        return allreduce(tf.constant(value, name=name)) / size()
    else:
        return allreduce(tf.constant(value, name=name))


def keras_allgather(value, name):
    return allgather(tf.constant(value, name=name), size=size())


def keras_broadcast(value, root_rank, name):
    return broadcast(tf.constant(value, name=name), root_rank)
