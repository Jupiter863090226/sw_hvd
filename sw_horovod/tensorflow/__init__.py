import tensorflow.compat.v1 as tf
from tensorflow.python.framework import dtypes

from sw_horovod.common.hvd_base import *
from sw_horovod.common.hvd_tf_ops import *
from sw_horovod.common.compression import Compression

try:
    _get_default_graph = tf.get_default_graph
except AttributeError:
    _get_default_graph = None

try:
    _SessionRunHook = tf.estimator.SessionRunHook
except AttributeError:
    try:
        _SessionRunHook = tf.train.SessionRunHook
    except AttributeError:
        _SessionRunHook = None

try:
    _global_variables = tf.global_variables
except AttributeError:
    _global_variables = None

if _global_variables is not None:
    def tf_make_broadcast_fn():
        global_variables_list = _global_variables()
        #for var in global_variables_list:
        #    print("var: {}, dtype: {}".format(var, var.dtype))

        def broadcast_global_variables(root_rank):
            broadcasts_variables = broadcasts([tensor if dtypes.as_dtype(tensor.dtype) == dtypes.float32 else tf.cast(tensor, dtype=tf.float32) for tensor in global_variables_list], root_rank)
            return tf.group(*[var.assign(bcasts_var) if dtypes.as_dtype(var.dtype) == dtypes.float32 else var.assign(tf.cast(bcasts_var, dtype=var.dtype)) for var, bcasts_var in zip(global_variables_list, broadcasts_variables)])
        
        return broadcast_global_variables

if _SessionRunHook is not None and _get_default_graph is not None:
    class BroadcastGlobalVariablesHook(_SessionRunHook):

        def __init__(self, root_rank):
            super(BroadcastGlobalVariablesHook, self).__init__()
            self.root_rank = root_rank
            self.bcast_op = None
            self.bcast_global_variables = tf_make_broadcast_fn()

        def begin(self):
            if not self.bcast_op or self.bcast_op.graph != _get_default_graph():
                self.bcast_op = self.bcast_global_variables(self.root_rank)

        def after_create_session(self, session, coord):
            if size() > 1:
                session.run(self.bcast_op)

def tf_make_allreduce_fn(name, compression=Compression.none):

    def allreduces_tensor_list(tensor_list):
        #for tensor in tensor_list:
        #    print("var: {}, dtype: {}".format(tensor, tensor.dtype))

        with tf.name_scope(name+ "_Allreduces"):
            print("****************************************now in allreduces_tensor_list*****************************\n")
            print("about to get into allreduces\n")
            allreduces_list = allreduces([tensor if dtypes.as_dtype(tensor.dtype) == dtypes.float32 else tf.cast(tensor, dtype=tf.float32) for tensor in [not_none_tensor for not_none_tensor in tensor_list if tensor_list is not None]])
            average_list = [tensor / size() for tensor in allreduces_list]

            avg_tensor_list = []
            not_none_tensor_num = 0
            for i in range(len(tensor_list)):
                if tensor_list[i] is not None:
                    avg_tensor_list.append(average_list[not_none_tensor_num] if dtypes.as_dtype(average_list[not_none_tensor_num].dtype) == dtypes.float32 else tf.cast(average_list[not_none_tensor_num], dtype=tf.float32))
                    not_none_tensor_num = not_none_tensor_num + 1
                else:
                    avg_tensor_list.append(None)
            return avg_tensor_list

    return allreduces_tensor_list


try:
    _LegacyOptimizer = tf.train.Optimizer
except AttributeError:
    _LegacyOptimizer = None

if _LegacyOptimizer is not None:
    class _DistributedOptimizer(_LegacyOptimizer):

        def __init__(self, optimizer, name=None, use_locking=False, compression=Compression.none):
            if name is None:
                name = "Distributed{}".format(type(optimizer).__name__)
            super(_DistributedOptimizer, self).__init__(name=name, use_locking=use_locking)

            self._optimizer = optimizer
            self._allreduce = tf_make_allreduce_fn(name, compression)

        def compute_gradients(self, *args, **kwargs):
            print("************************now in compute gradients*********************************\n")
            gradients = self._optimizer.compute_gradients(*args, **kwargs)
            """
            if size() > 1:
                grads, vars = zip(*gradients)
                avg_grads = self._allreduce(grads)
                return list(zip(avg_grads, vars))
            else:
                return gradients
            """
            grads, vars = zip(*gradients)
            avg_grads = self._allreduce(grads)
            return list(zip(avg_grads, vars))

        def apply_gradients(self, *args, **kwargs):
            return self._optimizer.apply_gradients(*args, **kwargs)

        def get_slot(self, *args, **kwargs):
            return self._optimizer.get_slot(*args, **kwargs)

        def get_slot_names(self, *args, **kwargs):
            return self._optimizer.get_slot_names(*args, **kwargs)

        def variables(self, *args, **kwargs):
            return self._optimizer.variables(*args, **kwargs)


def DistributedOptimizer(optimizer, name=None, use_locking=False, compression=Compression.none):

    if isinstance(optimizer, _LegacyOptimizer):
        #print("*****************************************************************************optimizer is an instance of _LegacyOptimizer*******************************************************************\n")
        return _DistributedOptimizer(optimizer, name, use_locking, compression)
    elif isinstance(optimizer, tf.keras.optimizers.Optimizer):
        import sw_horovod.tensorflow.keras as hvd_k
        return hvd_k.DistributedOptimizer(optimizer, name, compression)
    else:
        raise ValueError('Provided optimizer doesn\'t inherit from either legacy '
                         'TensorFlow or Keras optimizer: %s' % optimizer)

