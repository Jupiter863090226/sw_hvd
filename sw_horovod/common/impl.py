# Copyright 2017 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow.compat.v1 as tf
from .impl_base import *

def create_distributed_optimizer(keras, optimizer, name, device_dense, device_sparse, compression, sparse_as_dense):

    class _DistributedOptimizer(keras.optimizers.Optimizer):
        def __init__(self, **kwargs):
            self._name = name or "Distributed%s" % self.__class__.__base__.__name__
            self._get_gradients_used = False
            super(self.__class__, self).__init__(**kwargs)
            self._allreduce_fn =  keras_allreduce_fn(self._name, compression)

        def get_gradients(self, loss, params):
            """
            Compute gradients of all trainable variables.

            See Optimizer.get_gradients() for more info.

            In DistributedOptimizer, get_gradients() is overriden to also
            allreduce the gradients before returning them.
            """
            self._get_gradients_used = True
            gradients = super(self.__class__, self).get_gradients(loss, params)
            if size() > 1:
                return self._allreduce_fn(gradients)
            else:
                return gradients

        def apply_gradients(self, *args, **kwargs):
            if not self._get_gradients_used:
                raise Exception('`apply_gradients()` was called without a call to '
                                '`get_gradients()`. If you\'re using TensorFlow 2.0, '
                                'please specify `experimental_run_tf_function=False` in '
                                '`compile()`.')
            return super(self.__class__, self).apply_gradients(*args, **kwargs)

    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override get_gradients() method with an allreduce implementation.
    # This class will have the same name as the optimizer it's wrapping, so that the saved
    # model could be easily restored without Horovod.
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))
    return cls.from_config(optimizer.get_config())


def _eval(backend, op_or_result):
    return backend.get_session().run(op_or_result)

def broadcast_global_variables(backend, root_rank):
    return _eval(backend, keras_broadcast_global_variables(root_rank))


def allreduce(backend, value, name, average):
    return  _eval(backend, keras_allreduce(value, name=name, average=average))


def allgather(backend, value, name):
    return _eval(backend, keras_allgather(value, name=name))


def broadcast(backend, value, root_rank, name):
    return _eval(backend, keras_broadcast(value, root_rank, name=name))


def load_model(keras, wrap_optimizer, filepath, custom_optimizers, custom_objects):
    horovod_objects = {
        subclass.__name__.lower(): wrap_optimizer(subclass)
        for subclass in keras.optimizers.Optimizer.__subclasses__()
        if subclass.__module__ == keras.optimizers.Optimizer.__module__
    }

    if custom_optimizers is not None:
        horovod_objects.update({
            cls.__name__: wrap_optimizer(cls)
            for cls in custom_optimizers
        })

    if custom_objects is not None:
        horovod_objects.update(custom_objects)

    return keras.models.load_model(filepath, custom_objects=horovod_objects)
