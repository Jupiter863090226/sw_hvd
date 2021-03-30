import tensorflow.compat.v1 as tf
import os

#module_path = os.path.dirname(os.path.realpath(__file__))
#module_file = os.path.join(module_path, 'tf_mpi_ops.so')
#assert (os.path.isfile(module_file)), 'module tf_mpi_ops does not exist'
#mpi_ops = tf.load_op_library(module_file)
mpi_ops = tf.load_op_library('/home/zhaoxc/workspace/sw_hvd/sw_horovod/common/mpi_ops/mpi_ops/tf_mpi_ops.so')

def broadcast(in_tensor, root):
    return mpi_ops.tf_broadcast(in_tensor, root=root)

def allreduce(in_tensor):
    return mpi_ops.tf_allreduce(in_tensor)

def allgather(in_tensor, size):
    return mpi_ops.tf_allgather(in_tensor, size=size)

def gather(in_tensor, root, rank, size):
    return mpi_ops.tf_gather(in_tensor, root=root, rank=rank, size=size)

def allreduces(in_tensor_list, high_prec=0):
    print("****************************now in allreduces*********************************\n")
    print("about to get into mpi_ops.tf_allreduces\n")
    return mpi_ops.tf_allreduces(in_tensor_list, precision=high_prec)

def broadcasts(in_tensor_list, root):
    return mpi_ops.tf_broadcasts(in_tensor_list, root=root)
