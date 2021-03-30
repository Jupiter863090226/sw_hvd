import tensorflow as tf
module = tf.load_op_library('/home/zhaoxc/workspace/sw_hvd/sw_horovod/common/mpi_ops/mpi_ops/my_ops/swap.so')

with tf.Session(''):
    output_tensor = module.tf_swap([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [0, 0], [0, 2]).eval()
    print(output_tensor)
