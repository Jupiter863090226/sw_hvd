import tensorflow as tf
module = tf.load_op_library('/home/zhaoxc/workspace/sw_hvd/sw_horovod/common/mpi_ops/mpi_ops/my_ops/tf_exchange_op.so')

with tf.Session(''):
    output_tensor = module.tf_exchange([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[4, 5, 6], [7, 8, 9], [1, 2, 3]], [[7, 8, 9], [1, 2, 3], [4, 5, 6]], [[0, 0], [0, 1]], [1, 0]).eval()
    print(output_tensor)
