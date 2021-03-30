import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import sw_horovod as hvd

hvd.init()
rank = hvd.rank()
size = hvd.size()


A = tf.constant([[2,3], [4,5]], dtype=tf.float32)

B = hvd.broadcast(A, root=0)

C = hvd.allreduce(A)

D = hvd.allgather(A, size=size)

E = hvd.gather(A, root=0, rank=rank, size=size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if rank==0:
        print(sess.run(A))
        print(sess.run(B))
        print(sess.run(C))
        print(sess.run(D))
        print(sess.run(E))
    else:
        sess.run(A)
        sess.run(B)
        sess.run(C)
        sess.run(D)
        sess.run(E)


hvd.finalize()
