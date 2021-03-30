import ctypes
import os

#module_path = os.path.dirname(os.path.realpath(__file__))
#module_file = os.path.join(module_path, 'mpi_libs.so')
#assert (os.path.isfile(module_file)), 'module mpi_libs does not exist'
#mpilibs = ctypes.cdll.LoadLibrary(module_file)
mpilibs = ctypes.cdll.LoadLibrary('/home/zhaoxc/workspace/sw_hvd/sw_horovod/common/mpi_ops/mpi_libs.so')

def init():
    mpilibs.mpi_initialize()

def finalize():
    mpilibs.mpi_finalize()

def rank():
    return mpilibs.mpi_rank()

def size():
    return mpilibs.mpi_size()

def local_rank():
    return 0

def local_size():
    return 1

