CC:=mpic++

TFPREFIX:=/home/limingfan/tf_cpp_15/tf_root
INC_DIR:=  -I/home/zhaoxc/workspace/tensorflow/include
LINK_DIR:= -L/home/zhaoxc/workspace/tensorflow/lib
LINK_FLAGS:=-shared 
CC_FLAGS:=-std=c++11 -g -w -fPIC -D_GLIBCXX_USE_CXX11_ABI=0
LINK_LIB:=-ltensorflow_framework -Wl,-rpath=/home/zhaoxc/workspace/tensorflow/lib

tf_mpi_ops.so: mpi_ops.o mpi_ops_gather.o mpi_ops_allgather.o mpi_ops_allreduce.o mpi_ops_allreduces.o mpi_ops_bcast.o mpi_ops_bcasts.o /home/limingfan/tf_cpp_15/tf_root/lib/libtensorflow_framework.so
	$(CC) $(LINK_FLAGS) $(LINK_DIR) $(LINK_LIB) $^ -o $@
mpi_ops_bcast.o: mpi_ops_bcast.cc
	$(CC) $(CC_FLAGS) $(INC_DIR) -c $^ -o $@
mpi_ops_bcasts.o: mpi_ops_bcasts.cc
	$(CC) $(CC_FLAGS) $(INC_DIR) -c $^ -o $@
mpi_ops_allreduce.o: mpi_ops_allreduce.cc 
	$(CC) $(CC_FLAGS) $(INC_DIR) -c $^ -o $@
mpi_ops_allreduces.o: mpi_ops_allreduces.cc 
	$(CC) $(CC_FLAGS) $(INC_DIR) -c $^ -o $@
mpi_ops_allgather.o: mpi_ops_allgather.cc
	$(CC) $(CC_FLAGS) $(INC_DIR) -c $^ -o $@
mpi_ops_gather.o: mpi_ops_gather.cc
	$(CC) $(CC_FLAGS) $(INC_DIR) -c $^ -o $@
mpi_ops.o: mpi_ops.cc
	$(CC) $(CC_FLAGS) $(INC_DIR) -c $^ -o $@

all: tf_mpi_ops.so

clean:
	touch tf_mpi_ops.so && rm *.o *.so
