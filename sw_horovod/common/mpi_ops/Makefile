CC:=mpic++

LINK_FLAGS:=-shared 
CC_FLAGS:=-std=c++11 -g -w -fPIC

mpi_libs.so: mpi_libs.o
	$(CC) $(LINK_FLAGS) $^ -o $@
mpi_libs.o: mpi_libs.cc
	$(CC) $(CC_FLAGS) -c $^ -o $@

all: mpi_libs.so

clean:
	touch mpi_ops.o && rm *.o *.so
