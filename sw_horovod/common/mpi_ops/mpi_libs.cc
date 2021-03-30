#include <mpi.h>
//#include <stdio.h>


struct GlobalState {

  // Whether MPI_Init has been completed
  bool initialization_done;
  // The MPI rank, local rank, and size.
  int rank;
  int size;

  GlobalState(bool done, int r, int s) {
    initialization_done = done;
    rank = r;
    size = s;
  }

};

static GlobalState global(false, 0, 0);

extern "C"{
    void mpi_initialize() {
        if (!global.initialization_done) {
            global.initialization_done = true;
            MPI_Init(NULL, NULL);
            MPI_Comm_rank(MPI_COMM_WORLD, &global.rank);
            MPI_Comm_size(MPI_COMM_WORLD, &global.size);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
}

extern "C"{
    int mpi_rank() {
        mpi_initialize();
        return global.rank;
    }
}

extern "C"{
    int mpi_size() {
        mpi_initialize();
        return global.size;
    }
}

extern "C"{
    void mpi_finalize() {
        if(global.initialization_done) {
            MPI_Finalize();
        }
    }
}
