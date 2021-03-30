#!/bin/bash
source ../setenv-tf

mpe_num="$1"
shift
set -x
bsub -q q_debug_file -akernel -n $mpe_num -cgsp 64 -I python3 $@
#bsub -q q_test_yyz -node 61783,61784 -akernel -n $mpe_num -cgsp 64 -I python3 $@
