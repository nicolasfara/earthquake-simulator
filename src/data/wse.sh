#!/bin/bash
#
# wse.sh
# Copyright (C) 2019 nicolasfara <nicolas.farabegoli@gmail.com>
#
# Distributed under terms of the MIT license.
#


echo "p\tt1\tt2\tt3\tt4\tt5"

N0=64 # base problem size
CORES=`cat /proc/cpuinfo | grep processor | wc -l` # number of cores

for p in `seq $CORES`; do
    echo -n "$p\t"
    PROB_SIZE=`echo "e(l($N0 * $N0 * $p)/2)" | bc -l -q`
    for rep in `seq 5`; do
      EXEC_TIME="$( OMP_NUM_THREADS=$p ../omp-earthquake 100000 $PROB_SIZE > /dev/null )"
      echo -n "${EXEC_TIME}\t"
    done
    echo ""
done
