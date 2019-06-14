#!/bin/bash

# Run this in your build directory (cd build && ../scripts/auto_optimize.sh)
# Generates a ${BENCHMARK_FILE} which contains the threadblock dims and other
# constants used in the integration in addition to the time used.

MAX_THREADS=1024 # Max size of the thread block, depends on hardware

BENCHMARK_FILE="benchmark.out"
TBCONFCREATOR_SRC_PATH="../scripts/gen_rk3_threadblockconf.c"
TBCONFFILE_DST_PATH="../src/core/kernels"

C_COMPILER_NAME="gcc"

rm ${BENCHMARK_FILE}

for (( tz=2; tz<=8; tz*=2))
do
for (( ty=1; ty<=1; ty+=1))
do
for (( tx=16; tx<=64; tx*=2))
do

if ( (${tx}*${ty}*${tz}) > ${MAX_THREADS})
then break
fi

for (( launch_bound=1; launch_bound<=8; launch_bound*=2))
do
for (( elems_per_thread=1; elems_per_thread<=128; elems_per_thread*=2))
do
    # Generate the threadblock configuration
    ${C_COMPILER_NAME} ${TBCONFCREATOR_SRC_PATH} -o gen_rk3_threadblockconf
    ./gen_rk3_threadblockconf ${tx} ${ty} ${tz} ${elems_per_thread} ${launch_bound}
    rm gen_rk3_threadblockconf
    mv rk3_threadblock.conf ${TBCONFFILE_DST_PATH}

    # Compile and run the test build
    cmake -DBUILD_DEBUG=OFF -DDOUBLE_PRECISION=OFF -DAUTO_OPTIMIZE=ON .. && make -j
    #if ./ac_run -t; then
    #    echo Success
        ./ac_run -b
    #else
    #    echo fail!
    #fi
done 
done 
done 
done
done 

