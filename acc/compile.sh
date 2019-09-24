#!/bin/bash
# Usage ./compile <source file> <gcc preprocessor flags, f.ex. -I some/path>

ACC_DIR=`dirname $0`

FULL_NAME=$(basename -- $1)
FILENAME="${FULL_NAME%.*}"
EXTENSION="${FULL_NAME##*.}"

if [ "${EXTENSION}" = "sas" ]; then
    COMPILE_FLAGS="-sas" # Generate stencil assembly stage
    CUH_FILENAME="stencil_assembly.cuh"
    echo "-- Generating stencil assembly stage: ${FILENAME}.sas -> ${CUH_FILENAME}"
elif [ "${EXTENSION}" = "sps" ]; then
    COMPILE_FLAGS="-sps" # Generate stencil processing stage
    CUH_FILENAME="stencil_process.cuh"
    echo "-- Generating stencil processing stage:  ${FILENAME}.sps -> ${CUH_FILENAME}"
elif [ "${EXTENSION}" = "sdh" ]; then
    COMPILE_FLAGS="-sdh" # Generate stencil definition header
    CUH_FILENAME="stencil_defines.h"
    echo "-- Generating stencil definition header:  ${FILENAME}.sdh -> ${CUH_FILENAME}"
else
    echo "-- Error: unknown extension" ${EXTENSION} "of file" ${FULL_NAME}
    echo "-- Extension should be either .sas, .sps or .sdh"
    exit
fi

${ACC_DIR}/preprocess.sh $@ | ${ACC_DIR}/build/acc ${COMPILE_FLAGS} > ${CUH_FILENAME}
