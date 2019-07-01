#!/bin/bash
# Usage ./compile <source file>

ACC_DIR=`dirname $0`

FULL_NAME=$(basename -- $1)
FILENAME="${FULL_NAME%.*}"
EXTENSION="${FULL_NAME##*.}"

if [ "${EXTENSION}" = "sas" ]; then
    echo "Generating stencil assembly stage ${FILENAME}.sas -> stencil_assembly.cuh"
    COMPILE_FLAGS="-sas" # Generate stencil assembly stage
    CUH_FILENAME="stencil_assembly.cuh"
elif [ "${EXTENSION}" = "sps" ]; then
    echo "Generating stencil processing stage:  ${FILENAME}.sps -> stencil_process.cuh"
    COMPILE_FLAGS="-sps" # Generate stencil processing stage
    CUH_FILENAME="stencil_process.cuh"
else
    echo "Error: unknown extension" ${EXTENSION} "of file" ${FULL_NAME}
    echo "Extension should be either .sas or .sps"
    exit
fi

${ACC_DIR}/preprocess.sh $2 $1 | ${ACC_DIR}/build/acc ${COMPILE_FLAGS} > ${CUH_FILENAME}
