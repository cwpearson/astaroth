#!/bin/bash
ACC_DIR=$(readlink -f $(dirname $0)/../acc)
ACC_BINARY_DIR=$(pwd)/acc
MODULE_DIR=$(readlink -f $1)

echo "-- ACC binary dir:  ${ACC_BINARY_DIR}"
echo "-- ACC project dir: ${MODULE_DIR}"
for source in ${MODULE_DIR}/*.sas ${MODULE_DIR}/*.sps ${MODULE_DIR}/*.sdh
do
    ${ACC_DIR}/compile.sh ${ACC_BINARY_DIR}/acc $source -I ${MODULE_DIR}
done
