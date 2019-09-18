#!/bin/bash
ACC_DIR=$(realpath $(dirname $0)/../acc)
MODULE_DIR=$(realpath $1)

echo "-- Compiling project in "${MODULE_DIR}
for source in ${MODULE_DIR}/*.sas ${MODULE_DIR}/*.sps ${MODULE_DIR}/*.sdh
do
    ${ACC_DIR}/compile.sh $source -I ${MODULE_DIR}
done
