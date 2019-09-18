#!/bin/bash
cd `dirname $0` # Only operate in the same directory with this script

COMPILER_NAME="acc"

SRC_DIR=${PWD}/src
BUILD_DIR=${PWD}/build

echo "-- Compiling acc:" ${BUILD_DIR}

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

#echo ${BASE_DIR}
#echo ${SRC_DIR}
#echo ${BUILD_DIR}

# Generate Bison headers
bison --verbose -d ${SRC_DIR}/${COMPILER_NAME}.y

## Generate Flex sources and headers
flex ${SRC_DIR}/${COMPILER_NAME}.l

## Compile the ASPL compiler
gcc -std=gnu11 ${SRC_DIR}/code_generator.c ${COMPILER_NAME}.tab.c lex.yy.c -lfl -I ${BUILD_DIR} -I ${SRC_DIR} -o ${COMPILER_NAME}
