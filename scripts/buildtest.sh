#!/bin/bash
cmake -DCUDA_BUILD_LEGACY=OFF -DDOUBLE_PRECISION=ON .. && make -j && valgrind --leak-check=full --show-leak-kinds=all ./ac_run -t && make clean &&\
cmake -DCUDA_BUILD_LEGACY=OFF -DDOUBLE_PRECISION=OFF .. && make -j && valgrind --leak-check=full --show-leak-kinds=all ./ac_run -t
