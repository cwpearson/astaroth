#!/bin/bash
# Preprocesses the give file using GCC. This script is usually automatically called in
# ./compile.sh, but may be called also individually for debugging purposes.
cat ${@} | gcc -x c -E - | sed "s/#.*//g"
