#!/bin/bash
# Preprocesses the give file using GCC. This script is usually automatically called in
# ./compile.sh, but may be called also individually for debugging purposes.
gcc -E -x c ${@} | sed "s/#.*//g"
