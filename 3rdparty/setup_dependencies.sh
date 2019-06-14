#!/bin/bash
INITIAL_DIR=$(pwd)


# Fetch SDL2
git clone https://github.com/davidsiaw/SDL2.git
cd SDL2
git pull
mkdir build
cd build && cmake .. && make -j

# See https://github.com/davidsiaw/SDL2/blob/master/docs/README-linux.md
# if there are isses with building


# Done
cd $INITIAL_DIR
