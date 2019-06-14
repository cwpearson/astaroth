#!/bin/bash
if [[ $1 == "DO" && $2 == "IT!" ]]; then
    find -name \*.h -o -name \*.cc -o -name \*.cu -o -name \*.cuh | xargs clang-format-6.0 -i -style=file
    echo "It is done."
else
    find -name \*.h -o -name \*.cc -o -name \*.cu -o -name \*.cuh
    echo "I'm going to try to fix the style of these files."
    echo "If you're absolutely sure, give \"DO IT!\" (without quotes) as a parameter."
fi
