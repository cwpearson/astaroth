#!/bin/bash

#defaults
default_num_procs=8
default_num_nodes=2

num_procs=$default_num_procs
num_nodes=$default_num_nodes

script_name=$0

print_usage(){
    echo "Usage: $script_name [Options]"
    echo "\tRuns mpi_reduce_bench, which will write benchmark results"
    echo "Options:"
    echo "\t -n <num_procs>"
    echo "\t\t-n option to slurm, default=$default_num_procs"
    echo "\t -N <num_nodes>"
    echo "\t\t-N option to slurm, default=$default_num_nodes"
    echo "\t -t <tag>"
    echo "\t\tA benchmark tag that will be added to the mpi_reduction_benchmark.csv file"
    echo "\t\tBy default the current git HEAD short hash will be used as a tag"
}

while getopts n:N:t: opt
do
    case "$opt" in
        n)
            if [ $OPTARG ]
            then
                num_procs=$OPTARG
            else
                print_usage
                exit 1
            fi
        ;;
        N)
            if [ $OPTARG ]
            then
                num_nodes=$OPTARG
            else
                print_usage
                exit 1
            fi
        ;;
        t)
            if [ $OPTARG ]
            then
                benchmark_label=$OPTARG
            else
                print_usage
                exit 1
            fi
        ;;
    esac
done

if [ -z "$benchmark_label" ]
then
    benchmark_label=$(git rev-parse --short HEAD)
fi
set -x
srun --account=project_2000403 --gres=gpu:v100:4 --mem=48000 -t 00:14:59 -p gpu -n ${num_procs} -N ${num_nodes} ./mpi_reduce_bench ${benchmark_label}
