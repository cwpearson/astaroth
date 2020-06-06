#!/bin/bash

#defaults
default_num_procs=8
default_num_nodes=2

num_procs=$default_num_procs
num_nodes=$default_num_nodes

script_name=$0

print_usage(){
    echo "Usage: $script_name [Options]"
    echo "      Runs ./mpi_reduce_bench, which will write benchmark results to a csv file"
    echo "      Remember to run this script from your build directory"
    echo "      The benchmarks are submitted with sbatch, unless the -i option is passed"
    echo "Options:"
    echo "      -n <num_procs>"
    echo "              number of tasks for slurm, default=$default_num_procs"
    echo "      -N <num_nodes>"
    echo "              number of nodes for slurm, default=$default_num_nodes"
    echo "      -t <tag>"
    echo "              A benchmark tag that will be added to the mpi_reduction_benchmark.csv file"
    echo "              By default the current git HEAD short hash will be used as a tag"
    echo "      -i"
    echo "              Run the benchmark interactively with srun instead of sbatch"
    echo "      -h"
    echo "              Print this message"
}

while getopts :n:N:t:ih opt
do
    case "$opt" in
        n)
            num_procs=$OPTARG
        ;;
        N)
            num_nodes=$OPTARG
        ;;
        t)
            benchmark_label=$OPTARG
        ;;
        i)
            interactively=1
        ;;
        h)
            print_usage
            exit 0
        ;;
        ?)
            print_usage
            exit 1
    esac
done

if [ -z "$benchmark_label" ]
then
    benchmark_label=$(git rev-parse --short HEAD)
fi
set -x

if [ -z "$interactively"]
then
sbatch <<EOF
#!/bin/sh
#BATCH --job-name=astaroth
#SBATCH --account=project_2000403
#SBATCH --time=00:14:59
#SBATCH --mem=48000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:4
#SBATCH -n ${num_procs}
#SBATCH -N ${num_nodes}
srun ./mpi_reduce_bench ${benchmark_label}
EOF
else
    srun --account=project_2000403 --gres=gpu:v100:4 --mem=48000 -t 00:14:59 -p gpu -n ${num_procs} -N ${num_nodes} ./mpi_reduce_bench ${benchmark_label}
fi
