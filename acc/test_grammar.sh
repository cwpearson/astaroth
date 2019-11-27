#!/bin/bash
cd `dirname $0` # Only operate in the same directory with this script

./build_acc.sh

mkdir -p testbin
./compile.sh samples/sample_stencil_process.sps
./compile.sh samples/sample_stencil_assembly.sas

mv stencil_process.cuh testbin/
mv stencil_assembly.cuh testbin/

printf "
#include <stdio.h>
#include <stdlib.h>
#include \"%s\" // i.e. astaroth.h

__constant__ AcMeshInfo d_mesh_info;
#define DCONST(X)  (d_mesh_info.int_params[X])
#define DCONST_REAL(X) (d_mesh_info.real_params[X])
#define DEVICE_VTXBUF_IDX(i, j, k) ((i) + (j)*DCONST(AC_mx) + (k)*DCONST(AC_mxy))


static __device__ __forceinline__ int
IDX(const int i)
{
    return i;
}

static __device__ __forceinline__ int
IDX(const int i, const int j, const int k)
{
    return DEVICE_VTXBUF_IDX(i, j, k);
}

static __device__ __forceinline__ int
IDX(const int3 idx)
{
    return DEVICE_VTXBUF_IDX(idx.x, idx.y, idx.z);
}

#include \"%s\"
#include \"%s\"
int main(void) { printf(\"Grammar check complete.\\\nAll tests passed.\\\n\"); return EXIT_SUCCESS; }
" common_header.h stencil_assembly.cuh stencil_process.cuh >testbin/test.cu

cd testbin
nvcc -std=c++11 test.cu -I ../samples -o test && ./test
