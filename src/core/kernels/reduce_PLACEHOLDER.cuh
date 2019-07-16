/*
    Copyright (C) 2014-2019, Johannes Pekkilae, Miikka Vaeisalae.

    This file is part of Astaroth.

    Astaroth is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Astaroth is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Astaroth.  If not, see <http://www.gnu.org/licenses/>.
*/

/**
 * @file
 * \brief Brief info.
 *
 * Detailed info.
 *
 */
#pragma once
#include "device_globals.cuh"

#include "src/core/errchk.h"
#include "src/core/math_utils.h"

// Function pointer definitions
typedef AcReal (*ReduceFunc)(const AcReal&, const AcReal&);
typedef AcReal (*ReduceInitialScalFunc)(const AcReal&);
typedef AcReal (*ReduceInitialVecFunc)(const AcReal&, const AcReal&,
                                       const AcReal&);

// clang-format off
/* Comparison funcs */
__device__ inline AcReal
_device_max(const AcReal& a, const AcReal& b) { return a > b ? a : b; }

__device__ inline AcReal
_device_min(const AcReal& a, const AcReal& b) { return a < b ? a : b; }

__device__ inline AcReal
_device_sum(const AcReal& a, const AcReal& b) { return a + b; }

/* Function used to determine the values used during reduction */
__device__ inline AcReal
_device_length_scal(const AcReal& a) { return AcReal(a); }

__device__ inline AcReal
_device_squared_scal(const AcReal& a) { return (AcReal)(a*a); }

__device__ inline AcReal
_device_exp_squared_scal(const AcReal& a) { return exp(a)*exp(a); }

__device__ inline AcReal
_device_length_vec(const AcReal& a, const AcReal& b, const AcReal& c) { return sqrt(a*a + b*b + c*c); }

__device__ inline AcReal
_device_squared_vec(const AcReal& a, const AcReal& b, const AcReal& c) { return _device_squared_scal(a) + _device_squared_scal(b) + _device_squared_scal(c); }

__device__ inline AcReal
_device_exp_squared_vec(const AcReal& a, const AcReal& b, const AcReal& c) { return _device_exp_squared_scal(a) + _device_exp_squared_scal(b) + _device_exp_squared_scal(c); }
// clang-format on

__device__ inline bool
oob(const int& i, const int& j, const int& k)
{
    if (i >= d_mesh_info.int_params[AC_nx] ||
        j >= d_mesh_info.int_params[AC_ny] ||
        k >= d_mesh_info.int_params[AC_nz])
        return true;
    else
        return false;
}

template <ReduceInitialScalFunc reduce_initial>
__global__ void
_kernel_reduce_scal(const __restrict__ AcReal* src, AcReal* dst)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (oob(i, j, k))
        return;

    const int src_idx = DEVICE_VTXBUF_IDX(
        i + d_mesh_info.int_params[AC_nx_min],
        j + d_mesh_info.int_params[AC_ny_min],
        k + d_mesh_info.int_params[AC_nz_min]);
    const int dst_idx = DEVICE_1D_COMPDOMAIN_IDX(i, j, k);

    dst[dst_idx] = reduce_initial(src[src_idx]);
}

template <ReduceInitialVecFunc reduce_initial>
__global__ void
_kernel_reduce_vec(const __restrict__ AcReal* src_a,
                   const __restrict__ AcReal* src_b,
                   const __restrict__ AcReal* src_c, AcReal* dst)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (oob(i, j, k))
        return;

    const int src_idx = DEVICE_VTXBUF_IDX(
        i + d_mesh_info.int_params[AC_nx_min],
        j + d_mesh_info.int_params[AC_ny_min],
        k + d_mesh_info.int_params[AC_nz_min]);
    const int dst_idx = DEVICE_1D_COMPDOMAIN_IDX(i, j, k);

    dst[dst_idx] = reduce_initial(src_a[src_idx], src_b[src_idx],
                                  src_c[src_idx]);
}

///////////////////////////////////////////////////////////////////////////////
#define BLOCK_SIZE (1024)
#define ELEMS_PER_THREAD (32)

template <ReduceFunc reduce>
__global__ void
_kernel_reduce(AcReal* src, AcReal* result)
{
    const int idx = threadIdx.x + blockIdx.x * BLOCK_SIZE * ELEMS_PER_THREAD;
    const int scratchpad_size = DCONST_INT(AC_nxyz);

    if (idx >= scratchpad_size)
        return;

    __shared__ AcReal smem[BLOCK_SIZE];

    AcReal tmp = src[idx];

    for (int i = 1; i < ELEMS_PER_THREAD; ++i) {
        const int src_idx = idx + i * BLOCK_SIZE;
        if (src_idx >= scratchpad_size) {
            // This check is for safety: if accessing uninitialized values
            // beyond the mesh boundaries, we will immediately start seeing NANs
            if (threadIdx.x < BLOCK_SIZE)
                smem[threadIdx.x] = NAN;
            else
                break;
        }
        tmp = reduce(tmp, src[src_idx]);
    }

    smem[threadIdx.x] = tmp;
    __syncthreads();

    int offset = BLOCK_SIZE / 2;
    while (offset > 0) {

        if (threadIdx.x < offset) {
            tmp               = reduce(tmp, smem[threadIdx.x + offset]);
            smem[threadIdx.x] = tmp;
        }
        offset /= 2;
        __syncthreads();
    }
    if (threadIdx.x == 0)
        src[idx] = tmp;
}

template <ReduceFunc reduce>
__global__ void
_kernel_reduce_block(const __restrict__ AcReal* src, AcReal* result)
{
    const int scratchpad_size = DCONST_INT(AC_nxyz);
    const int idx = threadIdx.x + blockIdx.x * BLOCK_SIZE * ELEMS_PER_THREAD;
    AcReal tmp    = src[idx];
    const int block_offset = BLOCK_SIZE * ELEMS_PER_THREAD;
    for (int i = 1; idx + i * block_offset < scratchpad_size; ++i)
        tmp = reduce(tmp, src[idx + i * block_offset]);

    *result = tmp;
}
//////////////////////////////////////////////////////////////////////////////

AcReal
_reduce_scal(const cudaStream_t stream,
             const ReductionType& rtype, const int& nx, const int& ny,
             const int& nz, const AcReal* vertex_buffer,
             AcReal* reduce_scratchpad, AcReal* reduce_result)
{
    bool solve_mean = false;

    const dim3 tpb(32, 4, 1);
    const dim3 bpg(int(ceil(AcReal(nx) / tpb.x)), int(ceil(AcReal(ny) / tpb.y)),
                   int(ceil(AcReal(nz) / tpb.z)));

    const int scratchpad_size = nx * ny * nz;
    const int bpg2            = (unsigned int)ceil(AcReal(scratchpad_size) /
                                        AcReal(ELEMS_PER_THREAD * BLOCK_SIZE));

    switch (rtype) {
    case RTYPE_MAX:
        _kernel_reduce_scal<_device_length_scal>
            <<<bpg, tpb, 0, stream>>>(vertex_buffer, reduce_scratchpad);
        _kernel_reduce<_device_max>
            <<<bpg2, BLOCK_SIZE, 0, stream>>>(reduce_scratchpad, reduce_result);
        _kernel_reduce_block<_device_max>
            <<<1, 1, 0, stream>>>(reduce_scratchpad, reduce_result);
        break;
    case RTYPE_MIN:
        _kernel_reduce_scal<_device_length_scal>
            <<<bpg, tpb, 0, stream>>>(vertex_buffer, reduce_scratchpad);
        _kernel_reduce<_device_min>
            <<<bpg2, BLOCK_SIZE, 0, stream>>>(reduce_scratchpad, reduce_result);
        _kernel_reduce_block<_device_min>
            <<<1, 1, 0, stream>>>(reduce_scratchpad, reduce_result);
        break;
    case RTYPE_RMS:
        _kernel_reduce_scal<_device_squared_scal>
            <<<bpg, tpb, 0, stream>>>(vertex_buffer, reduce_scratchpad);
        _kernel_reduce<_device_sum>
            <<<bpg2, BLOCK_SIZE, 0, stream>>>(reduce_scratchpad, reduce_result);
        _kernel_reduce_block<_device_sum>
            <<<1, 1, 0, stream>>>(reduce_scratchpad, reduce_result);
        solve_mean = true;
        break;
    case RTYPE_RMS_EXP:
        _kernel_reduce_scal<_device_exp_squared_scal>
            <<<bpg, tpb, 0, stream>>>(vertex_buffer, reduce_scratchpad);
        _kernel_reduce<_device_sum>
            <<<bpg2, BLOCK_SIZE, 0, stream>>>(reduce_scratchpad, reduce_result);
        _kernel_reduce_block<_device_sum>
            <<<1, 1, 0, stream>>>(reduce_scratchpad, reduce_result);
        solve_mean = true;
        break;
    default:
        ERROR("Unrecognized RTYPE");
    }

    AcReal result;
    cudaMemcpy(&result, reduce_result, sizeof(AcReal), cudaMemcpyDeviceToHost);
    if (solve_mean) {
        const AcReal inv_n = AcReal(1.0) / (nx * ny * nz);
        return inv_n * result;
    }
    else {
        return result;
    }
}

AcReal
_reduce_vec(const cudaStream_t stream,
            const ReductionType& rtype, const int& nx, const int& ny,
            const int& nz, const AcReal* vertex_buffer_a,
            const AcReal* vertex_buffer_b, const AcReal* vertex_buffer_c,
            AcReal* reduce_scratchpad, AcReal* reduce_result)
{
    bool solve_mean = false;

    const dim3 tpb(32, 4, 1);
    const dim3 bpg(int(ceil(float(nx) / tpb.x)),
                   int(ceil(float(ny) / tpb.y)),
                   int(ceil(float(nz) / tpb.z)));

    const int scratchpad_size = nx * ny * nz;
    const int bpg2            = (unsigned int)ceil(float(scratchpad_size) /
                                        float(ELEMS_PER_THREAD * BLOCK_SIZE));

    // "Features" of this quick & efficient reduction:
    // Block size must be smaller than the computational domain size
    // (otherwise we would have do some additional bounds checking in the
    // second half of _kernel_reduce, which gets quite confusing)
    // Also the BLOCK_SIZE must be a multiple of two s.t. we can easily split
    // the work without worrying too much about the array bounds.
    ERRCHK(BLOCK_SIZE <= scratchpad_size);
    ERRCHK(!(BLOCK_SIZE % 2));
    // NOTE! Also does not work properly with non-power of two mesh dimension
    // Issue is with "smem[BLOCK_SIZE];". If you init smem to NANs, you can
    // see that uninitialized smem values are used in the comparison
    ERRCHK(is_power_of_two(nx));
    ERRCHK(is_power_of_two(ny));
    ERRCHK(is_power_of_two(nz));

    switch (rtype) {
    case RTYPE_MAX:
        _kernel_reduce_vec<_device_length_vec>
            <<<bpg, tpb, 0, stream>>>(vertex_buffer_a, vertex_buffer_b, vertex_buffer_c,
                           reduce_scratchpad);
        _kernel_reduce<_device_max>
            <<<bpg2, BLOCK_SIZE, 0, stream>>>(reduce_scratchpad, reduce_result);
        _kernel_reduce_block<_device_max>
            <<<1, 1, 0, stream>>>(reduce_scratchpad, reduce_result);
        break;
    case RTYPE_MIN:
        _kernel_reduce_vec<_device_length_vec>
            <<<bpg, tpb, 0, stream>>>(vertex_buffer_a, vertex_buffer_b, vertex_buffer_c,
                           reduce_scratchpad);
        _kernel_reduce<_device_min>
            <<<bpg2, BLOCK_SIZE, 0, stream>>>(reduce_scratchpad, reduce_result);
        _kernel_reduce_block<_device_min>
            <<<1, 1, 0, stream>>>(reduce_scratchpad, reduce_result);
        break;
    case RTYPE_RMS:
        _kernel_reduce_vec<_device_squared_vec>
            <<<bpg, tpb, 0, stream>>>(vertex_buffer_a, vertex_buffer_b, vertex_buffer_c,
                           reduce_scratchpad);
        _kernel_reduce<_device_sum>
            <<<bpg2, BLOCK_SIZE, 0, stream>>>(reduce_scratchpad, reduce_result);
        _kernel_reduce_block<_device_sum>
            <<<1, 1, 0, stream>>>(reduce_scratchpad, reduce_result);
        solve_mean = true;
        break;
    case RTYPE_RMS_EXP:
        _kernel_reduce_vec<_device_exp_squared_vec>
            <<<bpg, tpb, 0, stream>>>(vertex_buffer_a, vertex_buffer_b, vertex_buffer_c,
                           reduce_scratchpad);
        _kernel_reduce<_device_sum>
            <<<bpg2, BLOCK_SIZE, 0, stream>>>(reduce_scratchpad, reduce_result);
        _kernel_reduce_block<_device_sum>
            <<<1, 1, 0, stream>>>(reduce_scratchpad, reduce_result);
        solve_mean = true;
        break;
    default:
        ERROR("Unrecognized RTYPE");
    }

    AcReal result;
    cudaMemcpy(&result, reduce_result, sizeof(AcReal), cudaMemcpyDeviceToHost);
    if (solve_mean) {
        const AcReal inv_n = AcReal(1.0) / (nx * ny * nz);
        return inv_n * result;
    }
    else {
        return result;
    }
}
