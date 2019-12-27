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
#include "packing.cuh"

#include "common.cuh"
#include "src/core/errchk.h"

__global__ void
kernel_pack_data(const AcReal* unpacked, const int3 unpacked_start, const int3 packed_dimensions,
                 AcReal* packed)
{
    const int i_packed = threadIdx.x + blockIdx.x * blockDim.x;
    const int j_packed = threadIdx.y + blockIdx.y * blockDim.y;
    const int k_packed = threadIdx.z + blockIdx.z * blockDim.z;

    // If within the start-end range (this allows threadblock dims that are not
    // divisible by end - start)
    if (i_packed >= packed_dimensions.x || //
        j_packed >= packed_dimensions.y || //
        k_packed >= packed_dimensions.z) {
        return;
    }

    const int i_unpacked = i_packed + unpacked_start.x;
    const int j_unpacked = j_packed + unpacked_start.y;
    const int k_unpacked = k_packed + unpacked_start.z;

    const int unpacked_idx = DEVICE_VTXBUF_IDX(i_unpacked, j_unpacked, k_unpacked);
    const int packed_idx   = i_packed +                     //
                           j_packed * packed_dimensions.x + //
                           k_packed * packed_dimensions.x * packed_dimensions.y;

    packed[packed_idx] = unpacked[unpacked_idx];
}

__global__ void
kernel_unpack_data(const AcReal* packed, const int3 packed_dimensions, const int3 unpacked_start,
                   AcReal* unpacked)
{
    const int i_packed = threadIdx.x + blockIdx.x * blockDim.x;
    const int j_packed = threadIdx.y + blockIdx.y * blockDim.y;
    const int k_packed = threadIdx.z + blockIdx.z * blockDim.z;

    // If within the start-end range (this allows threadblock dims that are not
    // divisible by end - start)
    if (i_packed >= packed_dimensions.x || //
        j_packed >= packed_dimensions.y || //
        k_packed >= packed_dimensions.z) {
        return;
    }

    const int i_unpacked = i_packed + unpacked_start.x;
    const int j_unpacked = j_packed + unpacked_start.y;
    const int k_unpacked = k_packed + unpacked_start.z;

    const int unpacked_idx = DEVICE_VTXBUF_IDX(i_unpacked, j_unpacked, k_unpacked);
    const int packed_idx   = i_packed +                     //
                           j_packed * packed_dimensions.x + //
                           k_packed * packed_dimensions.x * packed_dimensions.y;

    unpacked[unpacked_idx] = packed[packed_idx];
}

AcResult
acKernelPackData(const cudaStream_t stream, const AcReal* unpacked, const int3 unpacked_start,
                 const int3 packed_dimensions, AcReal* packed)
{
    const dim3 tpb(32, 8, 1);
    const dim3 bpg((unsigned int)ceil(packed_dimensions.x / (float)tpb.x),
                   (unsigned int)ceil(packed_dimensions.y / (float)tpb.y),
                   (unsigned int)ceil(packed_dimensions.z / (float)tpb.z));

    kernel_pack_data<<<bpg, tpb, 0, stream>>>(unpacked, unpacked_start, packed_dimensions, packed);
    ERRCHK_CUDA_KERNEL_ALWAYS(); // TODO SET W/ DEBUG ONLY

    return AC_SUCCESS;
}

AcResult
acKernelUnpackData(const cudaStream_t stream, const AcReal* packed, const int3 packed_dimensions,
                   const int3 unpacked_start, AcReal* unpacked)
{
    const dim3 tpb(32, 8, 1);
    const dim3 bpg((unsigned int)ceil(packed_dimensions.x / (float)tpb.x),
                   (unsigned int)ceil(packed_dimensions.y / (float)tpb.y),
                   (unsigned int)ceil(packed_dimensions.z / (float)tpb.z));

    kernel_unpack_data<<<bpg, tpb, 0, stream>>>(packed, packed_dimensions, unpacked_start,
                                                unpacked);
    ERRCHK_CUDA_KERNEL_ALWAYS(); // TODO SET W/ DEBUG ONLY
    return AC_SUCCESS;
}

AcResult
acKernelPackCorner(void)
{
    return AC_FAILURE;
}
AcResult
acKernelUnpackCorner(void)
{
    return AC_FAILURE;
}

AcResult
acKernelPackEdge(void)
{
    return AC_FAILURE;
}
AcResult
acKernelUnpackEdge(void)
{
    return AC_FAILURE;
}

AcResult
acKernelPackSide(void)
{
    return AC_FAILURE;
}
AcResult
acKernelUnpackSide(void)
{
    return AC_FAILURE;
}
