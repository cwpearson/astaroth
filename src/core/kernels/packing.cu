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

#include "src/core/errchk.h"

__global__ void
kernel_pack_data(const VertexBufferArray vba, const int3 vba_start, PackedData packed)
{
    const int i_packed = threadIdx.x + blockIdx.x * blockDim.x;
    const int j_packed = threadIdx.y + blockIdx.y * blockDim.y;
    const int k_packed = threadIdx.z + blockIdx.z * blockDim.z;

    // If within the start-end range (this allows threadblock dims that are not
    // divisible by end - start)
    if (i_packed >= packed.dims.x || //
        j_packed >= packed.dims.y || //
        k_packed >= packed.dims.z) {
        return;
    }

    const int i_unpacked = i_packed + vba_start.x;
    const int j_unpacked = j_packed + vba_start.y;
    const int k_unpacked = k_packed + vba_start.z;

    const int unpacked_idx = DEVICE_VTXBUF_IDX(i_unpacked, j_unpacked, k_unpacked);
    const int packed_idx   = i_packed +               //
                           j_packed * packed.dims.x + //
                           k_packed * packed.dims.x * packed.dims.y;

    const size_t vtxbuf_offset = packed.dims.x * packed.dims.y * packed.dims.z;
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i)
        packed.data[packed_idx + i * vtxbuf_offset] = vba.in[i][unpacked_idx];
}

__global__ void
kernel_unpack_data(const PackedData packed, const int3 vba_start, VertexBufferArray vba)
{
    const int i_packed = threadIdx.x + blockIdx.x * blockDim.x;
    const int j_packed = threadIdx.y + blockIdx.y * blockDim.y;
    const int k_packed = threadIdx.z + blockIdx.z * blockDim.z;

    // If within the start-end range (this allows threadblock dims that are not
    // divisible by end - start)
    if (i_packed >= packed.dims.x || //
        j_packed >= packed.dims.y || //
        k_packed >= packed.dims.z) {
        return;
    }

    const int i_unpacked = i_packed + vba_start.x;
    const int j_unpacked = j_packed + vba_start.y;
    const int k_unpacked = k_packed + vba_start.z;

    const int unpacked_idx = DEVICE_VTXBUF_IDX(i_unpacked, j_unpacked, k_unpacked);
    const int packed_idx   = i_packed +               //
                           j_packed * packed.dims.x + //
                           k_packed * packed.dims.x * packed.dims.y;

    const size_t vtxbuf_offset = packed.dims.x * packed.dims.y * packed.dims.z;
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i)
        vba.in[i][unpacked_idx] = packed.data[packed_idx + i * vtxbuf_offset];
}

AcResult
acKernelPackData(const cudaStream_t stream, const VertexBufferArray vba, const int3 vba_start,
                 PackedData packed)
{
    const dim3 tpb(32, 8, 1);
    const dim3 bpg((unsigned int)ceil(packed.dims.x / (float)tpb.x),
                   (unsigned int)ceil(packed.dims.y / (float)tpb.y),
                   (unsigned int)ceil(packed.dims.z / (float)tpb.z));

    kernel_pack_data<<<bpg, tpb, 0, stream>>>(vba, vba_start, packed);
    ERRCHK_CUDA_KERNEL_ALWAYS(); // TODO SET W/ DEBUG ONLY

    return AC_SUCCESS;
}

AcResult
acKernelUnpackData(const cudaStream_t stream, const PackedData packed, const int3 vba_start,
                   VertexBufferArray vba)
{
    const dim3 tpb(32, 8, 1);
    const dim3 bpg((unsigned int)ceil(packed.dims.x / (float)tpb.x),
                   (unsigned int)ceil(packed.dims.y / (float)tpb.y),
                   (unsigned int)ceil(packed.dims.z / (float)tpb.z));

    kernel_unpack_data<<<bpg, tpb, 0, stream>>>(packed, vba_start, vba);
    ERRCHK_CUDA_KERNEL_ALWAYS(); // TODO SET W/ DEBUG ONLY
    return AC_SUCCESS;
}
