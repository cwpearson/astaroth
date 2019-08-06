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

__global__ void
kernel_periodic_boundconds(const int3 start, const int3 end, AcReal* vtxbuf)
{
    const int i_dst = start.x + threadIdx.x + blockIdx.x * blockDim.x;
    const int j_dst = start.y + threadIdx.y + blockIdx.y * blockDim.y;
    const int k_dst = start.z + threadIdx.z + blockIdx.z * blockDim.z;

    // If within the start-end range (this allows threadblock dims that are not
    // divisible by end - start)
    if (i_dst >= end.x || j_dst >= end.y || k_dst >= end.z)
        return;

    // if (i_dst >= DCONST_INT(AC_mx) || j_dst >= DCONST_INT(AC_my) || k_dst >= DCONST_INT(AC_mz))
    //    return;

    // If destination index is inside the computational domain, return since
    // the boundary conditions are only applied to the ghost zones
    if (i_dst >= DCONST_INT(AC_nx_min) && i_dst < DCONST_INT(AC_nx_max) &&
        j_dst >= DCONST_INT(AC_ny_min) && j_dst < DCONST_INT(AC_ny_max) &&
        k_dst >= DCONST_INT(AC_nz_min) && k_dst < DCONST_INT(AC_nz_max))
        return;

    // Find the source index
    // Map to nx, ny, nz coordinates
    int i_src = i_dst - DCONST_INT(AC_nx_min);
    int j_src = j_dst - DCONST_INT(AC_ny_min);
    int k_src = k_dst - DCONST_INT(AC_nz_min);

    // Translate (s.t. the index is always positive)
    i_src += DCONST_INT(AC_nx);
    j_src += DCONST_INT(AC_ny);
    k_src += DCONST_INT(AC_nz);

    // Wrap
    i_src %= DCONST_INT(AC_nx);
    j_src %= DCONST_INT(AC_ny);
    k_src %= DCONST_INT(AC_nz);

    // Map to mx, my, mz coordinates
    i_src += DCONST_INT(AC_nx_min);
    j_src += DCONST_INT(AC_ny_min);
    k_src += DCONST_INT(AC_nz_min);

    const int src_idx = DEVICE_VTXBUF_IDX(i_src, j_src, k_src);
    const int dst_idx = DEVICE_VTXBUF_IDX(i_dst, j_dst, k_dst);
    vtxbuf[dst_idx]   = vtxbuf[src_idx];
}

void
periodic_boundconds(const cudaStream_t stream, const int3& start, const int3& end, AcReal* vtxbuf)
{
    const dim3 tpb(8, 2, 8);
    const dim3 bpg((unsigned int)ceil((end.x - start.x) / (float)tpb.x),
                   (unsigned int)ceil((end.y - start.y) / (float)tpb.y),
                   (unsigned int)ceil((end.z - start.z) / (float)tpb.z));

    kernel_periodic_boundconds<<<bpg, tpb, 0, stream>>>(start, end, vtxbuf);
    ERRCHK_CUDA_KERNEL();
}
