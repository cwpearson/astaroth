/*
    Copyright (C) 2014-2020, Johannes Pekkila, Miikka Vaisala.

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
#include "astaroth_utils.h"

#include "errchk.h"

AcResult
acVertexBufferSet(const VertexBufferHandle handle, const AcReal value, AcMesh* mesh)
{
    const int n = acVertexBufferSize(mesh->info);
    for (int i = 0; i < n; ++i)
        mesh->vertex_buffer[handle][i] = value;

    return AC_SUCCESS;
}
AcResult
acMeshSet(const AcReal value, AcMesh* mesh)
{
    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
        acVertexBufferSet(w, value, mesh);

    return AC_SUCCESS;
}

AcResult
acMeshApplyPeriodicBounds(AcMesh* mesh)
{
    const AcMeshInfo info = mesh->info;
    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        const int3 start = (int3){0, 0, 0};
        const int3 end   = (int3){info.int_params[AC_mx], info.int_params[AC_my],
                                info.int_params[AC_mz]};

        const int nx = info.int_params[AC_nx];
        const int ny = info.int_params[AC_ny];
        const int nz = info.int_params[AC_nz];

        const int nx_min = info.int_params[AC_nx_min];
        const int ny_min = info.int_params[AC_ny_min];
        const int nz_min = info.int_params[AC_nz_min];

        // The old kxt was inclusive, but our mx_max is exclusive
        const int nx_max = info.int_params[AC_nx_max];
        const int ny_max = info.int_params[AC_ny_max];
        const int nz_max = info.int_params[AC_nz_max];

        // #pragma omp parallel for
        for (int k_dst = start.z; k_dst < end.z; ++k_dst) {
            for (int j_dst = start.y; j_dst < end.y; ++j_dst) {
                for (int i_dst = start.x; i_dst < end.x; ++i_dst) {

                    // If destination index is inside the computational domain, return since
                    // the boundary conditions are only applied to the ghost zones
                    if (i_dst >= nx_min && i_dst < nx_max && j_dst >= ny_min && j_dst < ny_max &&
                        k_dst >= nz_min && k_dst < nz_max)
                        continue;

                    // Find the source index
                    // Map to nx, ny, nz coordinates
                    int i_src = i_dst - nx_min;
                    int j_src = j_dst - ny_min;
                    int k_src = k_dst - nz_min;

                    // Translate (s.t. the index is always positive)
                    i_src += nx;
                    j_src += ny;
                    k_src += nz;

                    // Wrap
                    i_src %= nx;
                    j_src %= ny;
                    k_src %= nz;

                    // Map to mx, my, mz coordinates
                    i_src += nx_min;
                    j_src += ny_min;
                    k_src += nz_min;

                    const size_t src_idx = acVertexBufferIdx(i_src, j_src, k_src, info);
                    const size_t dst_idx = acVertexBufferIdx(i_dst, j_dst, k_dst, info);
                    ERRCHK(src_idx < acVertexBufferSize(info));
                    ERRCHK(dst_idx < acVertexBufferSize(info));
                    mesh->vertex_buffer[w][dst_idx] = mesh->vertex_buffer[w][src_idx];
                }
            }
        }
    }
    return AC_SUCCESS;
}

AcResult
acMeshClear(AcMesh* mesh)
{
    return acMeshSet(0, mesh);
}
