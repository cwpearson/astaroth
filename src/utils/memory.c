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
#include "memory.h"

#include <math.h>
#include <string.h>

#include "src/core/errchk.h"

AcResult
acMeshCreate(const AcMeshInfo info, AcMesh* mesh)
{
    mesh->info = info;

    const size_t bytes = acVertexBufferSizeBytes(mesh->info);
    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        mesh->vertex_buffer[w] = malloc(bytes);
        ERRCHK_ALWAYS(mesh->vertex_buffer[w]);
    }

    return AC_SUCCESS;
}

AcResult
acMeshDestroy(AcMesh* mesh)
{
    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
        free(mesh->vertex_buffer[w]);

    return AC_SUCCESS;
}

AcResult
acMeshSet(const AcReal value, AcMesh* mesh)
{
    const int n = acVertexBufferSize(mesh->info);
    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
        for (int i = 0; i < n; ++i)
            mesh->vertex_buffer[w][i] = value;

    return AC_SUCCESS;
}

AcResult
acMeshClear(AcMesh* mesh)
{
    return acMeshSet(0, mesh);
}
