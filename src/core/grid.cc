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
#include "astaroth_grid.h"

#include "astaroth_device.h"
#include "errchk.h"

/** */
AcResult
acGridInit(const AcMeshInfo info)
{
    int num_processes, pid;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    printf("Processor %s. Process %d of %d.\n", processor_name, pid, num_processes);

    AcMesh submesh;
    return AC_FAILURE;
}

/** */
AcResult
acGridQuit(void)
{
    WARNING("Proper multinode not yet implemented");
    return AC_FAILURE;
}

#if 0
/** */
AcResult
acGridSynchronizeStream(const Stream stream)
{
    WARNING("Proper multinode not yet implemented");
    return AC_FAILURE;
}

/** */
AcResult
acGridSwapBuffers(void)
{
    WARNING("Proper multinode not yet implemented");
    return AC_FAILURE;
}

/** */
AcResult
acGridLoadConstant(const Stream stream, const AcRealParam param, const AcReal value)
{
    WARNING("Proper multinode not yet implemented");
    return AC_FAILURE;
}

/** */
AcResult
acGridLoadVertexBufferWithOffset(const Stream stream, const AcMesh host_mesh,
                                 const VertexBufferHandle vtxbuf_handle, const int3 src,
                                 const int3 dst, const int num_vertices)
{
    WARNING("Proper multinode not yet implemented");
    return AC_FAILURE;
}

/** */
AcResult
acGridLoadMeshWithOffset(const Stream stream, const AcMesh host_mesh, const int3 src,
                         const int3 dst, const int num_vertices)
{
    WARNING("Proper multinode not yet implemented");
    return AC_FAILURE;
}

/** */
AcResult
acGridLoadVertexBuffer(const Stream stream, const AcMesh host_mesh,
                       const VertexBufferHandle vtxbuf_handle)
{
    WARNING("Proper multinode not yet implemented");
    return AC_FAILURE;
}

#endif
/** */
AcResult
acGridLoadMesh(const Stream stream, const AcMesh host_mesh)
{
    (void)stream;
    (void)host_mesh;
    WARNING("Not implemented");
    return AC_FAILURE;
}
#if 0

/** */
AcResult
acGridStoreVertexBufferWithOffset(const Stream stream, const VertexBufferHandle vtxbuf_handle,
                                  const int3 src, const int3 dst, const int num_vertices,
                                  AcMesh* host_mesh)
{
    WARNING("Not implemented");
    return AC_FAILURE;
}

/** */
AcResult
acGridStoreMeshWithOffset(const Stream stream, const int3 src, const int3 dst,
                          const int num_vertices, AcMesh* host_mesh)
{
    WARNING("Not implemented");
    return AC_FAILURE;
}

/** */
AcResult
acGridStoreVertexBuffer(const Stream stream, const VertexBufferHandle vtxbuf_handle,
                        AcMesh* host_mesh)
{
    WARNING("Not implemented");
    return AC_FAILURE;
}

#endif
/** */
AcResult
acGridStoreMesh(const Stream stream, AcMesh* host_mesh)
{
    (void)stream;
    (void)host_mesh;
    WARNING("Not implemented");
    return AC_FAILURE;
}
#if 0

/** */
AcResult
acGridIntegrateSubstep(const Stream stream, const int step_number, const int3 start, const int3 end,
                       const AcReal dt)
{
    WARNING("Not implemented");
    return AC_FAILURE;
}

/** */
AcResult
acGridIntegrateStep(const Stream stream, const AcReal dt)
{
    WARNING("Not implemented");
    return AC_FAILURE;
}

/** */
AcResult
acGridPeriodicBoundcondStep(const Stream stream)
{
    WARNING("Not implemented");
    return AC_FAILURE;
}

/** */
AcResult
acGridReduceScal(const Stream stream, const ReductionType rtype,
                 const VertexBufferHandle vtxbuf_handle, AcReal* result)
{
    return AC_FAILURE;
}
/** */
AcResult
acGridReduceVec(const Stream stream, const ReductionType rtype, const VertexBufferHandle vec0,
                const VertexBufferHandle vec1, const VertexBufferHandle vec2, AcReal* result)
{
    return AC_FAILURE;
}
#endif
