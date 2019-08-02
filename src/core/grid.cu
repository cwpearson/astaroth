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

#include "astaroth_node.h"

const size_t MAX_NUM_NODES = 32;
size_t num_nodes           = 0;
static Node nodes[MAX_NUM_NODES];

/** */
AcResult
acGridInit(const AcMeshInfo node_config)
{
    acNodeCreate(0, node_config, &nodes[0]);
    ++num_nodes;
    WARNING("Proper multinode not yet implemented");
    return AC_FAILURE;
}

/** */
AcResult
acGridQuit(void)
{
    acNodeDestroy(nodes[0]);
    --num_nodes;
    WARNING("Proper multinode not yet implemented");
    return AC_FAILURE;
}

/** */
AcResult
acGridSynchronizeStream(const Stream stream)
{
    for (int i = 0; i < num_nodes; ++i) {
        acNodeSynchronizeStream(nodes[i], stream);
    }
    WARNING("Proper multinode not yet implemented");
    return AC_FAILURE;
}

/** */
AcResult
acGridSwapBuffers(void)
{
    for (int i = 0; i < num_nodes; ++i) {
        acNodeSwapBuffers(nodes[i]);
    }
    WARNING("Proper multinode not yet implemented");
    return AC_FAILURE;
}

/** */
AcResult
acGridLoadConstant(const Stream stream, const AcRealParam param, const AcReal value)
{
    for (int i = 0; i < num_nodes; ++i) {
        acNodeLoadConstant(node, stream, param, value);
    }
    WARNING("Proper multinode not yet implemented");
    return AC_FAILURE;
}

/** */
AcResult
acGridLoadVertexBufferWithOffset(const Stream stream, const AcMesh host_mesh,
                                 const VertexBufferHandle vtxbuf_handle, const int3 src,
                                 const int3 dst, const int num_vertices)
{
    for (int i = 0; i < num_nodes; ++i) {
        acNodeLoadVertexBufferWithOffset(node, stream, host_mesh, vtxbuf_handle)
    }
    WARNING("Proper multinode not yet implemented");
    return AC_FAILURE;
}

/** */
AcResult
acGridLoadMeshWithOffset(const Stream stream, const AcMesh host_mesh, const int3 src,
                         const int3 dst, const int num_vertices)
{
    for (int i = 0; i < num_nodes; ++i) {
        acNodeLoadMeshWithOffset(nodes[i], stream, host_mesh, src, dst, num_vertices);
    }
    WARNING("Proper multinode not yet implemented");
    return AC_FAILURE;
}

/** */
AcResult
acGridLoadVertexBuffer(const Stream stream, const AcMesh host_mesh,
                       const VertexBufferHandle vtxbuf_handle)
{
    for (int i = 0; i < num_nodes; ++i) {
        acNodeLoadVertexBuffer(node, stream, host_mesh, vtxbuf_handle);
    }
    WARNING("Proper multinode not yet implemented");
    return AC_FAILURE;
}

/** */
AcResult
acGridLoadMesh(const Stream stream, const AcMesh host_mesh)
{
    WARNING("Not implemented");
    return AC_FAILURE;
}

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

/** */
AcResult
acGridStoreMesh(const Stream stream, AcMesh* host_mesh)
{
    WARNING("Not implemented");
    return AC_FAILURE;
}

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

#if 0
/** */
AcResult acGridReduceScal(const Stream stream, const ReductionType rtype,
                          const VertexBufferHandle vtxbuf_handle, AcReal* result);
/** */
AcResult acGridReduceVec(const Stream stream, const ReductionType rtype,
                         const VertexBufferHandle vec0, const VertexBufferHandle vec1,
                         const VertexBufferHandle vec2, AcReal* result);
#endif
/** */
AcResult
acGridReduceScal(const Stream stream, const ReductionType rtype,
                 const VertexBufferHandle vtxbuf_handle, AcReal* result)
{
    return 0;
}
/** */
AcResult
acGridReduceVec(const Stream stream, const ReductionType rtype, const VertexBufferHandle vec0,
                const VertexBufferHandle vec1, const VertexBufferHandle vec2, AcReal* result)
{
    return 0;
}
