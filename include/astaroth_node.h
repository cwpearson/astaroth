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
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "astaroth_defines.h"

typedef struct node_s* Node;

/** */
AcResult acNodeCreate(const AcMeshInfo node_config, Node* node);

/** */
AcResult acNodeDestroy(Node node);

/** */
AcResult acNodeQueryDeviceConfiguration(const Node node, DeviceConfiguration* config);

/** */
AcResult acNodeSynchronizeStream(const Node node, const Stream stream);

/** */
AcResult acNodeSynchronizeVertexBuffer(const Node node, const Stream stream,
                                       const VertexBufferHandle vtxbuf_handle);

/** */
AcResult acNodeSynchronizeMesh(const Node node, const Stream stream);

/** */
AcResult acNodeSwapBuffers(const Node node);

/** */
AcResult acNodeLoadConstant(const Node node, const Stream stream, const AcRealParam param,
                            const AcReal value);

/** */
AcResult acNodeLoadVertexBufferWithOffset(const Node node, const Stream stream,
                                          const AcMesh host_mesh,
                                          const VertexBufferHandle vtxbuf_handle, const int3 src,
                                          const int3 dst, const int num_vertices);

/** */
AcResult acNodeLoadMeshWithOffset(const Node node, const Stream stream, const AcMesh host_mesh,
                                  const int3 src, const int3 dst, const int num_vertices);

/** */
AcResult acNodeLoadVertexBuffer(const Node node, const Stream stream, const AcMesh host_mesh,
                                const VertexBufferHandle vtxbuf_handle);

/** */
AcResult acNodeLoadMesh(const Node node, const Stream stream, const AcMesh host_mesh);

/** */
AcResult acNodeStoreVertexBufferWithOffset(const Node node, const Stream stream,
                                           const VertexBufferHandle vtxbuf_handle, const int3 src,
                                           const int3 dst, const int num_vertices,
                                           AcMesh* host_mesh);

/** */
AcResult acNodeStoreMeshWithOffset(const Node node, const Stream stream, const int3 src,
                                   const int3 dst, const int num_vertices, AcMesh* host_mesh);

/** */
AcResult acNodeStoreVertexBuffer(const Node node, const Stream stream,
                                 const VertexBufferHandle vtxbuf_handle, AcMesh* host_mesh);

/** */
AcResult acNodeStoreMesh(const Node node, const Stream stream, AcMesh* host_mesh);

/** */
AcResult acNodeTransferVertexBufferWithOffset(const Node src_node, const Stream stream,
                                              const VertexBufferHandle vtxbuf_handle,
                                              const int3 src, const int3 dst,
                                              const int num_vertices, Node* dst_node);

/** */
AcResult acNodeTransferMeshWithOffset(const Node src_node, const Stream stream, const int3 src,
                                      const int3 dst, const int num_vertices, Node* dst_node);

/** */
AcResult acNodeTransferVertexBuffer(const Node src_node, const Stream stream,
                                    const VertexBufferHandle vtxbuf_handle, Node* dst_node);

/** */
AcResult acNodeTransferMesh(const Node src_node, const Stream stream, Node* dst_node);

/** */
AcResult acNodeIntegrateSubstep(const Node node, const Stream stream, const int step_number,
                                const int3 start, const int3 end, const AcReal dt);
/** */
AcResult acNodePeriodicBoundcondStep(const Node node, const Stream stream, const int3 start,
                                     const int3 end);
/** */
AcResult acNodeReduceScal(const Node node, const Stream stream, const ReductionType rtype,
                          const VertexBufferHandle vtxbuf_handle, AcReal* result);
/** */
AcResult acNodeReduceVec(const Node node, const Stream stream, const ReductionType rtype,
                         const VertexBufferHandle vec0, const VertexBufferHandle vec1,
                         const VertexBufferHandle vec2, AcReal* result);

#ifdef __cplusplus
} // extern "C"
#endif
