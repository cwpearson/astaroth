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

#include "astaroth_device.h" // TODO: Should this really be here?

typedef struct node_s* Node; // Opaque pointer to node_s.

typedef struct {
    int3 m;
    int3 n;
} Grid;

typedef struct {
    int num_devices;
    Device* devices;

    Grid grid;
    Grid subgrid;
} DeviceConfiguration;

/** */
AcResult acNodeCreate(const int id, const AcMeshInfo node_config, Node* node);

/** */
AcResult acNodeDestroy(Node node);

/** */
AcResult acNodePrintInfo(const Node node);

/** */
AcResult acNodeQueryDeviceConfiguration(const Node node, DeviceConfiguration* config);

/** */
AcResult acNodeAutoOptimize(const Node node);

/** */
AcResult acNodeSynchronizeStream(const Node node, const Stream stream);

/** Deprecated ? */
AcResult acNodeSynchronizeVertexBuffer(const Node node, const Stream stream,
                                       const VertexBufferHandle vtxbuf_handle); // Not in Device

/** */
AcResult acNodeSynchronizeMesh(const Node node, const Stream stream); // Not in Device

/** */
AcResult acNodeSwapBuffers(const Node node);

/** */
AcResult acNodeLoadConstant(const Node node, const Stream stream, const AcRealParam param,
                            const AcReal value);

/** Deprecated ? Might be useful though if the user wants to load only one vtxbuf. But in this case
 * the user should supply a AcReal* instead of vtxbuf_handle */
AcResult acNodeLoadVertexBufferWithOffset(const Node node, const Stream stream,
                                          const AcMesh host_mesh,
                                          const VertexBufferHandle vtxbuf_handle, const int3 src,
                                          const int3 dst, const int num_vertices);

/** */
AcResult acNodeLoadMeshWithOffset(const Node node, const Stream stream, const AcMesh host_mesh,
                                  const int3 src, const int3 dst, const int num_vertices);

/** Deprecated ? */
AcResult acNodeLoadVertexBuffer(const Node node, const Stream stream, const AcMesh host_mesh,
                                const VertexBufferHandle vtxbuf_handle);

/** */
AcResult acNodeLoadMesh(const Node node, const Stream stream, const AcMesh host_mesh);

/** Deprecated ? */
AcResult acNodeStoreVertexBufferWithOffset(const Node node, const Stream stream,
                                           const VertexBufferHandle vtxbuf_handle, const int3 src,
                                           const int3 dst, const int num_vertices,
                                           AcMesh* host_mesh);

/** */
AcResult acNodeStoreMeshWithOffset(const Node node, const Stream stream, const int3 src,
                                   const int3 dst, const int num_vertices, AcMesh* host_mesh);

/** Deprecated ? */
AcResult acNodeStoreVertexBuffer(const Node node, const Stream stream,
                                 const VertexBufferHandle vtxbuf_handle, AcMesh* host_mesh);

/** */
AcResult acNodeStoreMesh(const Node node, const Stream stream, AcMesh* host_mesh);

/** */
AcResult acNodeIntegrateSubstep(const Node node, const Stream stream, const int step_number,
                                const int3 start, const int3 end, const AcReal dt);

/** */
AcResult acNodeIntegrate(const Node node, const AcReal dt);

/** */
AcResult acNodePeriodicBoundcondStep(const Node node, const Stream stream,
                                     const VertexBufferHandle vtxbuf_handle);

/** */
AcResult acNodePeriodicBoundconds(const Node node, const Stream stream);

/** */
AcResult acNodeReduceScal(const Node node, const Stream stream, const ReductionType rtype,
                          const VertexBufferHandle vtxbuf_handle, AcReal* result);
/** */
AcResult acNodeReduceVec(const Node node, const Stream stream_type, const ReductionType rtype,
                         const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                         const VertexBufferHandle vtxbuf2, AcReal* result);

#ifdef __cplusplus
} // extern "C"
#endif
