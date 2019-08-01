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

/** */
AcResult acGridInit(const AcMeshInfo node_config);

/** */
AcResult acGridQuit(void);

/** */
AcResult acGridSynchronizeStream(const Stream stream);

/** */
AcResult acGridSwapBuffers(void);

/** */
AcResult acGridLoadConstant(const Stream stream, const AcRealParam param, const AcReal value);

/** */
AcResult acGridLoadVertexBufferWithOffset(const Stream stream, const AcMesh host_mesh,
                                          const VertexBufferHandle vtxbuf_handle, const int3 src,
                                          const int3 dst, const int num_vertices);

/** */
AcResult acGridLoadMeshWithOffset(const Stream stream, const AcMesh host_mesh, const int3 src,
                                  const int3 dst, const int num_vertices);

/** */
AcResult acGridLoadVertexBuffer(const Stream stream, const AcMesh host_mesh,
                                const VertexBufferHandle vtxbuf_handle);

/** */
AcResult acGridLoadMesh(const Stream stream, const AcMesh host_mesh);

/** */
AcResult acGridStoreVertexBufferWithOffset(const Stream stream,
                                           const VertexBufferHandle vtxbuf_handle, const int3 src,
                                           const int3 dst, const int num_vertices,
                                           AcMesh* host_mesh);

/** */
AcResult acGridStoreMeshWithOffset(const Stream stream, const int3 src, const int3 dst,
                                   const int num_vertices, AcMesh* host_mesh);

/** */
AcResult acGridStoreVertexBuffer(const Stream stream, const VertexBufferHandle vtxbuf_handle,
                                 AcMesh* host_mesh);

/** */
AcResult acGridStoreMesh(const Stream stream, AcMesh* host_mesh);

/** */
AcResult acGridIntegrateSubstep(const Stream stream, const int step_number, const int3 start,
                                const int3 end, const AcReal dt);

/** */
AcResult acGridIntegrateStep(const Stream stream, const AcReal dt);

/** */
AcResult acGridPeriodicBoundcondStep(const Stream stream);
/** */
/* acGridReduceScal(const Stream stream, const ReductionType rtype,
  const VertexBufferHandle vtxbuf_handle, AcReal* result); */
AcReal acGridReduceScal(const Stream stream, const ReductionType rtype,
                        const VertexBufferHandle vtxbuf_handle);
/** */
/*AcResult acGridReduceVec(const Stream stream, const ReductionType rtype,
                         const VertexBufferHandle vec0, const VertexBufferHandle vec1,
                         const VertexBufferHandle vec2, AcReal* result);*/
AcReal acGridReduceVec(const Stream stream, const ReductionType rtype,
                       const VertexBufferHandle vec0, const VertexBufferHandle vec1,
                       const VertexBufferHandle vec2);

#ifdef __cplusplus
} // extern "C"
#endif
