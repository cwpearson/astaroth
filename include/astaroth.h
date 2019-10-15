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

#include "astaroth_device.h"
#include "astaroth_node.h"

/*
#include "astaroth_grid.h"
#define acInit(x) acGridInit(x)
#define acQuit() acGridQuit()
#define acLoad(x) acGridLoadMesh(STREAM_DEFAULT, x)
#define acReduceScal(x, y) acGridReduceScal(STREAM_DEFAULT, x, y)
#define acReduceVec(x, y, z, w) acGridReduceVec(STREAM_DEFAULT, x, y, z, w)
#define acBoundcondStep() acGridPeriodicBoundcondStep(STREAM_DEFAULT)
#define acIntegrate(x) acGridIntegrateStep(STREAM_DEFAULT, x)
#define acStore(x) acGridStoreMesh(STREAM_DEFAULT, x)
#define acSynchronizeStream(x) acGridSynchronizeStream(x)
#define acLoadDeviceConstant(x, y) acGridLoadConstant(STREAM_DEFAULT, x, y)
*/

/** Allocates all memory and initializes the devices visible to the caller. Should be
 * called before any other function in this interface. */
AcResult acInit(const AcMeshInfo mesh_info);

/** Frees all GPU allocations and resets all devices in the node. Should be
 * called at exit. */
AcResult acQuit(void);

/** Checks whether there are any CUDA devices available. Returns AC_SUCCESS if there is 1 or more,
 * AC_FAILURE otherwise. */
AcResult acCheckDeviceAvailability(void);

/** Synchronizes a specific stream. All streams are synchronized if STREAM_ALL is passed as a
 * parameter*/
AcResult acSynchronizeStream(const Stream stream);

/** */
AcResult acSynchronizeMesh(void);

/** Loads a constant to the memories of the devices visible to the caller */
AcResult acLoadDeviceConstant(const AcRealParam param, const AcReal value);

/** Loads an AcMesh to the devices visible to the caller */
AcResult acLoad(const AcMesh host_mesh);

/** Stores the AcMesh distributed among the devices visible to the caller back to the host*/
AcResult acStore(AcMesh* host_mesh);

/** Performs Runge-Kutta 3 integration. Note: Boundary conditions are not applied after the final
 * substep and the user is responsible for calling acBoundcondStep before reading the data. */
AcResult acIntegrate(const AcReal dt);

/** Applies periodic boundary conditions for the Mesh distributed among the devices visible to
 * the caller*/
AcResult acBoundcondStep(void);

/** Does a scalar reduction with the data stored in some vertex buffer */
AcReal acReduceScal(const ReductionType rtype, const VertexBufferHandle vtxbuf_handle);

/** Does a vector reduction with vertex buffers where the vector components are (a, b, c) */
AcReal acReduceVec(const ReductionType rtype, const VertexBufferHandle a,
                   const VertexBufferHandle b, const VertexBufferHandle c);

/** Stores a subset of the mesh stored across the devices visible to the caller back to host memory.
 */
AcResult acStoreWithOffset(const int3 dst, const size_t num_vertices, AcMesh* host_mesh);

/** Will potentially be deprecated in later versions. Added only to fix backwards compatibility with
 * PC for now.*/
AcResult acIntegrateStep(const int isubstep, const AcReal dt);
AcResult acIntegrateStepWithOffset(const int isubstep, const AcReal dt, const int3 start,
                                   const int3 end);
AcResult acSynchronize(void);
AcResult acLoadWithOffset(const AcMesh host_mesh, const int3 src, const int num_vertices);

/** */
int acGetNumDevicesPerNode(void);

#ifdef __cplusplus
} // extern "C"
#endif
