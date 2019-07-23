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

/** Checks whether there are any CUDA devices available. Returns AC_SUCCESS if there is 1 or more,
 * AC_FAILURE otherwise. */
AcResult acCheckDeviceAvailability(void);

/** Synchronizes the stream shared by all GPUs in the node. Synchronizes all streams if STREAM_ALL
 * passed as a parameter */
AcResult acSynchronizeStream(const StreamType stream);

/** Synchronizes the mesh distributed across multiple GPUs. Must be called if the data in the halos
 * of neighboring GPUs has been modified by an asynchronous function, f.ex. acBoundcondStep() */
AcResult acSynchronizeMesh(void);

/** Starting point of all GPU computation. Handles the allocation and
initialization of *all memory needed on all GPUs in the node*. In other words,
setups everything GPU-side so that calling any other GPU interface function
afterwards does not result in illegal memory accesses. */
AcResult acInit(const AcMeshInfo mesh_info);

/** Frees all GPU allocations and resets all devices in the node. Should be
 * called at exit. */
AcResult acQuit(void);

/** Does all three substeps of the RK3 integration and computes the boundary
conditions when necessary. The result is synchronized and the boundary conditions are applied
after the final substep, after which the result can be fetched to CPU memory with acStore. */
AcResult acIntegrate(const AcReal dt);

/** Performs a scalar reduction on all GPUs in the node and returns the result. Operates on the
 * whole computational domain, which must be up to date and synchronized before calling
 * acReduceScal.
 */
AcReal acReduceScal(const ReductionType rtype, const VertexBufferHandle a);

/** Performs a vector reduction on all GPUs in the node and returns the result. Operates on the
 * whole computational domain, which must be up to date and synchronized before calling
 * acReduceVec.
 */
AcReal acReduceVec(const ReductionType rtype, const VertexBufferHandle a,
                   const VertexBufferHandle b, const VertexBufferHandle c);

/** Distributes the host mesh among the GPUs in the node. Synchronous. */
AcResult acLoad(const AcMesh host_mesh);

/** Gathers the mesh stored across GPUs in the node and stores it back to host memory. Synchronous.
 */
AcResult acStore(AcMesh* host_mesh);

/*
 * =============================================================================
 * Astaroth interface: Advanced functions. Asynchronous.
 * =============================================================================
 */
/** Loads a parameter to the constant memory of all GPUs in the node. Asynchronous. */
AcResult acLoadDeviceConstant(const AcRealParam param, const AcReal value);
AcResult acLoadDeviceConstantAsync(const AcRealParam param, const AcReal value,
                                   const StreamType stream);

/** Splits a subset of the host_mesh and distributes it among the GPUs in the node. Asynchronous. */
AcResult acLoadWithOffset(const AcMesh host_mesh, const int3 start, const int num_vertices);
AcResult acLoadWithOffsetAsync(const AcMesh host_mesh, const int3 start, const int num_vertices,
                               const StreamType stream);

/** Gathers a subset of the data distributed among the GPUs in the node and stores the mesh back to
 * CPU memory. Asynchronous.
 */
AcResult acStoreWithOffset(const int3 start, const int num_vertices, AcMesh* host_mesh);
AcResult acStoreWithOffsetAsync(const int3 start, const int num_vertices, AcMesh* host_mesh,
                                const StreamType stream);

/** Performs a single RK3 step without computing boundary conditions. Asynchronous.*/
AcResult acIntegrateStep(const int isubstep, const AcReal dt);
AcResult acIntegrateStepAsync(const int isubstep, const AcReal dt, const StreamType stream);

/** Performs a single RK3 step on a subset of the mesh without computing the boundary conditions.
 * Asynchronous.*/
AcResult acIntegrateStepWithOffset(const int isubstep, const AcReal dt, const int3 start,
                                   const int3 end);
AcResult acIntegrateStepWithOffsetAsync(const int isubstep, const AcReal dt, const int3 start,
                                        const int3 end, const StreamType stream);

/** Performs the boundary condition step on the GPUs in the node. Asynchronous. */
AcResult acBoundcondStep(void);
AcResult acBoundcondStepAsync(const StreamType stream);

/*
 * =============================================================================
 * Revised interface
 * =============================================================================
 */

#ifdef __cplusplus
} // extern "C"
#endif
