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
// #include "astaroth_defines.h"
#include "astaroth.h"

#include "errchk.h"
#include "math_utils.h" // int3 + int3

#define AC_GEN_STR(X) #X
const char* intparam_names[]   = {AC_FOR_BUILTIN_INT_PARAM_TYPES(AC_GEN_STR) //
                                AC_FOR_USER_INT_PARAM_TYPES(AC_GEN_STR)};
const char* int3param_names[]  = {AC_FOR_BUILTIN_INT3_PARAM_TYPES(AC_GEN_STR) //
                                 AC_FOR_USER_INT3_PARAM_TYPES(AC_GEN_STR)};
const char* realparam_names[]  = {AC_FOR_BUILTIN_REAL_PARAM_TYPES(AC_GEN_STR) //
                                 AC_FOR_USER_REAL_PARAM_TYPES(AC_GEN_STR)};
const char* real3param_names[] = {AC_FOR_BUILTIN_REAL3_PARAM_TYPES(AC_GEN_STR) //
                                  AC_FOR_USER_REAL3_PARAM_TYPES(AC_GEN_STR)};
const char* vtxbuf_names[]     = {AC_FOR_VTXBUF_HANDLES(AC_GEN_STR)};
#undef AC_GEN_STR

static const int num_nodes = 1;
static Node nodes[num_nodes];
static int3 grid_n;

AcResult
acInit(const AcMeshInfo mesh_info)
{
    grid_n = (int3){mesh_info.int_params[AC_nx], mesh_info.int_params[AC_ny],
                    mesh_info.int_params[AC_nz]};
    return acNodeCreate(0, mesh_info, &nodes[0]);
}

AcResult
acQuit(void)
{
    return acNodeDestroy(nodes[0]);
}

AcResult
acCheckDeviceAvailability(void)
{
    int device_count; // Separate from num_devices to avoid side effects
    ERRCHK_CUDA_ALWAYS(cudaGetDeviceCount(&device_count));
    if (device_count > 0)
        return AC_SUCCESS;
    else
        return AC_FAILURE;
}

AcResult
acSynchronize(void)
{
    return acNodeSynchronizeStream(nodes[0], STREAM_ALL);
}

AcResult
acSynchronizeStream(const Stream stream)
{
    return acNodeSynchronizeStream(nodes[0], stream);
}

AcResult
acLoadDeviceConstant(const AcRealParam param, const AcReal value)
{
    return acNodeLoadConstant(nodes[0], STREAM_DEFAULT, param, value);
}

AcResult
acLoad(const AcMesh host_mesh)
{
    return acNodeLoadMesh(nodes[0], STREAM_DEFAULT, host_mesh);
}

AcResult
acStore(AcMesh* host_mesh)
{
    return acNodeStoreMesh(nodes[0], STREAM_DEFAULT, host_mesh);
}

AcResult
acIntegrate(const AcReal dt)
{
    /*
    acNodeIntegrate(nodes[0], dt);
    return acBoundcondStep();
    */
    return acNodeIntegrate(nodes[0], dt);
}

AcResult
acIntegrateStep(const int& isubstep, const AcReal dt)
{
    const int3 start = (int3){NGHOST, NGHOST, NGHOST};
    const int3 end   = start + grid_n;
    return acNodeIntegrateSubstep(nodes[0], STREAM_DEFAULT, isubstep, start, end, dt);
}

AcResult
acIntegrateStepWithOffset(const int isubstep, const AcReal dt, const int3 start, const int3 end)
{
    return acNodeIntegrateSubstep(nodes[0], STREAM_DEFAULT, isubstep, start, end, dt);
}

AcResult
acBoundcondStep(void)
{
    return acNodePeriodicBoundconds(nodes[0], STREAM_DEFAULT);
}

AcReal
acReduceScal(const ReductionType rtype, const VertexBufferHandle vtxbuf_handle)
{
    AcReal result;
    acNodeReduceScal(nodes[0], STREAM_DEFAULT, rtype, vtxbuf_handle, &result);
    return result;
}

AcReal
acReduceVec(const ReductionType rtype, const VertexBufferHandle a, const VertexBufferHandle b,
            const VertexBufferHandle c)
{
    AcReal result;
    acNodeReduceVec(nodes[0], STREAM_DEFAULT, rtype, a, b, c, &result);
    return result;
}

AcResult
acStoreWithOffset(const int3 dst, const size_t num_vertices, AcMesh* host_mesh)
{
    return acNodeStoreMeshWithOffset(nodes[0], STREAM_DEFAULT, dst, dst, num_vertices, host_mesh);
}
