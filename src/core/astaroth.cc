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
#include "astaroth.h"

#include "errchk.h"
#include "math_utils.h"

static const int max_num_nodes   = 1;
static Node nodes[max_num_nodes] = {0};
static int num_nodes             = 0;

AcResult
acInit(const AcMeshInfo mesh_info)
{
    num_nodes = 1;
    return acNodeCreate(0, mesh_info, &nodes[0]);
}

AcResult
acQuit(void)
{
    num_nodes = 0;
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
    return acNodeIntegrate(nodes[0], dt);
}

AcResult
acIntegrateStep(const int isubstep, const AcReal dt)
{
    DeviceConfiguration config;
    acNodeQueryDeviceConfiguration(nodes[0], &config);

    const int3 start = (int3){NGHOST, NGHOST, NGHOST};
    const int3 end   = start + config.grid.n;
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

AcReal
acReduceVecScal(const ReductionType rtype, const VertexBufferHandle a, const VertexBufferHandle b,
            const VertexBufferHandle c, const VertexBufferHandle d)
{
    AcReal result;
    acNodeReduceVecScal(nodes[0], STREAM_DEFAULT, rtype, a, b, c, d, &result);
    return result;
}

AcResult
acStoreWithOffset(const int3 dst, const size_t num_vertices, AcMesh* host_mesh)
{
    return acNodeStoreMeshWithOffset(nodes[0], STREAM_DEFAULT, dst, dst, num_vertices, host_mesh);
}

AcResult
acLoadWithOffset(const AcMesh host_mesh, const int3 src, const int num_vertices)
{
    return acNodeLoadMeshWithOffset(nodes[0], STREAM_DEFAULT, host_mesh, src, src, num_vertices);
}

AcResult
acSynchronizeMesh(void)
{
    return acNodeSynchronizeMesh(nodes[0], STREAM_DEFAULT);
}

int
acGetNumDevicesPerNode(void)
{
    int num_devices;
    ERRCHK_CUDA_ALWAYS(cudaGetDeviceCount(&num_devices));
    return num_devices;
}

Node
acGetNode(void)
{
    ERRCHK_ALWAYS(num_nodes > 0);
    return nodes[0];
}

AcResult
acUpdateBuiltinParams(AcMeshInfo* config)
{
    config->int_params[AC_mx] = config->int_params[AC_nx] + STENCIL_ORDER;
    ///////////// PAD TEST
    // config->int_params[AC_mx] = config->int_params[AC_nx] + STENCIL_ORDER + PAD_SIZE;
    ///////////// PAD TEST
    config->int_params[AC_my] = config->int_params[AC_ny] + STENCIL_ORDER;
    config->int_params[AC_mz] = config->int_params[AC_nz] + STENCIL_ORDER;

    // Bounds for the computational domain, i.e. nx_min <= i < nx_max
    config->int_params[AC_nx_min] = STENCIL_ORDER / 2;
    config->int_params[AC_nx_max] = config->int_params[AC_nx_min] + config->int_params[AC_nx];
    config->int_params[AC_ny_min] = STENCIL_ORDER / 2;
    config->int_params[AC_ny_max] = config->int_params[AC_ny] + STENCIL_ORDER / 2;
    config->int_params[AC_nz_min] = STENCIL_ORDER / 2;
    config->int_params[AC_nz_max] = config->int_params[AC_nz] + STENCIL_ORDER / 2;

// These do not have to be defined by empty projects any more.
// These should be set only if stdderiv.h is included
#ifdef AC_dsx
    config->real_params[AC_inv_dsx] = (AcReal)(1.) / config->real_params[AC_dsx];
#endif
#ifdef AC_dsy
    config->real_params[AC_inv_dsy] = (AcReal)(1.) / config->real_params[AC_dsy];
#endif
#ifdef AC_dsz
    config->real_params[AC_inv_dsz] = (AcReal)(1.) / config->real_params[AC_dsz];
#endif

    /* Additional helper params */
    // Int helpers
    config->int_params[AC_mxy]  = config->int_params[AC_mx] * config->int_params[AC_my];
    config->int_params[AC_nxy]  = config->int_params[AC_nx] * config->int_params[AC_ny];
    config->int_params[AC_nxyz] = config->int_params[AC_nxy] * config->int_params[AC_nz];

    return AC_SUCCESS;
}

AcResult
acMeshCreate(const AcMeshInfo info, AcMesh* mesh)
{
    mesh->info = info;

    const size_t bytes = acVertexBufferSizeBytes(mesh->info);
    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        mesh->vertex_buffer[w] = (AcReal*)malloc(bytes);
        ERRCHK_ALWAYS(mesh->vertex_buffer[w]);
    }

    return AC_SUCCESS;
}

static AcReal
randf(void)
{
    return (AcReal)rand() / (AcReal)RAND_MAX;
}

AcResult
acMeshRandomize(AcMesh* mesh)
{
    const int n = acVertexBufferSize(mesh->info);
    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
        for (int i = 0; i < n; ++i)
            mesh->vertex_buffer[w][i] = randf();

    return AC_SUCCESS;
}

AcResult
acMeshDestroy(AcMesh* mesh)
{
    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
        free(mesh->vertex_buffer[w]);

    return AC_SUCCESS;
}
