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
#include "astaroth_node.h"

#include "astaroth_device.h"
#include "errchk.h"
#include "math_utils.h" // sum for reductions

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

static const int MAX_NUM_DEVICES = 32;
static Node node                 = NULL;

struct node_s {
    int id;

    int num_devices;
    Device devices[MAX_NUM_DEVICES];

    Grid grid;
    Grid subgrid;
};

static int
gridIdx(const Grid grid, const int3 idx)
{
    return idx.x + idx.y * grid.m.x + idx.z * grid.m.x * grid.m.y;
}

static int3
gridIdx3d(const Grid grid, const int idx)
{
    return (int3){idx % grid.m.x, (idx % (grid.m.x * grid.m.y)) / grid.m.x,
                  idx / (grid.m.x * grid.m.y)};
}

static void
printInt3(const int3 vec)
{
    printf("(%d, %d, %d)", vec.x, vec.y, vec.z);
}

static inline void
print(const AcMeshInfo config)
{
    for (int i = 0; i < NUM_INT_PARAMS; ++i)
        printf("[%s]: %d\n", intparam_names[i], config.int_params[i]);
    for (int i = 0; i < NUM_REAL_PARAMS; ++i)
        printf("[%s]: %g\n", realparam_names[i], double(config.real_params[i]));
}

static void
update_builtin_params(AcMeshInfo* config)
{
    config->int_params[AC_mx] = config->int_params[AC_nx] + STENCIL_ORDER;
    ///////////// PAD TEST
    // config->int_params[AC_mx] = config->int_params[AC_nx] + STENCIL_ORDER + PAD_SIZE;
    ///////////// PAD TEST
    config->int_params[AC_my] = config->int_params[AC_ny] + STENCIL_ORDER;
    config->int_params[AC_mz] = config->int_params[AC_nz] + STENCIL_ORDER;

    // Bounds for the computational domain, i.e. nx_min <= i < nx_max
    config->int_params[AC_nx_min] = NGHOST;
    config->int_params[AC_nx_max] = config->int_params[AC_nx_min] + config->int_params[AC_nx];
    config->int_params[AC_ny_min] = NGHOST;
    config->int_params[AC_ny_max] = config->int_params[AC_ny] + NGHOST;
    config->int_params[AC_nz_min] = NGHOST;
    config->int_params[AC_nz_max] = config->int_params[AC_nz] + NGHOST;

    /* Additional helper params */
    // Int helpers
    config->int_params[AC_mxy]  = config->int_params[AC_mx] * config->int_params[AC_my];
    config->int_params[AC_nxy]  = config->int_params[AC_nx] * config->int_params[AC_ny];
    config->int_params[AC_nxyz] = config->int_params[AC_nxy] * config->int_params[AC_nz];

#if VERBOSE_PRINTING // Defined in astaroth.h
    printf("###############################################################\n");
    printf("Config dimensions recalculated:\n");
    print(*config);
    printf("###############################################################\n");
#endif
}

static Grid
createGrid(const AcMeshInfo config)
{
    Grid grid;

    grid.m = (int3){config.int_params[AC_mx], config.int_params[AC_my], config.int_params[AC_mz]};
    grid.n = (int3){config.int_params[AC_nx], config.int_params[AC_ny], config.int_params[AC_nz]};

    return grid;
}

AcResult
acNodeCreate(const int id, const AcMeshInfo node_config, Node* node_handle)
{
    struct node_s* node = (struct device_s*)malloc(sizeof(*node));

    // Get num_devices
    ERRCHK_CUDA_ALWAYS(cudaGetDeviceCount(&num_devices));
    if (num_devices < 1) {
        ERROR("No CUDA devices found!");
        return AC_FAILURE;
    }
    if (num_devices > MAX_NUM_DEVICES) {
        WARNING("More devices found than MAX_NUM_DEVICES. Using only MAX_NUM_DEVICES");
        num_devices = MAX_NUM_DEVICES;
    }
    if (!AC_MULTIGPU_ENABLED) {
        WARNING("MULTIGPU_ENABLED was false. Using only one device");
        num_devices = 1; // Use only one device if multi-GPU is not enabled
    }
    // Check that num_devices is divisible with AC_nz. This makes decomposing the
    // problem domain to multiple GPUs much easier since we do not have to worry
    // about remainders
    ERRCHK_ALWAYS(config.int_params[AC_nz] % num_devices == 0);

    // Decompose the problem domain
    // The main grid
    grid = createGrid(config);

    // Subgrids
    AcMeshInfo subgrid_config = config;
    subgrid_config.int_params[AC_nz] /= num_devices;
    update_builtin_params(&subgrid_config);
    subgrid = createGrid(subgrid_config);

    // Periodic boundary conditions become weird if the system can "fold unto itself".
    ERRCHK_ALWAYS(subgrid.n.x >= STENCIL_ORDER);
    ERRCHK_ALWAYS(subgrid.n.y >= STENCIL_ORDER);
    ERRCHK_ALWAYS(subgrid.n.z >= STENCIL_ORDER);

#if VERBOSE_PRINTING
    // clang-format off
    printf("Grid m ");   printInt3(grid.m);    printf("\n");
    printf("Grid n ");   printInt3(grid.n);    printf("\n");
    printf("Subrid m "); printInt3(subgrid.m); printf("\n");
    printf("Subrid n "); printInt3(subgrid.n); printf("\n");
    // clang-format on
#endif

    // Initialize the devices
    for (int i = 0; i < num_devices; ++i) {
        createDevice(i, subgrid_config, &devices[i]);
        printDeviceInfo(devices[i]);
    }

    // Enable peer access
    for (int i = 0; i < num_devices; ++i) {
        const int front = (i + 1) % num_devices;
        const int back  = (i - 1 + num_devices) % num_devices;

        int can_access_front, can_access_back;
        cudaDeviceCanAccessPeer(&can_access_front, i, front);
        cudaDeviceCanAccessPeer(&can_access_back, i, back);
#if VERBOSE_PRINTING
        printf(
            "Trying to enable peer access from %d to %d (can access: %d) and %d (can access: %d)\n",
            i, front, can_access_front, back, can_access_back);
#endif

        cudaSetDevice(i);
        if (can_access_front) {
            ERRCHK_CUDA_ALWAYS(cudaDeviceEnablePeerAccess(front, 0));
        }
        if (can_access_back) {
            ERRCHK_CUDA_ALWAYS(cudaDeviceEnablePeerAccess(back, 0));
        }
    }
    acNodeSynchronizeStream(STREAM_ALL);

    *node_handle = node;
    return AC_SUCCESS;
}

AcResult
acNodeDestroy(Node node)
{
    acSynchronizeStream(STREAM_ALL);

    for (int i = 0; i < num_devices; ++i) {
        destroyDevice(devices[i]);
    }
    free(node);

    return AC_SUCCESS;
}

AcResult
acNodePrintInfo(const Node node)
{
    WARNING("Not implemented");
    return AC_FAILURE;
}

AcResult
acNodeQueryDeviceConfiguration(const Node node, DeviceConfiguration* config)
{
    WARNING("Not implemented");
    return AC_FAILURE;
}

AcResult
acNodeAutoOptimize(const Node node)
{
    WARNING("Not implemented");
    return AC_FAILURE;
}

AcResult
acNodeSynchronizeStream(const Node node, const Stream stream)
{
    for (int i = 0; i < num_devices; ++i) {
        synchronize(devices[i], stream);
    }

    return AC_SUCCESS;
}

AcResult
acNodeSynchronizeVertexBuffer(const Node node, const Stream stream,
                              const VertexBufferHandle vtxbuf_handle)
{
    // Exchanges the halos of subgrids
    // After this step, the data within the main grid ranging from
    // (0, 0, NGHOST) -> grid.m.x, grid.m.y, NGHOST + grid.n.z
    // has been synchronized and transferred to appropriate subgrids

    // We loop only to num_devices - 1 since the front and back plate of the grid is not
    // transferred because their contents depend on the boundary conditions.

    // IMPORTANT NOTE: the boundary conditions must be applied before calling this function!
    // I.e. the halos of subgrids must contain up-to-date data!

    const size_t num_vertices = subgrid.m.x * subgrid.m.y * NGHOST;

    for (int i = 0; i < num_devices - 1; ++i) {
        // ...|ooooxxx|... -> xxx|ooooooo|...
        const int3 src = (int3){0, 0, subgrid.n.z};
        const int3 dst = (int3){0, 0, 0};

        const Device src_device = devices[i];
        Device dst_device       = devices[i + 1];

        acDeviceTransferVertexBufferWithOffset(src_device, stream, vtxbuf_handle, src, dst,
                                               num_vertices, dst_device);
    }
    for (int i = 1; i < num_devices; ++i) {
        // ...|ooooooo|xxx <- ...|xxxoooo|...
        const int3 src = (int3){0, 0, NGHOST};
        const int3 dst = (int3){0, 0, NGHOST + subgrid.n.z};

        const Device src_device = devices[i];
        Device dst_device       = devices[i - 1];

        acDeviceTransferVertexBufferWithOffset(src_device, stream, vtxbuf_handle, src, dst,
                                               num_vertices, dst_device);
    }
    return AC_SUCCESS;
}

AcResult
acNodeSynchronizeMesh(const Node node, const Stream stream)
{
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        acNodeSynchronizeVertexBuffer(node, stream, (VertexBufferHandle)i);
    }

    return AC_SUCCESS;
}

AcResult
acNodeSwapBuffers(const Node node)
{
    for (int i = 0; i < num_devices; ++i) {
        acDeviceSwapBuffers(devices[i]);
    }
    return AC_SUCCESS;
}

AcResult
acNodeLoadConstant(const Node node, const Stream stream, const AcRealParam param,
                   const AcReal value)
{
    for (int i = 0; i < num_devices; ++i) {
        acDeviceLoadConstant(devices[i], stream, param, value);
    }
    return AC_SUCCESS;
}

AcResult
acNodeLoadVertexBufferWithOffset(const Node node, const Stream stream, const AcMesh host_mesh,
                                 const VertexBufferHandle vtxbuf_handle, const int3 src,
                                 const int3 dst, const int num_vertices)
{
    // See the beginning of the file for an explanation of the index mapping
    // #pragma omp parallel for
    for (int i = 0; i < num_devices; ++i) {
        const int3 d0 = (int3){0, 0, i * subgrid.n.z}; // DECOMPOSITION OFFSET HERE
        const int3 d1 = (int3){subgrid.m.x, subgrid.m.y, d0.z + subgrid.m.z};

        const int3 s0 = src;
        const int3 s1 = gridIdx3d(grid, gridIdx(grid, s0) + num_vertices);

        const int3 da = max(s0, d0);
        const int3 db = min(s1, d1);
        /*
        printf("Device %d\n", i);
        printf("\ts0: "); printInt3(s0); printf("\n");
        printf("\td0: "); printInt3(d0); printf("\n");
        printf("\tda: "); printInt3(da); printf("\n");
        printf("\tdb: "); printInt3(db); printf("\n");
        printf("\td1: "); printInt3(d1); printf("\n");
        printf("\ts1: "); printInt3(s1); printf("\n");
        printf("\t-> %s to device %d\n", db.z >= da.z ? "Copy" : "Do not copy", i);
        */
        if (db.z >= da.z) {
            const int copy_cells = gridIdx(subgrid, db) - gridIdx(subgrid, da);
            // DECOMPOSITION OFFSET HERE
            const int3 da_local = (int3){da.x, da.y, da.z - i * grid.n.z / num_devices};
            // printf("\t\tcopy %d cells to local index ", copy_cells); printInt3(da_local);
            // printf("\n");
            acDeviceLoadVertexBufferWithOffset(devices[i], stream, host_mesh, vtxbuf_handle, da,
                                               da_local, copy_cells);
        }
        // printf("\n");
    }
    return AC_SUCCESS;
}

AcResult
acNodeLoadMeshWithOffset(const Node node, const Stream stream, const AcMesh host_mesh,
                         const int3 src, const int3 dst, const int num_vertices)
{
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        acNodeLoadVertexBufferWithOffset(node, stream, host_mesh, (VertexBufferHandle)i, src, dst,
                                         num_vertices);
    }
    return AC_SUCCESS;
}

AcResult
acNodeLoadVertexBuffer(const Node node, const Stream stream, const AcMesh host_mesh,
                       const VertexBufferHandle vtxbuf_handle)
{
    const int3 src            = (int3){0, 0, 0};
    const int3 dst            = src;
    const size_t num_vertices = acVertexBufferSize(host_mesh.info);

    acNodeLoadVertexBufferWithOffset(node, stream, host_mesh, vtxbuf_handle, src, dst,
                                     num_vertices);
    return AC_SUCCESS;
}

AcResult
acNodeLoadMesh(const Node node, const Stream stream, const AcMesh host_mesh)
{
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        acNodeLoadVertexBuffer(node, stream, host_mesh, (VertexBufferHandle)i);
    }
    return AC_SUCCESS;
}

AcResult
acNodeStoreVertexBufferWithOffset(const Node node, const Stream stream,
                                  const VertexBufferHandle vtxbuf_handle, const int3 src,
                                  const int3 dst, const int num_vertices, AcMesh* host_mesh)
{
    for (int i = 0; i < num_devices; ++i) {
        const int3 d0 = (int3){0, 0, i * subgrid.n.z}; // DECOMPOSITION OFFSET HERE
        const int3 d1 = (int3){subgrid.m.x, subgrid.m.y, d0.z + subgrid.m.z};

        const int3 s0 = src;
        const int3 s1 = gridIdx3d(grid, gridIdx(grid, s0) + num_vertices);

        const int3 da = max(s0, d0);
        const int3 db = min(s1, d1);
        if (db.z >= da.z) {
            const int copy_cells = gridIdx(subgrid, db) - gridIdx(subgrid, da);
            // DECOMPOSITION OFFSET HERE
            const int3 da_local = (int3){da.x, da.y, da.z - i * grid.n.z / num_devices};
            acDeviceStoreVertexBufferWithOffset(devices[i], stream, vtxbuf_handle, da_local, da,
                                                copy_cells, host_mesh);
        }
    }
    return AC_SUCCESS;
}

AcResult
acNodeStoreMeshWithOffset(const Node node, const Stream stream, const int3 src, const int3 dst,
                          const int num_vertices, AcMesh* host_mesh)
{
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        acNodeStoreVertexBufferWithOffset(node, stream, (VertexBufferHandle)i, src, dst,
                                          num_vertices, host_mesh);
    }
    return AC_SUCCESS;
}

AcResult
acNodeStoreVertexBuffer(const Node node, const Stream stream,
                        const VertexBufferHandle vtxbuf_handle, AcMesh* host_mesh)
{
    const int3 src            = (int3){0, 0, 0};
    const int3 dst            = src;
    const size_t num_vertices = acVertexBufferSize(host_mesh.info);

    acNodeStoreVertexBufferWithOffset(node, stream, vtxbuf_handle, src, dst, num_vertices,
                                      host_mesh);

    return AC_SUCCESS;
}

AcResult
acNodeStoreMesh(const Node node, const Stream stream, AcMesh* host_mesh)
{
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        acNodeStoreVertexBuffer(node, stream, (VertexBufferHandle)i, host_mesh);
    }
    return AC_SUCCESS;
}

AcResult
acNodeIntegrateSubstep(const Node node, const Stream stream, const int step_number,
                       const int3 start, const int3 end, const AcReal dt)
{
    for (int i = 0; i < num_devices; ++i) {
        // DECOMPOSITION OFFSET HERE
        const int3 d0 = (int3){NGHOST, NGHOST, NGHOST + i * subgrid.n.z};
        const int3 d1 = d0 + (int3){subgrid.n.x, subgrid.n.y, subgrid.n.z};

        const int3 da = max(start, d0);
        const int3 db = min(end, d1);

        if (db.z >= da.z) {
            const int3 da_local = da - (int3){0, 0, i * subgrid.n.z};
            const int3 db_local = db - (int3){0, 0, i * subgrid.n.z};
            acDeviceIntegrateSubstep(devices[i], stream, isubstep, da_local, db_local, dt);
        }
    }
    return AC_SUCCESS;
}

AcResult
acNodeIntegrate(const Node node, const AcReal dt)
{
    acNodeSynchronizeStream(STREAM_ALL);

    WARNING("Not implementad\n");

    acNodeSynchronizeStream(STREAM_ALL);
    return AC_SUCCESS;
}

static AcResult
local_boundcondstep(const Node node, const StreamType stream, const VertexBufferHandle vtxbuf)
{
    if (num_devices == 1) {
        acDeviceBoundcondStep(devices[0], stream, vtxbuf, (int3){0, 0, 0}, subgrid.m);
    }
    else {
        // Local boundary conditions
        for (int i = 0; i < num_devices; ++i) {
            const int3 d0 = (int3){0, 0, NGHOST}; // DECOMPOSITION OFFSET HERE
            const int3 d1 = (int3){subgrid.m.x, subgrid.m.y, d0.z + subgrid.n.z};
            acDeviceBoundcondStep(devices[i], stream, vtxbuf, d0, d1);
        }
    }
    return AC_SUCCESS;
}

static AcResult
global_boundcondstep(const Node node, const StreamType stream, const VertexBufferHandle vtxbuf)
{
    if (num_devices > 1) {
        const size_t num_vertices = subgrid.m.x * subgrid.m.y * NGHOST;
        {
            // ...|ooooxxx|... -> xxx|ooooooo|...
            const int3 src = (int3){0, 0, subgrid.n.z};
            const int3 dst = (int3){0, 0, 0};

            const Device src_device = devices[num_devices - 1];
            Device dst_device       = devices[0];

            acDeviceTransferVertexBufferWithOffset(src_device, stream, vtxbuf_handle, src, dst,
                                                   num_vertices, dst_device);
        }
        {
            // ...|ooooooo|xxx <- ...|xxxoooo|...
            const int3 src = (int3){0, 0, NGHOST};
            const int3 dst = (int3){0, 0, NGHOST + subgrid.n.z};

            const Device src_device = devices[0];
            Device dst_device       = devices[num_devices - 1];

            acDeviceTransferVertexBufferWithOffset(src_device, stream, vtxbuf_handle, src, dst,
                                                   num_vertices, dst_device);
        }
    }
    return AC_SUCCESS;
}

AcResult
acNodePeriodicBoundcondStep(const Node node, const Stream stream,
                            const VertexBufferHandle vtxbuf_handle)
{
    local_boundcondstep(node, stream, vtxbuf_handle);
    global_boundcondstep(node, stream, vtxbuf_handle);
    acNodeSynchronizeVertexBuffer(node, stream, vtxbuf_handle);

    return AC_SUCCESS;
}

AcResult
acNodePeriodicBoundconds(const Node node, const Stream stream)
{
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        acNodePeriodicBoundcondStep(node, stream, (VertexBufferHandle)i);
    }
    return AC_SUCCESS;
}

static AcReal
simple_final_reduce_scal(const ReductionType& rtype, const AcReal* results, const int& n)
{
    AcReal res = results[0];
    for (int i = 1; i < n; ++i) {
        if (rtype == RTYPE_MAX) {
            res = max(res, results[i]);
        }
        else if (rtype == RTYPE_MIN) {
            res = min(res, results[i]);
        }
        else if (rtype == RTYPE_RMS || rtype == RTYPE_RMS_EXP) {
            res = sum(res, results[i]);
        }
        else {
            ERROR("Invalid rtype");
        }
    }

    if (rtype == RTYPE_RMS || rtype == RTYPE_RMS_EXP) {
        const AcReal inv_n = AcReal(1.) / (grid.n.x * grid.n.y * grid.n.z);
        res                = sqrt(inv_n * res);
    }

    return res;
}

AcResult
acNodeReduceScal(const Node node, const Stream stream, const ReductionType rtype,
                 const VertexBufferHandle vtxbuf_handle, AcReal* result)
{
    acSynchronizeStream(STREAM_ALL);

    AcReal results[num_devices];
    for (int i = 0; i < num_devices; ++i) {
        acDeviceReduceScal(devices[i], STREAM_DEFAULT, rtype, vtxbuffer_handle, &results[i]);
    }

    return simple_final_reduce_scal(rtype, results, num_devices);
}

AcResult
acNodeReduceVec(const Node node, const Stream stream_type, const ReductionType rtype,
                const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                const VertexBufferHandle vtxbuf2, AcReal* result)
{
    acSynchronizeStream(STREAM_ALL);

    AcReal results[num_devices];
    for (int i = 0; i < num_devices; ++i) {
        acDeviceReduceScal(devices[i], STREAM_DEFAULT, rtype, a, b, c, &results[i]);
    }

    return simple_final_reduce_scal(rtype, results, num_devices);
}
