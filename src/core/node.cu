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

/**
 * @file
 * \brief Multi-GPU implementation.
 *
 %JP: The old way for computing boundary conditions conflicts with the
 way we have to do things with multiple GPUs.

 The older approach relied on unified memory, which represented the whole
 memory area as one huge mesh instead of several smaller ones. However, unified memory
 in its current state is more meant for quick prototyping when performance is not an issue.
 Getting the CUDA driver to migrate data intelligently across GPUs is much more difficult
 than when managing the memory explicitly.

 In this new approach, I have simplified the multi- and single-GPU layers significantly.
 Quick rundown:
         New struct: Grid. There are two global variables, "grid" and "subgrid", which
         contain the extents of the whole simulation domain and the decomposed grids,
 respectively. To simplify thing, we require that each GPU is assigned the same amount of
 work, therefore each GPU in the node is assigned and "subgrid.m" -sized block of data to
 work with.

         The whole simulation domain is decomposed with respect to the z dimension.
         For example, if the grid contains (nx, ny, nz) vertices, then the subgrids
         contain (nx, ny, nz / num_devices) vertices.

         An local index (i, j, k) in some subgrid can be mapped to the global grid with
                 global idx = (i, j, k + device_id * subgrid.n.z)

 Terminology:
         - Single-GPU function: a function defined on the single-GPU layer (device.cu)

 Changes required to this commented code block:
         - The thread block dimensions (tpb) are no longer passed to the kernel here but in
 device.cu instead. Same holds for any complex index calculations. Instead, the local
 coordinates should be passed as an int3 type without having to consider how the data is
 actually laid out in device memory
         - The unified memory buffer no longer exists (d_buffer). Instead, we have an opaque
 handle of type "Device" which should be passed to single-GPU functions. In this file, all
 devices are stored in a global array "devices[num_devices]".
         - Every single-GPU function is executed asynchronously by default such that we
           can optimize Astaroth by executing memory transactions concurrently with
 computation. Therefore a StreamType should be passed as a parameter to single-GPU functions.
           Refresher: CUDA function calls are non-blocking when a stream is explicitly passed
           as a parameter and commands executing in different streams can be processed
           in parallel/concurrently.


 Note on periodic boundaries (might be helpful when implementing other boundary conditions):

         With multiple GPUs, periodic boundary conditions applied on indices ranging from

                 (0, 0, STENCIL_ORDER/2) to (subgrid.m.x, subgrid.m.y, subgrid.m.z -
 STENCIL_ORDER/2)

         on a single device are "local", in the sense that they can be computed without
 having to exchange data with neighboring GPUs. Special care is needed only for transferring
         the data to the fron and back plates outside this range. In the solution we use
 here, we solve the local boundaries first, and then just exchange the front and back plates
         in a "ring", like so
                                 device_id
                     (n) <-> 0 <-> 1 <-> ... <-> n <-> (0)

### Throughout this file we use the following notation and names for various index offsets

    Global coordinates: coordinates with respect to the global grid (static Grid grid)
    Local coordinates: coordinates with respect to the local subgrid (static Subgrid subgrid)

    s0, s1: source indices in global coordinates
    d0, d1: destination indices in global coordinates
    da = max(s0, d0);
    db = min(s1, d1);

    These are used in at least
    acLoad()
    acStore()
    acSynchronizeHalos()

     Here we decompose the host mesh and distribute it among the GPUs in
     the node.

     The host mesh is a huge contiguous block of data. Its dimensions are given by
     the global variable named "grid". A "grid" is decomposed into "subgrids",
     one for each GPU. Here we check which parts of the range s0...s1 maps
     to the memory space stored by some GPU, ranging d0...d1, and transfer
     the data if needed.

     The index mapping is inherently quite involved, but here's a picture which
     hopefully helps make sense out of all this.


     Grid
                                      |----num_vertices---|
     xxx|....................................................|xxx
              ^                   ^   ^                   ^
             d0                  d1  s0 (src)            s1

     Subgrid

              xxx|.............|xxx
              ^                   ^
             d0                  d1

                                  ^   ^
                                 db  da
 *
 */
#include "astaroth_node.h"

#include "astaroth_device.h"
#include "errchk.h"
#include "math_utils.h" // sum for reductions

static const int MAX_NUM_DEVICES = 32;

typedef struct {
    int3 m;
    int3 n;
} Grid;

struct node_s {
    int id;

    int num_devices;
    Device devices[MAX_NUM_DEVICES];

    Grid grid;
    Grid subgrid;

    AcMeshInfo config;
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
    struct node_s* node = (struct node_s*)malloc(sizeof(*node));
    node->id            = id;
    node->config        = node_config;

    // Get node->num_devices
    ERRCHK_CUDA_ALWAYS(cudaGetDeviceCount(&node->num_devices));
    if (node->num_devices < 1) {
        ERROR("No CUDA devices found!");
        return AC_FAILURE;
    }
    if (node->num_devices > MAX_NUM_DEVICES) {
        WARNING("More devices found than MAX_NUM_DEVICES. Using only MAX_NUM_DEVICES");
        node->num_devices = MAX_NUM_DEVICES;
    }
    if (!AC_MULTIGPU_ENABLED) {
        WARNING("MULTIGPU_ENABLED was false. Using only one device");
        node->num_devices = 1; // Use only one device if multi-GPU is not enabled
    }
    // Check that node->num_devices is divisible with AC_nz. This makes decomposing the
    // problem domain to multiple GPUs much easier since we do not have to worry
    // about remainders
    ERRCHK_ALWAYS(node->config.int_params[AC_nz] % node->num_devices == 0);

    // Decompose the problem domain
    // The main grid
    node->grid = createGrid(node->config);

    // Subgrids
    AcMeshInfo subgrid_config = node->config;
    subgrid_config.int_params[AC_nz] /= node->num_devices;
    update_builtin_params(&subgrid_config);
#if VERBOSE_PRINTING // Defined in astaroth.h
    printf("###############################################################\n");
    printf("Config dimensions recalculated:\n");
    print(subgrid_config);
    printf("###############################################################\n");
#endif
    node->subgrid = createGrid(subgrid_config);

    // Periodic boundary conditions become weird if the system can "fold unto itself".
    ERRCHK_ALWAYS(node->subgrid.n.x >= STENCIL_ORDER);
    ERRCHK_ALWAYS(node->subgrid.n.y >= STENCIL_ORDER);
    ERRCHK_ALWAYS(node->subgrid.n.z >= STENCIL_ORDER);

#if VERBOSE_PRINTING
    // clang-format off
    printf("Grid m ");   printInt3(node->grid.m);    printf("\n");
    printf("Grid n ");   printInt3(node->grid.n);    printf("\n");
    printf("Subrid m "); printInt3(node->subgrid.m); printf("\n");
    printf("Subrid n "); printInt3(node->subgrid.n); printf("\n");
    // clang-format on
#endif

    // Initialize the devices
    // #pragma omp parallel for
    for (int i = 0; i < node->num_devices; ++i) {
        const int3 multinode_offset                    = (int3){0, 0, 0}; // Placeholder
        const int3 multigpu_offset                     = (int3){0, 0, i * node->subgrid.n.z};
        subgrid_config.int3_params[AC_global_grid_n]   = node->grid.n;
        subgrid_config.int3_params[AC_multigpu_offset] = multinode_offset + multigpu_offset;

        acDeviceCreate(i, subgrid_config, &node->devices[i]);
        acDevicePrintInfo(node->devices[i]);
    }

    // Enable peer access
    // #pragma omp parallel for
    for (int i = 0; i < node->num_devices; ++i) {
        const int front = (i + 1) % node->num_devices;
        const int back  = (i - 1 + node->num_devices) % node->num_devices;

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
    acNodeSynchronizeStream(node, STREAM_ALL);

    *node_handle = node;
    return AC_SUCCESS;
}

AcResult
acNodeDestroy(Node node)
{
    acNodeSynchronizeStream(node, STREAM_ALL);

    // #pragma omp parallel for
    for (int i = 0; i < node->num_devices; ++i) {
        acDeviceDestroy(node->devices[i]);
    }
    free(node);

    return AC_SUCCESS;
}

AcResult
acNodePrintInfo(const Node node)
{
    (void)node;
    WARNING("Not implemented");
    return AC_FAILURE;
}

AcResult
acNodeQueryDeviceConfiguration(const Node node, DeviceConfiguration* config)
{
    (void)node;
    (void)config;
    WARNING("Not implemented");
    return AC_FAILURE;
}

AcResult
acNodeAutoOptimize(const Node node)
{
    (void)node;
    WARNING("Not implemented");
    return AC_FAILURE;
}

AcResult
acNodeSynchronizeStream(const Node node, const Stream stream)
{
    // #pragma omp parallel for
    for (int i = 0; i < node->num_devices; ++i) {
        acDeviceSynchronizeStream(node->devices[i], stream);
    }

    return AC_SUCCESS;
}

AcResult
acNodeSynchronizeVertexBuffer(const Node node, const Stream stream,
                              const VertexBufferHandle vtxbuf_handle)
{
    acNodeSynchronizeStream(node, stream);
    // Exchanges the halos of subgrids
    // After this step, the data within the main grid ranging from
    // (0, 0, NGHOST) -> grid.m.x, grid.m.y, NGHOST + grid.n.z
    // has been synchronized and transferred to appropriate subgrids

    // We loop only to node->num_devices - 1 since the front and back plate of the grid is not
    // transferred because their contents depend on the boundary conditions.

    // IMPORTANT NOTE: the boundary conditions must be applied before
    // callingacNodeSynchronizeStream(node,  this function! I.e. the halos of subgrids must contain
    // up-to-date data!

    const size_t num_vertices = node->subgrid.m.x * node->subgrid.m.y * NGHOST;

    // #pragma omp parallel for
    for (int i = 0; i < node->num_devices - 1; ++i) {
        // ...|ooooxxx|... -> xxx|ooooooo|...
        const int3 src = (int3){0, 0, node->subgrid.n.z};
        const int3 dst = (int3){0, 0, 0};

        const Device src_device = node->devices[i];
        Device dst_device       = node->devices[i + 1];

        acDeviceTransferVertexBufferWithOffset(src_device, stream, vtxbuf_handle, src, dst,
                                               num_vertices, dst_device);
    }
    // #pragma omp parallel for
    for (int i = 1; i < node->num_devices; ++i) {
        // ...|ooooooo|xxx <- ...|xxxoooo|...
        const int3 src = (int3){0, 0, NGHOST};
        const int3 dst = (int3){0, 0, NGHOST + node->subgrid.n.z};

        const Device src_device = node->devices[i];
        Device dst_device       = node->devices[i - 1];

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
    // #pragma omp parallel for
    for (int i = 0; i < node->num_devices; ++i) {
        acDeviceSwapBuffers(node->devices[i]);
    }
    return AC_SUCCESS;
}

AcResult
acNodeLoadConstant(const Node node, const Stream stream, const AcRealParam param,
                   const AcReal value)
{
    acNodeSynchronizeStream(node, stream);
    // #pragma omp parallel for
    for (int i = 0; i < node->num_devices; ++i) {
        acDeviceLoadConstant(node->devices[i], stream, param, value);
    }
    return AC_SUCCESS;
}

AcResult
acNodeLoadVertexBufferWithOffset(const Node node, const Stream stream, const AcMesh host_mesh,
                                 const VertexBufferHandle vtxbuf_handle, const int3 src,
                                 const int3 dst, const int num_vertices)
{
    acNodeSynchronizeStream(node, stream);
    // See the beginning of the file for an explanation of the index mapping
    // // #pragma omp parallel for
    // #pragma omp parallel for
    for (int i = 0; i < node->num_devices; ++i) {
        const int3 d0 = (int3){0, 0, i * node->subgrid.n.z}; // DECOMPOSITION OFFSET HERE
        const int3 d1 = (int3){node->subgrid.m.x, node->subgrid.m.y, d0.z + node->subgrid.m.z};

        const int3 s0 = src; // dst; // TODO fix
        (void)dst;           // TODO fix
        const int3 s1 = gridIdx3d(node->grid, gridIdx(node->grid, s0) + num_vertices);

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
            const int copy_cells = gridIdx(node->subgrid, db) - gridIdx(node->subgrid, da);
            // DECOMPOSITION OFFSET HERE
            const int3 da_global = da; // src + da - dst; // TODO fix
            const int3 da_local = (int3){da.x, da.y, da.z - i * node->grid.n.z / node->num_devices};
            // printf("\t\tcopy %d cells to local index ", copy_cells); printInt3(da_local);
            // printf("\n");
            acDeviceLoadVertexBufferWithOffset(node->devices[i], stream, host_mesh, vtxbuf_handle,
                                               da_global, da_local, copy_cells);
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
    acNodeSynchronizeStream(node, stream);
    // #pragma omp parallel for
    for (int i = 0; i < node->num_devices; ++i) {
        const int3 d0 = (int3){0, 0, i * node->subgrid.n.z}; // DECOMPOSITION OFFSET HERE
        const int3 d1 = (int3){node->subgrid.m.x, node->subgrid.m.y, d0.z + node->subgrid.m.z};

        const int3 s0 = src; // TODO fix
        (void)dst;           // TODO fix
        const int3 s1 = gridIdx3d(node->grid, gridIdx(node->grid, s0) + num_vertices);

        const int3 da = max(s0, d0);
        const int3 db = min(s1, d1);
        if (db.z >= da.z) {
            const int copy_cells = gridIdx(node->subgrid, db) - gridIdx(node->subgrid, da);
            // DECOMPOSITION OFFSET HERE
            const int3 da_local = (int3){da.x, da.y, da.z - i * node->grid.n.z / node->num_devices};
            const int3 da_global = da; // dst + da - src; // TODO fix
            acDeviceStoreVertexBufferWithOffset(node->devices[i], stream, vtxbuf_handle, da_local,
                                                da_global, copy_cells, host_mesh);
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
    const size_t num_vertices = acVertexBufferSize(host_mesh->info);

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
acNodeIntegrateSubstep(const Node node, const Stream stream, const int isubstep, const int3 start,
                       const int3 end, const AcReal dt)
{
    acNodeSynchronizeStream(node, stream);

    // #pragma omp parallel for
    for (int i = 0; i < node->num_devices; ++i) {
        // DECOMPOSITION OFFSET HERE
        const int3 d0 = (int3){NGHOST, NGHOST, NGHOST + i * node->subgrid.n.z};
        const int3 d1 = d0 + (int3){node->subgrid.n.x, node->subgrid.n.y, node->subgrid.n.z};

        const int3 da = max(start, d0);
        const int3 db = min(end, d1);

        if (db.z >= da.z) {
            const int3 da_local = da - (int3){0, 0, i * node->subgrid.n.z};
            const int3 db_local = db - (int3){0, 0, i * node->subgrid.n.z};
            acDeviceIntegrateSubstep(node->devices[i], stream, isubstep, da_local, db_local, dt);
        }
    }
    return AC_SUCCESS;
}

static AcResult
local_boundcondstep(const Node node, const Stream stream, const VertexBufferHandle vtxbuf)
{
    acNodeSynchronizeStream(node, stream);

    if (node->num_devices > 1) {
        // Local boundary conditions
        // #pragma omp parallel for
        for (int i = 0; i < node->num_devices; ++i) {
            const int3 d0 = (int3){0, 0, NGHOST}; // DECOMPOSITION OFFSET HERE
            const int3 d1 = (int3){node->subgrid.m.x, node->subgrid.m.y, d0.z + node->subgrid.n.z};
            acDevicePeriodicBoundcondStep(node->devices[i], stream, vtxbuf, d0, d1);
        }
    }
    else {
        acDevicePeriodicBoundcondStep(node->devices[0], stream, vtxbuf, (int3){0, 0, 0},
                                      node->subgrid.m);
    }
    return AC_SUCCESS;
}

static AcResult
global_boundcondstep(const Node node, const Stream stream, const VertexBufferHandle vtxbuf_handle)
{
    acNodeSynchronizeStream(node, stream);

    if (node->num_devices > 1) {
        const size_t num_vertices = node->subgrid.m.x * node->subgrid.m.y * NGHOST;
        {
            // ...|ooooxxx|... -> xxx|ooooooo|...
            const int3 src = (int3){0, 0, node->subgrid.n.z};
            const int3 dst = (int3){0, 0, 0};

            const Device src_device = node->devices[node->num_devices - 1];
            Device dst_device       = node->devices[0];

            acDeviceTransferVertexBufferWithOffset(src_device, stream, vtxbuf_handle, src, dst,
                                                   num_vertices, dst_device);
        }
        {
            // ...|ooooooo|xxx <- ...|xxxoooo|...
            const int3 src = (int3){0, 0, NGHOST};
            const int3 dst = (int3){0, 0, NGHOST + node->subgrid.n.z};

            const Device src_device = node->devices[0];
            Device dst_device       = node->devices[node->num_devices - 1];

            acDeviceTransferVertexBufferWithOffset(src_device, stream, vtxbuf_handle, src, dst,
                                                   num_vertices, dst_device);
        }
    }
    return AC_SUCCESS;
}

AcResult
acNodeIntegrate(const Node node, const AcReal dt)
{
    acNodeSynchronizeStream(node, STREAM_ALL);
    // xxx|OOO OOOOOOOOO OOO|xxx
    //    ^    ^         ^  ^
    //   n0   n1        n2  n3
    // const int3 n0 = (int3){NGHOST, NGHOST, NGHOST};
    // const int3 n1 = (int3){2 * NGHOST, 2 * NGHOST, 2 * NGHOST};
    // const int3 n2 = node->grid.n;
    // const int3 n3 = n0 + node->grid.n;

    for (int isubstep = 0; isubstep < 3; ++isubstep) {
        acNodeSynchronizeStream(node, STREAM_ALL);
        for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
            local_boundcondstep(node, (Stream)vtxbuf, (VertexBufferHandle)vtxbuf);
        }
        acNodeSynchronizeStream(node, STREAM_ALL);

        // Inner inner
        // #pragma omp parallel for
        for (int i = 0; i < node->num_devices; ++i) {
            const int3 m1 = (int3){2 * NGHOST, 2 * NGHOST, 2 * NGHOST};
            const int3 m2 = node->subgrid.n;
            acDeviceIntegrateSubstep(node->devices[i], STREAM_16, isubstep, m1, m2, dt);
        }

        for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
            acNodeSynchronizeVertexBuffer(node, (Stream)vtxbuf, (VertexBufferHandle)vtxbuf);
            global_boundcondstep(node, (Stream)vtxbuf, (VertexBufferHandle)vtxbuf);
        }
        for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
            acNodeSynchronizeStream(node, (Stream)vtxbuf);
        }

        // #pragma omp parallel for
        for (int i = 0; i < node->num_devices; ++i) { // Front
            const int3 m1 = (int3){NGHOST, NGHOST, NGHOST};
            const int3 m2 = m1 + (int3){node->subgrid.n.x, node->subgrid.n.y, NGHOST};
            acDeviceIntegrateSubstep(node->devices[i], STREAM_0, isubstep, m1, m2, dt);
        }
        // #pragma omp parallel for
        for (int i = 0; i < node->num_devices; ++i) { // Back
            const int3 m1 = (int3){NGHOST, NGHOST, node->subgrid.n.z};
            const int3 m2 = m1 + (int3){node->subgrid.n.x, node->subgrid.n.y, NGHOST};
            acDeviceIntegrateSubstep(node->devices[i], STREAM_1, isubstep, m1, m2, dt);
        }
        // #pragma omp parallel for
        for (int i = 0; i < node->num_devices; ++i) { // Bottom
            const int3 m1 = (int3){NGHOST, NGHOST, 2 * NGHOST};
            const int3 m2 = m1 + (int3){node->subgrid.n.x, NGHOST, node->subgrid.n.z - 2 * NGHOST};
            acDeviceIntegrateSubstep(node->devices[i], STREAM_2, isubstep, m1, m2, dt);
        }
        // #pragma omp parallel for
        for (int i = 0; i < node->num_devices; ++i) { // Top
            const int3 m1 = (int3){NGHOST, node->subgrid.n.y, 2 * NGHOST};
            const int3 m2 = m1 + (int3){node->subgrid.n.x, NGHOST, node->subgrid.n.z - 2 * NGHOST};
            acDeviceIntegrateSubstep(node->devices[i], STREAM_3, isubstep, m1, m2, dt);
        }
        // #pragma omp parallel for
        for (int i = 0; i < node->num_devices; ++i) { // Left
            const int3 m1 = (int3){NGHOST, 2 * NGHOST, 2 * NGHOST};
            const int3 m2 = m1 + (int3){NGHOST, node->subgrid.n.y - 2 * NGHOST,
                                        node->subgrid.n.z - 2 * NGHOST};
            acDeviceIntegrateSubstep(node->devices[i], STREAM_4, isubstep, m1, m2, dt);
        }
        // #pragma omp parallel for
        for (int i = 0; i < node->num_devices; ++i) { // Right
            const int3 m1 = (int3){node->subgrid.n.x, 2 * NGHOST, 2 * NGHOST};
            const int3 m2 = m1 + (int3){NGHOST, node->subgrid.n.y - 2 * NGHOST,
                                        node->subgrid.n.z - 2 * NGHOST};
            acDeviceIntegrateSubstep(node->devices[i], STREAM_5, isubstep, m1, m2, dt);
        }
        acNodeSwapBuffers(node);
    }
    acNodeSynchronizeStream(node, STREAM_ALL);
    return AC_SUCCESS;
}

AcResult
acNodePeriodicBoundcondStep(const Node node, const Stream stream,
                            const VertexBufferHandle vtxbuf_handle)
{
    local_boundcondstep(node, stream, vtxbuf_handle);
    acNodeSynchronizeVertexBuffer(node, stream, vtxbuf_handle);

    // TODO NOTE GLOBAL BOUNDCONDS NOT DONE HERE IF MORE THAN 1 NODE
    global_boundcondstep(node, stream, vtxbuf_handle);
    // WARNING("Global boundconds should not be done here with multinode");

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
simple_final_reduce_scal(const Node node, const ReductionType& rtype, const AcReal* results,
                         const int& n)
{
    AcReal res = results[0];
    for (int i = 1; i < n; ++i) {
        if (rtype == RTYPE_MAX) {
            res = max(res, results[i]);
        }
        else if (rtype == RTYPE_MIN) {
            res = min(res, results[i]);
        }
        else if (rtype == RTYPE_RMS || rtype == RTYPE_RMS_EXP || rtype == RTYPE_SUM) {
            res = sum(res, results[i]);
        }
        else {
            ERROR("Invalid rtype");
        }
    }

    if (rtype == RTYPE_RMS || rtype == RTYPE_RMS_EXP) {
        const AcReal inv_n = AcReal(1.) / (node->grid.n.x * node->grid.n.y * node->grid.n.z);
        res                = sqrt(inv_n * res);
    }
    return res;
}

AcResult
acNodeReduceScal(const Node node, const Stream stream, const ReductionType rtype,
                 const VertexBufferHandle vtxbuf_handle, AcReal* result)
{
    acNodeSynchronizeStream(node, STREAM_ALL);

    AcReal results[node->num_devices];
    // #pragma omp parallel for
    for (int i = 0; i < node->num_devices; ++i) {
        acDeviceReduceScal(node->devices[i], stream, rtype, vtxbuf_handle, &results[i]);
    }

    *result = simple_final_reduce_scal(node, rtype, results, node->num_devices);
    return AC_SUCCESS;
}

AcResult
acNodeReduceVec(const Node node, const Stream stream, const ReductionType rtype,
                const VertexBufferHandle a, const VertexBufferHandle b, const VertexBufferHandle c,
                AcReal* result)
{
    acNodeSynchronizeStream(node, STREAM_ALL);

    AcReal results[node->num_devices];
    // #pragma omp parallel for
    for (int i = 0; i < node->num_devices; ++i) {
        acDeviceReduceVec(node->devices[i], stream, rtype, a, b, c, &results[i]);
    }

    *result = simple_final_reduce_scal(node, rtype, results, node->num_devices);
    return AC_SUCCESS;
}
