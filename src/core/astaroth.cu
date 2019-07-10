/*
    Copyright (C) 2014-2018, Johannes Pekkilae, Miikka Vaeisalae.

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
#include "astaroth.h"
#include "errchk.h"

#include "device.cuh"
#include "math_utils.h"               // sum for reductions
#include "standalone/config_loader.h" // update_config

const char* intparam_names[]  = {AC_FOR_BUILTIN_INT_PARAM_TYPES(AC_GEN_STR)
                                    AC_FOR_USER_INT_PARAM_TYPES(AC_GEN_STR)};
const char* realparam_names[] = {AC_FOR_REAL_PARAM_TYPES(AC_GEN_STR)};
const char* vtxbuf_names[]    = {AC_FOR_VTXBUF_HANDLES(AC_GEN_STR)};

static const int MAX_NUM_DEVICES       = 32;
static int num_devices                 = 0;
static Device devices[MAX_NUM_DEVICES] = {};

static Grid grid; // A grid consists of num_devices subgrids
static Grid subgrid;

static int
gridIdx(const Grid grid, const int3 idx)
{
    return idx.x + idx.y * grid.m.x + idx.z * grid.m.x * grid.m.y;
}

static int3
gridIdx3d(const Grid& grid, const int idx)
{
    return (int3){idx % grid.m.x, (idx % (grid.m.x * grid.m.y)) / grid.m.x,
                  idx / (grid.m.x * grid.m.y)};
}

static void
printInt3(const int3 vec)
{
    printf("(%d, %d, %d)", vec.x, vec.y, vec.z);
}

static Grid
createGrid(const AcMeshInfo& config)
{
    Grid grid;

    grid.m = (int3){config.int_params[AC_mx], config.int_params[AC_my], config.int_params[AC_mz]};
    grid.n = (int3){config.int_params[AC_nx], config.int_params[AC_ny], config.int_params[AC_nz]};

    return grid;
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
acSynchronizeStream(const StreamType stream)
{
    // #pragma omp parallel for
    for (int i = 0; i < num_devices; ++i) {
        synchronize(devices[i], stream);
    }

    return AC_SUCCESS;
}

static AcResult
synchronize_halos(const StreamType stream)
{
    // Exchanges the halos of subgrids
    // After this step, the data within the main grid ranging from
    // (0, 0, NGHOST) -> grid.m.x, grid.m.y, NGHOST + grid.n.z
    // has been synchronized and transferred to appropriate subgrids

    // We loop only to num_devices - 1 since the front and back plate of the grid is not
    // transferred because their contents depend on the boundary conditions.

    // IMPORTANT NOTE: the boundary conditions must be applied before calling this function!
    // I.e. the halos of subgrids must contain up-to-date data!
    // #pragma omp parallel for
    for (int i = 0; i < num_devices - 1; ++i) {
        const int num_vertices = subgrid.m.x * subgrid.m.y * NGHOST;
        // ...|ooooxxx|... -> xxx|ooooooo|...
        {
            const int3 src = (int3){0, 0, subgrid.n.z};
            const int3 dst = (int3){0, 0, 0};
            copyMeshDeviceToDevice(devices[i], stream, src, devices[(i + 1) % num_devices], dst,
                                   num_vertices);
        }
        // ...|ooooooo|xxx <- ...|xxxoooo|...
        {
            const int3 src = (int3){0, 0, NGHOST};
            const int3 dst = (int3){0, 0, NGHOST + subgrid.n.z};
            copyMeshDeviceToDevice(devices[(i + 1) % num_devices], stream, src, devices[i], dst,
                                   num_vertices);
        }
    }
    return AC_SUCCESS;
}

AcResult
acSynchronizeMesh(void)
{
    acSynchronizeStream(STREAM_ALL);
    synchronize_halos(STREAM_DEFAULT);
    acSynchronizeStream(STREAM_ALL);

    return AC_SUCCESS;
}

AcResult
acInit(const AcMeshInfo& config)
{
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
    update_config(&subgrid_config);
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
        loadGlobalGrid(devices[i], grid);
        printDeviceInfo(devices[i]);
    }

    acSynchronizeStream(STREAM_ALL);
    return AC_SUCCESS;
}

AcResult
acQuit(void)
{
    acSynchronizeStream(STREAM_ALL);

    for (int i = 0; i < num_devices; ++i) {
        destroyDevice(devices[i]);
    }
    return AC_SUCCESS;
}

AcResult
acIntegrateStepWithOffsetAsync(const int& isubstep, const AcReal& dt, const int3& start,
                               const int3& end, const StreamType stream)
{
    // See the beginning of the file for an explanation of the index mapping
    // #pragma omp parallel for
    for (int i = 0; i < num_devices; ++i) {
        // DECOMPOSITION OFFSET HERE
        const int3 d0 = (int3){NGHOST, NGHOST, NGHOST + i * subgrid.n.z};
        const int3 d1 = d0 + (int3){subgrid.n.x, subgrid.n.y, subgrid.n.z};

        const int3 da = max(start, d0);
        const int3 db = min(end, d1);

        if (db.z >= da.z) {
            const int3 da_local = da - (int3){0, 0, i * subgrid.n.z};
            const int3 db_local = db - (int3){0, 0, i * subgrid.n.z};
            rkStep(devices[i], stream, isubstep, da_local, db_local, dt);
        }
    }
    return AC_SUCCESS;
}

AcResult
acIntegrateStepWithOffset(const int& isubstep, const AcReal& dt, const int3& start, const int3& end)
{
    return acIntegrateStepWithOffsetAsync(isubstep, dt, start, end, STREAM_DEFAULT);
}

AcResult
acIntegrateStepAsync(const int& isubstep, const AcReal& dt, const StreamType stream)
{
    const int3 start = (int3){NGHOST, NGHOST, NGHOST};
    const int3 end   = start + grid.n;
    return acIntegrateStepWithOffsetAsync(isubstep, dt, start, end, stream);
}

AcResult
acIntegrateStep(const int& isubstep, const AcReal& dt)
{
    return acIntegrateStepAsync(isubstep, dt, STREAM_DEFAULT);
}

static AcResult
local_boundcondstep(const StreamType stream)
{
    if (num_devices == 1) {
        boundcondStep(devices[0], stream, (int3){0, 0, 0}, subgrid.m);
    }
    else {
        // Local boundary conditions
        // #pragma omp parallel for
        for (int i = 0; i < num_devices; ++i) {
            const int3 d0 = (int3){0, 0, NGHOST}; // DECOMPOSITION OFFSET HERE
            const int3 d1 = (int3){subgrid.m.x, subgrid.m.y, d0.z + subgrid.n.z};
            boundcondStep(devices[i], stream, d0, d1);
        }
    }
    return AC_SUCCESS;
}

static AcResult
global_boundcondstep(const StreamType stream)
{
    if (num_devices > 1) {
        // With periodic boundary conditions we exchange the front and back plates of the
        // grid. The exchange is done between the first and last device (0 and num_devices - 1).
        const int num_vertices = subgrid.m.x * subgrid.m.y * NGHOST;
        // ...|ooooxxx|... -> xxx|ooooooo|...
        {
            const int3 src = (int3){0, 0, subgrid.n.z};
            const int3 dst = (int3){0, 0, 0};
            copyMeshDeviceToDevice(devices[num_devices - 1], stream, src, devices[0], dst,
                                   num_vertices);
        }
        // ...|ooooooo|xxx <- ...|xxxoooo|...
        {
            const int3 src = (int3){0, 0, NGHOST};
            const int3 dst = (int3){0, 0, NGHOST + subgrid.n.z};
            copyMeshDeviceToDevice(devices[0], stream, src, devices[num_devices - 1], dst,
                                   num_vertices);
        }
    }
    return AC_SUCCESS;
}

AcResult
acBoundcondStepAsync(const StreamType stream)
{
    ERRCHK_ALWAYS(stream < NUM_STREAM_TYPES);

    local_boundcondstep(stream);
    acSynchronizeStream(stream);
    global_boundcondstep(stream);
    synchronize_halos(stream);
    acSynchronizeStream(stream);
    return AC_SUCCESS;
}

AcResult
acBoundcondStep(void)
{
    return acBoundcondStepAsync(STREAM_DEFAULT);
}

static AcResult
swap_buffers(void)
{
    // #pragma omp parallel for
    for (int i = 0; i < num_devices; ++i) {
        swapBuffers(devices[i]);
    }
    return AC_SUCCESS;
}

AcResult
acIntegrate(const AcReal& dt)
{
    acSynchronizeStream(STREAM_ALL);
    for (int isubstep = 0; isubstep < 3; ++isubstep) {
        acIntegrateStep(isubstep, dt); // Note: boundaries must be initialized.
        swap_buffers();
        acBoundcondStep();
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

AcReal
acReduceScal(const ReductionType& rtype, const VertexBufferHandle& vtxbuffer_handle)
{
    acSynchronizeStream(STREAM_ALL);

    AcReal results[num_devices];
    // #pragma omp parallel for
    for (int i = 0; i < num_devices; ++i) {
        reduceScal(devices[i], STREAM_DEFAULT, rtype, vtxbuffer_handle, &results[i]);
    }

    return simple_final_reduce_scal(rtype, results, num_devices);
}

AcReal
acReduceVec(const ReductionType& rtype, const VertexBufferHandle& a, const VertexBufferHandle& b,
            const VertexBufferHandle& c)
{
    acSynchronizeStream(STREAM_ALL);

    AcReal results[num_devices];
    // #pragma omp parallel for
    for (int i = 0; i < num_devices; ++i) {
        reduceVec(devices[i], STREAM_DEFAULT, rtype, a, b, c, &results[i]);
    }

    return simple_final_reduce_scal(rtype, results, num_devices);
}

AcResult
acLoadWithOffsetAsync(const AcMesh& host_mesh, const int3& src, const int num_vertices,
                      const StreamType stream)
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
            copyMeshToDevice(devices[i], stream, host_mesh, da, da_local, copy_cells);
        }
        // printf("\n");
    }
    return AC_SUCCESS;
}

AcResult
acLoadWithOffset(const AcMesh& host_mesh, const int3& src, const int num_vertices)
{
    return acLoadWithOffsetAsync(host_mesh, src, num_vertices, STREAM_DEFAULT);
}

AcResult
acLoad(const AcMesh& host_mesh)
{
    acLoadWithOffset(host_mesh, (int3){0, 0, 0}, AC_VTXBUF_SIZE(host_mesh.info));
    acSynchronizeStream(STREAM_ALL);
    return AC_SUCCESS;
}

AcResult
acStoreWithOffsetAsync(const int3& src, const int num_vertices, AcMesh* host_mesh,
                       const StreamType stream)
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
        if (db.z >= da.z) {
            const int copy_cells = gridIdx(subgrid, db) - gridIdx(subgrid, da);
            // DECOMPOSITION OFFSET HERE
            const int3 da_local = (int3){da.x, da.y, da.z - i * grid.n.z / num_devices};
            copyMeshToHost(devices[i], stream, da_local, da, copy_cells, host_mesh);
        }
    }
    return AC_SUCCESS;
}

AcResult
acStoreWithOffset(const int3& src, const int num_vertices, AcMesh* host_mesh)
{
    return acStoreWithOffsetAsync(src, num_vertices, host_mesh, STREAM_DEFAULT);
}

AcResult
acStore(AcMesh* host_mesh)
{
    acStoreWithOffset((int3){0, 0, 0}, AC_VTXBUF_SIZE(host_mesh->info), host_mesh);
    acSynchronizeStream(STREAM_ALL);
    return AC_SUCCESS;
}

AcResult
acLoadDeviceConstantAsync(const AcRealParam param, const AcReal value, const StreamType stream)
{
    // #pragma omp parallel for
    for (int i = 0; i < num_devices; ++i) {
        loadDeviceConstant(devices[i], stream, param, value);
    }
    return AC_SUCCESS;
}

AcResult
acLoadDeviceConstant(const AcRealParam param, const AcReal value)
{
    return acLoadDeviceConstantAsync(param, value, STREAM_DEFAULT);
}
