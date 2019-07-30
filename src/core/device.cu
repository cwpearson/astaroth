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
 * \brief Brief info.
 *
 * Detailed info.
 *
 */
#include "device.cuh"

#include "errchk.h"

// Device info
#define REGISTERS_PER_THREAD (255)
#define MAX_REGISTERS_PER_BLOCK (65536)
#define MAX_THREADS_PER_BLOCK (1024)
#define WARP_SIZE (32)

typedef struct {
    AcReal* in[NUM_VTXBUF_HANDLES];
    AcReal* out[NUM_VTXBUF_HANDLES];
} VertexBufferArray;

__constant__ AcMeshInfo d_mesh_info;
__constant__ int3 d_multigpu_offset;
__constant__ Grid globalGrid;
#define DCONST_INT(X) (d_mesh_info.int_params[X])
#define DCONST_INT3(X) (d_mesh_info.int3_params[X])
#define DCONST_REAL(X) (d_mesh_info.real_params[X])
#define DCONST_REAL3(X) (d_mesh_info.real3_params[X])
#define DEVICE_VTXBUF_IDX(i, j, k) ((i) + (j)*DCONST_INT(AC_mx) + (k)*DCONST_INT(AC_mxy))
#define DEVICE_1D_COMPDOMAIN_IDX(i, j, k) ((i) + (j)*DCONST_INT(AC_nx) + (k)*DCONST_INT(AC_nxy))
#include "kernels/kernels.cuh"

static dim3 rk3_tpb = (dim3){32, 1, 4};

#if PACKED_DATA_TRANSFERS // Defined in device.cuh
// #include "kernels/pack_unpack.cuh"
#endif

struct device_s {
    int id;
    AcMeshInfo local_config;

    // Concurrency
    cudaStream_t streams[NUM_STREAM_TYPES];

    // Memory
    VertexBufferArray vba;
    AcReal* reduce_scratchpad;
    AcReal* reduce_result;

#if PACKED_DATA_TRANSFERS
// Declare memory for buffers needed for packed data transfers here
// AcReal* data_packing_buffer;
#endif
};

AcResult
printDeviceInfo(const Device device)
{
    const int device_id = device->id;

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);
    printf("--------------------------------------------------\n");
    printf("Device Number: %d\n", device_id);
    const size_t bus_id_max_len = 128;
    char bus_id[bus_id_max_len];
    cudaDeviceGetPCIBusId(bus_id, bus_id_max_len, device_id);
    printf("  PCI bus ID: %s\n", bus_id);
    printf("    Device name: %s\n", props.name);
    printf("    Compute capability: %d.%d\n", props.major, props.minor);

    // Compute
    printf("  Compute\n");
    printf("    Clock rate (GHz): %g\n", props.clockRate / 1e6); // KHz -> GHz
    printf("    Stream processors: %d\n", props.multiProcessorCount);
    printf("    SP to DP flops performance ratio: %d:1\n", props.singleToDoublePrecisionPerfRatio);
    printf(
        "    Compute mode: %d\n",
        (int)props
            .computeMode); // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g7eb25f5413a962faad0956d92bae10d0
    // Memory
    printf("  Global memory\n");
    printf("    Memory Clock Rate (MHz): %d\n", props.memoryClockRate / (1000));
    printf("    Memory Bus Width (bits): %d\n", props.memoryBusWidth);
    printf("    Peak Memory Bandwidth (GiB/s): %f\n",
           2 * (props.memoryClockRate * 1e3) * props.memoryBusWidth / (8. * 1024. * 1024. * 1024.));
    printf("    ECC enabled: %d\n", props.ECCEnabled);

    // Memory usage
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    const size_t used_bytes = total_bytes - free_bytes;
    printf("    Total global mem: %.2f GiB\n", props.totalGlobalMem / (1024.0 * 1024 * 1024));
    printf("    Gmem used (GiB): %.2f\n", used_bytes / (1024.0 * 1024 * 1024));
    printf("    Gmem memory free (GiB): %.2f\n", free_bytes / (1024.0 * 1024 * 1024));
    printf("    Gmem memory total (GiB): %.2f\n", total_bytes / (1024.0 * 1024 * 1024));
    printf("  Caches\n");
    printf("    Local L1 cache supported: %d\n", props.localL1CacheSupported);
    printf("    Global L1 cache supported: %d\n", props.globalL1CacheSupported);
    printf("    L2 size: %d KiB\n", props.l2CacheSize / (1024));
    // MV: props.totalConstMem and props.sharedMemPerBlock cause assembler error
    // MV: while compiling in TIARA gp cluster. Therefore commeted out.
    //!!    printf("    Total const mem: %ld KiB\n", props.totalConstMem / (1024));
    //!!    printf("    Shared mem per block: %ld KiB\n", props.sharedMemPerBlock / (1024));
    printf("  Other\n");
    printf("    Warp size: %d\n", props.warpSize);
    // printf("    Single to double perf. ratio: %dx\n",
    // props.singleToDoublePrecisionPerfRatio); //Not supported with older CUDA
    // versions
    printf("    Stream priorities supported: %d\n", props.streamPrioritiesSupported);
    printf("--------------------------------------------------\n");

    return AC_SUCCESS;
}

static __global__ void
dummy_kernel(void)
{
}

AcResult
createDevice(const int id, const AcMeshInfo device_config, Device* device_handle)
{
    cudaSetDevice(id);
    cudaDeviceReset();

    // Create Device
    struct device_s* device = (struct device_s*)malloc(sizeof(*device));
    ERRCHK_ALWAYS(device);

    device->id           = id;
    device->local_config = device_config;

    // Check that the code was compiled for the proper GPU architecture
    printf("Trying to run a dummy kernel. If this fails, make sure that your\n"
           "device supports the CUDA architecture you are compiling for.\n"
           "Running dummy kernel... ");
    fflush(stdout);
    dummy_kernel<<<1, 1>>>();
    ERRCHK_CUDA_KERNEL_ALWAYS();
    printf("Success!\n");

    // Concurrency
    for (int i = 0; i < NUM_STREAM_TYPES; ++i) {
        cudaStreamCreateWithPriority(&device->streams[i], cudaStreamNonBlocking, 0);
    }

    // Memory
    const size_t vba_size_bytes = acVertexBufferSizeBytes(device_config);
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        ERRCHK_CUDA_ALWAYS(cudaMalloc(&device->vba.in[i], vba_size_bytes));
        ERRCHK_CUDA_ALWAYS(cudaMalloc(&device->vba.out[i], vba_size_bytes));
    }
    ERRCHK_CUDA_ALWAYS(
        cudaMalloc(&device->reduce_scratchpad, acVertexBufferCompdomainSizeBytes(device_config)));
    ERRCHK_CUDA_ALWAYS(cudaMalloc(&device->reduce_result, sizeof(AcReal)));

#if PACKED_DATA_TRANSFERS
// Allocate data required for packed transfers here (cudaMalloc)
#endif

    // Device constants
    ERRCHK_CUDA_ALWAYS(cudaMemcpyToSymbol(d_mesh_info, &device_config, sizeof(device_config), 0,
                                          cudaMemcpyHostToDevice));

    // Multi-GPU offset. This is used to compute globalVertexIdx.
    // Might be better to calculate this in astaroth.cu instead of here, s.t.
    // everything related to the decomposition is limited to the multi-GPU layer
    const int3 multigpu_offset = (int3){0, 0, device->id * device->local_config.int_params[AC_nz]};
    ERRCHK_CUDA_ALWAYS(cudaMemcpyToSymbol(d_multigpu_offset, &multigpu_offset,
                                          sizeof(multigpu_offset), 0, cudaMemcpyHostToDevice));

    printf("Created device %d (%p)\n", device->id, device);
    *device_handle = device;

    // Autoptimize
    if (id == 0)
        autoOptimize(device);

    return AC_SUCCESS;
}

AcResult
destroyDevice(Device device)
{
    cudaSetDevice(device->id);
    printf("Destroying device %d (%p)\n", device->id, device);

    // Memory
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        cudaFree(device->vba.in[i]);
        cudaFree(device->vba.out[i]);
    }
    cudaFree(device->reduce_scratchpad);
    cudaFree(device->reduce_result);

#if PACKED_DATA_TRANSFERS
// Free data required for packed tranfers here (cudaFree)
#endif

    // Concurrency
    for (int i = 0; i < NUM_STREAM_TYPES; ++i) {
        cudaStreamDestroy(device->streams[i]);
    }

    // Destroy Device
    free(device);
    return AC_SUCCESS;
}

AcResult
boundcondStep(const Device device, const StreamType stream_type, const int3& start, const int3& end)
{
    cudaSetDevice(device->id);
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        periodic_boundconds(device->streams[stream_type], start, end, device->vba.in[i]);
    }
    return AC_SUCCESS;
}

AcResult
reduceScal(const Device device, const StreamType stream_type, const ReductionType rtype,
           const VertexBufferHandle vtxbuf_handle, AcReal* result)
{
    cudaSetDevice(device->id);

    const int3 start = (int3){device->local_config.int_params[AC_nx_min],
                              device->local_config.int_params[AC_ny_min],
                              device->local_config.int_params[AC_nz_min]};

    const int3 end = (int3){device->local_config.int_params[AC_nx_max],
                            device->local_config.int_params[AC_ny_max],
                            device->local_config.int_params[AC_nz_max]};

    *result = reduce_scal(device->streams[stream_type], rtype, start, end,
                          device->vba.in[vtxbuf_handle], device->reduce_scratchpad,
                          device->reduce_result);
    return AC_SUCCESS;
}

AcResult
reduceVec(const Device device, const StreamType stream_type, const ReductionType rtype,
          const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
          const VertexBufferHandle vtxbuf2, AcReal* result)
{
    cudaSetDevice(device->id);

    const int3 start = (int3){device->local_config.int_params[AC_nx_min],
                              device->local_config.int_params[AC_ny_min],
                              device->local_config.int_params[AC_nz_min]};

    const int3 end = (int3){device->local_config.int_params[AC_nx_max],
                            device->local_config.int_params[AC_ny_max],
                            device->local_config.int_params[AC_nz_max]};

    *result = reduce_vec(device->streams[stream_type], rtype, start, end, device->vba.in[vtxbuf0],
                         device->vba.in[vtxbuf1], device->vba.in[vtxbuf2],
                         device->reduce_scratchpad, device->reduce_result);
    return AC_SUCCESS;
}

AcResult
rkStep(const Device device, const StreamType stream_type, const int step_number, const int3& start,
       const int3& end, const AcReal dt)
{
    cudaSetDevice(device->id);

    // const dim3 tpb(32, 1, 4);
    const dim3 tpb = rk3_tpb;

    const int3 n = end - start;
    const dim3 bpg((unsigned int)ceil(n.x / AcReal(tpb.x)), //
                   (unsigned int)ceil(n.y / AcReal(tpb.y)), //
                   (unsigned int)ceil(n.z / AcReal(tpb.z)));

    if (step_number == 0)
        solve<0><<<bpg, tpb, 0, device->streams[stream_type]>>>(start, end, device->vba, dt);
    else if (step_number == 1)
        solve<1><<<bpg, tpb, 0, device->streams[stream_type]>>>(start, end, device->vba, dt);
    else
        solve<2><<<bpg, tpb, 0, device->streams[stream_type]>>>(start, end, device->vba, dt);

    ERRCHK_CUDA_KERNEL();

    return AC_SUCCESS;
}

AcResult
synchronize(const Device device, const StreamType stream_type)
{
    cudaSetDevice(device->id);
    if (stream_type == STREAM_ALL) {
        cudaDeviceSynchronize();
    }
    else {
        cudaStreamSynchronize(device->streams[stream_type]);
    }
    return AC_SUCCESS;
}

static AcResult
loadWithOffset(const Device device, const StreamType stream_type, const AcReal* src,
               const size_t bytes, AcReal* dst)
{
    cudaSetDevice(device->id);
    ERRCHK_CUDA(
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, device->streams[stream_type]));
    return AC_SUCCESS;
}

static AcResult
storeWithOffset(const Device device, const StreamType stream_type, const AcReal* src,
                const size_t bytes, AcReal* dst)
{
    cudaSetDevice(device->id);
    ERRCHK_CUDA(
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, device->streams[stream_type]));
    return AC_SUCCESS;
}

AcResult
copyMeshToDevice(const Device device, const StreamType stream_type, const AcMesh& host_mesh,
                 const int3& src, const int3& dst, const int num_vertices)
{
    const size_t src_idx = acVertexBufferIdx(src.x, src.y, src.z, host_mesh.info);
    const size_t dst_idx = acVertexBufferIdx(dst.x, dst.y, dst.z, device->local_config);
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        loadWithOffset(device, stream_type, &host_mesh.vertex_buffer[i][src_idx],
                       num_vertices * sizeof(AcReal), &device->vba.in[i][dst_idx]);
    }
    return AC_SUCCESS;
}

AcResult
copyMeshToHost(const Device device, const StreamType stream_type, const int3& src, const int3& dst,
               const int num_vertices, AcMesh* host_mesh)
{
    const size_t src_idx = acVertexBufferIdx(src.x, src.y, src.z, device->local_config);
    const size_t dst_idx = acVertexBufferIdx(dst.x, dst.y, dst.z, host_mesh->info);
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        storeWithOffset(device, stream_type, &device->vba.in[i][src_idx],
                        num_vertices * sizeof(AcReal), &host_mesh->vertex_buffer[i][dst_idx]);
    }
    return AC_SUCCESS;
}

AcResult
copyMeshDeviceToDevice(const Device src_device, const StreamType stream_type, const int3& src,
                       Device dst_device, const int3& dst, const int num_vertices)
{
    cudaSetDevice(src_device->id);
    const size_t src_idx = acVertexBufferIdx(src.x, src.y, src.z, src_device->local_config);
    const size_t dst_idx = acVertexBufferIdx(dst.x, dst.y, dst.z, dst_device->local_config);

    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        ERRCHK_CUDA(cudaMemcpyPeerAsync(&dst_device->vba.in[i][dst_idx], dst_device->id,
                                        &src_device->vba.in[i][src_idx], src_device->id,
                                        sizeof(src_device->vba.in[i][0]) * num_vertices,
                                        src_device->streams[stream_type]));
    }
    return AC_SUCCESS;
}

AcResult
swapBuffers(const Device device)
{
    cudaSetDevice(device->id);
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        AcReal* tmp        = device->vba.in[i];
        device->vba.in[i]  = device->vba.out[i];
        device->vba.out[i] = tmp;
    }
    return AC_SUCCESS;
}

AcResult
loadDeviceConstant(const Device device, const StreamType stream_type, const AcIntParam param,
                   const int value)
{
    cudaSetDevice(device->id);
    // CUDA 10 apparently creates only a single name for a device constant (d_mesh_info here)
    // and something like d_mesh_info.real_params[] cannot be directly accessed.
    // Therefore we have to obfuscate the code a bit and compute the offset address before
    // invoking cudaMemcpyToSymbol.
    const size_t offset = (size_t)&d_mesh_info.int_params[param] - (size_t)&d_mesh_info;
    ERRCHK_CUDA_ALWAYS(cudaMemcpyToSymbolAsync(d_mesh_info, &value, sizeof(value), offset,
                                               cudaMemcpyHostToDevice,
                                               device->streams[stream_type]));
    return AC_SUCCESS;
}

AcResult
loadDeviceConstant(const Device device, const StreamType stream_type, const AcRealParam param,
                   const AcReal value)
{
    cudaSetDevice(device->id);
    const size_t offset = (size_t)&d_mesh_info.real_params[param] - (size_t)&d_mesh_info;
    ERRCHK_CUDA_ALWAYS(cudaMemcpyToSymbolAsync(d_mesh_info, &value, sizeof(value), offset,
                                               cudaMemcpyHostToDevice,
                                               device->streams[stream_type]));
    return AC_SUCCESS;
}

AcResult
loadGlobalGrid(const Device device, const Grid grid)
{
    cudaSetDevice(device->id);
    ERRCHK_CUDA_ALWAYS(
        cudaMemcpyToSymbol(globalGrid, &grid, sizeof(grid), 0, cudaMemcpyHostToDevice));
    return AC_SUCCESS;
}

AcResult
autoOptimize(const Device device)
{
    cudaSetDevice(device->id);

    // RK3
    const int3 start = (int3){NGHOST, NGHOST, NGHOST};
    const int3 end   = start + (int3){device->local_config.int_params[AC_nx], //
                                    device->local_config.int_params[AC_ny], //
                                    device->local_config.int_params[AC_nz]};

    dim3 best_dims(0, 0, 0);
    float best_time          = INFINITY;
    const int num_iterations = 10;

    for (int z = 1; z <= MAX_THREADS_PER_BLOCK; ++z) {
        for (int y = 1; y <= MAX_THREADS_PER_BLOCK; ++y) {
            for (int x = WARP_SIZE; x <= MAX_THREADS_PER_BLOCK; x += WARP_SIZE) {

                if (x > end.x - start.x || y > end.y - start.y || z > end.z - start.z)
                    break;
                if (x * y * z > MAX_THREADS_PER_BLOCK)
                    break;

                if (x * y * z * REGISTERS_PER_THREAD > MAX_REGISTERS_PER_BLOCK)
                    break;

                if (((x * y * z) % WARP_SIZE) != 0)
                    continue;

                const dim3 tpb(x, y, z);
                const int3 n = end - start;
                const dim3 bpg((unsigned int)ceil(n.x / AcReal(tpb.x)), //
                               (unsigned int)ceil(n.y / AcReal(tpb.y)), //
                               (unsigned int)ceil(n.z / AcReal(tpb.z)));

                cudaDeviceSynchronize();
                if (cudaGetLastError() != cudaSuccess) // resets the error if any
                    continue;

                // printf("(%d, %d, %d)\n", x, y, z);

                cudaEvent_t tstart, tstop;
                cudaEventCreate(&tstart);
                cudaEventCreate(&tstop);

                cudaEventRecord(tstart); // ---------------------------------------- Timing start

                for (int i = 0; i < num_iterations; ++i)
                    solve<2><<<bpg, tpb>>>(start, end, device->vba, FLT_EPSILON);

                cudaEventRecord(tstop); // ----------------------------------------- Timing end
                cudaEventSynchronize(tstop);
                float milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, tstart, tstop);

                ERRCHK_CUDA_KERNEL_ALWAYS();
                if (milliseconds < best_time) {
                    best_time = milliseconds;
                    best_dims = tpb;
                }
            }
        }
    }
#if VERBOSE_PRINTING
    printf(
        "Auto-optimization done. The best threadblock dimensions for rkStep: (%d, %d, %d) %f ms\n",
        best_dims.x, best_dims.y, best_dims.z, double(best_time) / num_iterations);
#endif
    /*
    FILE* fp = fopen("../config/rk3_tbdims.cuh", "w");
    ERRCHK(fp);
    fprintf(fp, "%d, %d, %d\n", best_dims.x, best_dims.y, best_dims.z);
    fclose(fp);
    */

    rk3_tpb = best_dims;
    return AC_SUCCESS;
}

#if PACKED_DATA_TRANSFERS
// Functions for calling packed data transfers
#endif
