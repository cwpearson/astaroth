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

/**
 * @file
 * \brief Brief info.
 *
 * Detailed info.
 *
 */
#include "astaroth_device.h"

#include <string.h> // memcpy

#include "errchk.h"
#include "math_utils.h"

#include "kernels/common.cuh"

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))

struct device_s {
    int id;
    AcMeshInfo local_config;

    // Concurrency
    cudaStream_t streams[NUM_STREAMS];

    // Memory
    VertexBufferArray vba;
    AcReal* reduce_scratchpad;
    AcReal* reduce_result;
};

#include "kernels/boundconds.cuh"
#include "kernels/integration.cuh"
#include "kernels/reductions.cuh"

#if PACKED_DATA_TRANSFERS // TODO DEPRECATED, see AC_MPI_ENABLED instead
// #include "kernels/pack_unpack.cuh"
#endif

AcResult
acDeviceCreate(const int id, const AcMeshInfo device_config, Device* device_handle)
{
    cudaSetDevice(id);
    // cudaDeviceReset(); // Would be good for safety, but messes stuff up if we want to emulate
    // multiple devices with a single GPU

    // Create Device
    struct device_s* device = (struct device_s*)malloc(sizeof(*device));
    ERRCHK_ALWAYS(device);

    device->id           = id;
    device->local_config = device_config;
    acDevicePrintInfo(device);

    // Check that the code was compiled for the proper GPU architecture
    printf("Trying to run a dummy kernel. If this fails, make sure that your\n"
           "device supports the CUDA architecture you are compiling for.\n"
           "Running dummy kernel... ");
    fflush(stdout);
    acKernelDummy();
    printf("Success!\n");

    // Concurrency
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreateWithPriority(&device->streams[i], cudaStreamNonBlocking, i);
    }

    // Memory
    // VBA in/out
    const size_t vba_size_bytes = acVertexBufferSizeBytes(device_config);
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&device->vba.in[i], vba_size_bytes));
        ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&device->vba.out[i], vba_size_bytes));
    }
    // VBA Profiles
    const size_t profile_size_bytes = sizeof(AcReal) * max(device_config.int_params[AC_mx],
                                                           max(device_config.int_params[AC_my],
                                                               device_config.int_params[AC_mz]));
    for (int i = 0; i < NUM_SCALARARRAY_HANDLES; ++i) {
        ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&device->vba.profiles[i], profile_size_bytes));
    }

    // Reductions
    ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&device->reduce_scratchpad,
                                  acVertexBufferCompdomainSizeBytes(device_config)));
    ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&device->reduce_result, sizeof(AcReal)));

    // Device constants
    acDeviceLoadMeshInfo(device, STREAM_DEFAULT, device_config);

    printf("Created device %d (%p)\n", device->id, device);
    *device_handle = device;

    // Autoptimize
    if (id == 0) {
        acDeviceAutoOptimize(device);
    }

    return AC_SUCCESS;
}

AcResult
acDeviceDestroy(Device device)
{
    cudaSetDevice(device->id);
    printf("Destroying device %d (%p)\n", device->id, device);
    acDeviceSynchronizeStream(device, STREAM_ALL);

    // Memory
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        cudaFree(device->vba.in[i]);
        cudaFree(device->vba.out[i]);
    }
    for (int i = 0; i < NUM_SCALARARRAY_HANDLES; ++i) {
        cudaFree(device->vba.profiles[i]);
    }

    cudaFree(device->reduce_scratchpad);
    cudaFree(device->reduce_result);

    // Concurrency
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamDestroy(device->streams[i]);
    }

    // Destroy Device
    free(device);
    return AC_SUCCESS;
}

AcResult
acDevicePrintInfo(const Device device)
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

AcResult
acDeviceAutoOptimize(const Device device)
{
    cudaSetDevice(device->id);
    const int3 start = (int3){
        device->local_config.int_params[AC_nx_min],
        device->local_config.int_params[AC_ny_min],
        device->local_config.int_params[AC_nz_min],
    };
    const int3 end = (int3){
        device->local_config.int_params[AC_nx_max],
        device->local_config.int_params[AC_ny_max],
        device->local_config.int_params[AC_nz_max],
    };
    return acKernelAutoOptimizeIntegration(start, end, device->vba);
}

AcResult
acDeviceSynchronizeStream(const Device device, const Stream stream)
{
    cudaSetDevice(device->id);
    if (stream == STREAM_ALL) {
        cudaDeviceSynchronize();
    }
    else {
        cudaStreamSynchronize(device->streams[stream]);
    }
    return AC_SUCCESS;
}

AcResult
acDeviceSwapBuffers(const Device device)
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
acDeviceLoadScalarUniform(const Device device, const Stream stream, const AcRealParam param,
                          const AcReal value)
{
    cudaSetDevice(device->id);

    if (param >= NUM_REAL_PARAMS)
        return AC_FAILURE;

    const size_t offset = (size_t)&d_mesh_info.real_params[param] - (size_t)&d_mesh_info;
    ERRCHK_CUDA(cudaMemcpyToSymbolAsync(&d_mesh_info, &value, sizeof(value), offset,
                                        cudaMemcpyHostToDevice, device->streams[stream]));
    return AC_SUCCESS;
}

AcResult
acDeviceLoadVectorUniform(const Device device, const Stream stream, const AcReal3Param param,
                          const AcReal3 value)
{
    cudaSetDevice(device->id);

    if (param >= NUM_REAL3_PARAMS || !NUM_REAL3_PARAMS)
        return AC_FAILURE;

    const size_t offset = (size_t)&d_mesh_info.real3_params[param] - (size_t)&d_mesh_info;
    ERRCHK_CUDA(cudaMemcpyToSymbolAsync(&d_mesh_info, &value, sizeof(value), offset,
                                        cudaMemcpyHostToDevice, device->streams[stream]));
    return AC_SUCCESS;
}

AcResult
acDeviceLoadIntUniform(const Device device, const Stream stream, const AcIntParam param,
                       const int value)
{
    cudaSetDevice(device->id);

    if (param >= NUM_INT_PARAMS)
        return AC_FAILURE;

    const size_t offset = (size_t)&d_mesh_info.int_params[param] - (size_t)&d_mesh_info;
    ERRCHK_CUDA(cudaMemcpyToSymbolAsync(&d_mesh_info, &value, sizeof(value), offset,
                                        cudaMemcpyHostToDevice, device->streams[stream]));
    return AC_SUCCESS;
}

AcResult
acDeviceLoadInt3Uniform(const Device device, const Stream stream, const AcInt3Param param,
                        const int3 value)
{
    cudaSetDevice(device->id);

    if (param >= NUM_INT3_PARAMS)
        return AC_FAILURE;

    const size_t offset = (size_t)&d_mesh_info.int3_params[param] - (size_t)&d_mesh_info;
    ERRCHK_CUDA(cudaMemcpyToSymbolAsync(&d_mesh_info, &value, sizeof(value), offset,
                                        cudaMemcpyHostToDevice, device->streams[stream]));
    return AC_SUCCESS;
}

AcResult
acDeviceLoadScalarArray(const Device device, const Stream stream, const ScalarArrayHandle handle,
                        const size_t start, const AcReal* data, const size_t num)
{
    cudaSetDevice(device->id);

    if (handle >= NUM_SCALARARRAY_HANDLES || !NUM_SCALARARRAY_HANDLES)
        return AC_FAILURE;

    ERRCHK((int)(start + num) <= max(device->local_config.int_params[AC_mx],
                                     max(device->local_config.int_params[AC_my],
                                         device->local_config.int_params[AC_mz])));
    ERRCHK_ALWAYS(handle < NUM_SCALARARRAY_HANDLES);
    ERRCHK_CUDA(cudaMemcpyAsync(&device->vba.profiles[handle][start], data, sizeof(data[0]) * num,
                                cudaMemcpyHostToDevice, device->streams[stream]));
    return AC_SUCCESS;
}

AcResult
acDeviceLoadMeshInfo(const Device device, const Stream stream, const AcMeshInfo device_config)
{
    cudaSetDevice(device->id);

    ERRCHK_ALWAYS(device_config.int_params[AC_nx] == device->local_config.int_params[AC_nx]);
    ERRCHK_ALWAYS(device_config.int_params[AC_ny] == device->local_config.int_params[AC_ny]);
    ERRCHK_ALWAYS(device_config.int_params[AC_nz] == device->local_config.int_params[AC_nz]);
    ERRCHK_ALWAYS(device_config.int_params[AC_multigpu_offset] ==
                  device->local_config.int_params[AC_multigpu_offset]);

    ERRCHK_CUDA_ALWAYS(cudaMemcpyToSymbolAsync(&d_mesh_info, &device_config, sizeof(device_config),
                                               0, cudaMemcpyHostToDevice, device->streams[stream]));
    return AC_SUCCESS;
}

AcResult
acDeviceLoadVertexBufferWithOffset(const Device device, const Stream stream, const AcMesh host_mesh,
                                   const VertexBufferHandle vtxbuf_handle, const int3 src,
                                   const int3 dst, const int num_vertices)
{
    cudaSetDevice(device->id);
    const size_t src_idx = acVertexBufferIdx(src.x, src.y, src.z, host_mesh.info);
    const size_t dst_idx = acVertexBufferIdx(dst.x, dst.y, dst.z, device->local_config);

    const AcReal* src_ptr = &host_mesh.vertex_buffer[vtxbuf_handle][src_idx];
    AcReal* dst_ptr       = &device->vba.in[vtxbuf_handle][dst_idx];
    const size_t bytes    = num_vertices * sizeof(src_ptr[0]);

    ERRCHK_CUDA(                                                                                  //
        cudaMemcpyAsync(dst_ptr, src_ptr, bytes, cudaMemcpyHostToDevice, device->streams[stream]) //
    );

    return AC_SUCCESS;
}

AcResult
acDeviceLoadMeshWithOffset(const Device device, const Stream stream, const AcMesh host_mesh,
                           const int3 src, const int3 dst, const int num_vertices)
{
    WARNING("This function is deprecated");
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        acDeviceLoadVertexBufferWithOffset(device, stream, host_mesh, (VertexBufferHandle)i, src,
                                           dst, num_vertices);
    }
    return AC_SUCCESS;
}

AcResult
acDeviceLoadVertexBuffer(const Device device, const Stream stream, const AcMesh host_mesh,
                         const VertexBufferHandle vtxbuf_handle)
{
    const int3 src            = (int3){0, 0, 0};
    const int3 dst            = src;
    const size_t num_vertices = acVertexBufferSize(device->local_config);
    acDeviceLoadVertexBufferWithOffset(device, stream, host_mesh, vtxbuf_handle, src, dst,
                                       num_vertices);

    return AC_SUCCESS;
}

AcResult
acDeviceLoadMesh(const Device device, const Stream stream, const AcMesh host_mesh)
{
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        acDeviceLoadVertexBuffer(device, stream, host_mesh, (VertexBufferHandle)i);
    }

    return AC_SUCCESS;
}

AcResult
acDeviceStoreVertexBufferWithOffset(const Device device, const Stream stream,
                                    const VertexBufferHandle vtxbuf_handle, const int3 src,
                                    const int3 dst, const int num_vertices, AcMesh* host_mesh)
{
    cudaSetDevice(device->id);
    const size_t src_idx = acVertexBufferIdx(src.x, src.y, src.z, device->local_config);
    const size_t dst_idx = acVertexBufferIdx(dst.x, dst.y, dst.z, host_mesh->info);

    const AcReal* src_ptr = &device->vba.in[vtxbuf_handle][src_idx];
    AcReal* dst_ptr       = &host_mesh->vertex_buffer[vtxbuf_handle][dst_idx];
    const size_t bytes    = num_vertices * sizeof(src_ptr[0]);

    ERRCHK_CUDA(                                                                                  //
        cudaMemcpyAsync(dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToHost, device->streams[stream]) //
    );

    return AC_SUCCESS;
}

AcResult
acDeviceStoreMeshWithOffset(const Device device, const Stream stream, const int3 src,
                            const int3 dst, const int num_vertices, AcMesh* host_mesh)
{
    WARNING("This function is deprecated");
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        acDeviceStoreVertexBufferWithOffset(device, stream, (VertexBufferHandle)i, src, dst,
                                            num_vertices, host_mesh);
    }

    return AC_SUCCESS;
}

AcResult
acDeviceStoreVertexBuffer(const Device device, const Stream stream,
                          const VertexBufferHandle vtxbuf_handle, AcMesh* host_mesh)
{
    int3 src                  = (int3){0, 0, 0};
    int3 dst                  = src;
    const size_t num_vertices = acVertexBufferSize(device->local_config);

    acDeviceStoreVertexBufferWithOffset(device, stream, vtxbuf_handle, src, dst, num_vertices,
                                        host_mesh);

    return AC_SUCCESS;
}

AcResult
acDeviceStoreMesh(const Device device, const Stream stream, AcMesh* host_mesh)
{
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        acDeviceStoreVertexBuffer(device, stream, (VertexBufferHandle)i, host_mesh);
    }

    return AC_SUCCESS;
}

AcResult
acDeviceTransferVertexBufferWithOffset(const Device src_device, const Stream stream,
                                       const VertexBufferHandle vtxbuf_handle, const int3 src,
                                       const int3 dst, const int num_vertices, Device dst_device)
{
    cudaSetDevice(src_device->id);
    const size_t src_idx = acVertexBufferIdx(src.x, src.y, src.z, src_device->local_config);
    const size_t dst_idx = acVertexBufferIdx(dst.x, dst.y, dst.z, dst_device->local_config);

    const AcReal* src_ptr = &src_device->vba.in[vtxbuf_handle][src_idx];
    AcReal* dst_ptr       = &dst_device->vba.in[vtxbuf_handle][dst_idx];
    const size_t bytes    = num_vertices * sizeof(src_ptr[0]);

    ERRCHK_CUDA(cudaMemcpyPeerAsync(dst_ptr, dst_device->id, src_ptr, src_device->id, bytes,
                                    src_device->streams[stream]));
    return AC_SUCCESS;
}

AcResult
acDeviceTransferMeshWithOffset(const Device src_device, const Stream stream, const int3 src,
                               const int3 dst, const int num_vertices, Device dst_device)
{
    WARNING("This function is deprecated");
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        acDeviceTransferVertexBufferWithOffset(src_device, stream, (VertexBufferHandle)i, src, dst,
                                               num_vertices, dst_device);
    }
    return AC_SUCCESS;
}

AcResult
acDeviceTransferVertexBuffer(const Device src_device, const Stream stream,
                             const VertexBufferHandle vtxbuf_handle, Device dst_device)
{
    int3 src                  = (int3){0, 0, 0};
    int3 dst                  = src;
    const size_t num_vertices = acVertexBufferSize(src_device->local_config);

    acDeviceTransferVertexBufferWithOffset(src_device, stream, vtxbuf_handle, src, dst,
                                           num_vertices, dst_device);
    return AC_SUCCESS;
}

AcResult
acDeviceTransferMesh(const Device src_device, const Stream stream, Device dst_device)
{
    WARNING("This function is deprecated");
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        acDeviceTransferVertexBuffer(src_device, stream, (VertexBufferHandle)i, dst_device);
    }
    return AC_SUCCESS;
}

AcResult
acDeviceIntegrateSubstep(const Device device, const Stream stream, const int step_number,
                         const int3 start, const int3 end, const AcReal dt)
{
    cudaSetDevice(device->id);
    acDeviceLoadScalarUniform(device, stream, AC_dt, dt);
    return acKernelIntegrateSubstep(device->streams[stream], step_number, start, end, device->vba);
}

AcResult
acDevicePeriodicBoundcondStep(const Device device, const Stream stream,
                              const VertexBufferHandle vtxbuf_handle, const int3 start,
                              const int3 end)
{
    cudaSetDevice(device->id);
    return acKernelPeriodicBoundconds(device->streams[stream], start, end,
                                      device->vba.in[vtxbuf_handle]);
}

AcResult
acDevicePeriodicBoundconds(const Device device, const Stream stream, const int3 start,
                           const int3 end)
{
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        acDevicePeriodicBoundcondStep(device, stream, (VertexBufferHandle)i, start, end);
    }
    return AC_SUCCESS;
}

AcResult
acDeviceReduceScal(const Device device, const Stream stream, const ReductionType rtype,
                   const VertexBufferHandle vtxbuf_handle, AcReal* result)
{
    cudaSetDevice(device->id);

    const int3 start = (int3){device->local_config.int_params[AC_nx_min],
                              device->local_config.int_params[AC_ny_min],
                              device->local_config.int_params[AC_nz_min]};

    const int3 end = (int3){device->local_config.int_params[AC_nx_max],
                            device->local_config.int_params[AC_ny_max],
                            device->local_config.int_params[AC_nz_max]};

    *result = acKernelReduceScal(device->streams[stream], rtype, start, end,
                                 device->vba.in[vtxbuf_handle], device->reduce_scratchpad,
                                 device->reduce_result);
    return AC_SUCCESS;
}

AcResult
acDeviceReduceVec(const Device device, const Stream stream, const ReductionType rtype,
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

    *result = acKernelReduceVec(device->streams[stream], rtype, start, end, device->vba.in[vtxbuf0],
                                device->vba.in[vtxbuf1], device->vba.in[vtxbuf2],
                                device->reduce_scratchpad, device->reduce_result);
    return AC_SUCCESS;
}

#if PACKED_DATA_TRANSFERS // TODO DEPRECATED, see AC_MPI_ENABLED instead
// Functions for calling packed data transfers
#endif

#if AC_MPI_ENABLED == 0
AcResult
acDeviceRunMPITest(void)
{
    WARNING("MPI was not enabled but acDeviceRunMPITest() was called");
    return AC_FAILURE;
}
#else // MPI_ENABLED ///////////////////////////////////////////////////////////////////////////////
#include <mpi.h>

// Kernels
#include "kernels/packing.cuh"

// From Astaroth Utils
#include "src/utils/config_loader.h"
#include "src/utils/memory.h"
#include "src/utils/timer_hires.h"
#include "src/utils/verification.h"

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))

static int
mod(const int a, const int b)
{
    const int r = a % b;
    return r < 0 ? r + b : r;
}

static int
getPid(const int3 pid, const int3 decomposition)
{
    return mod(pid.x, decomposition.x) +                   //
           mod(pid.y, decomposition.y) * decomposition.x + //
           mod(pid.z, decomposition.z) * decomposition.x * decomposition.y;
}

static int3
getPid3D(const int pid, const int3 decomposition)
{
    const int3 pid3d = (int3){
        mod(pid, decomposition.x),
        mod(pid / decomposition.x, decomposition.y),
        (pid / (decomposition.x * decomposition.y)),
    };
    return pid3d;
}

static int3
decompose(const int target)
{
    int decomposition[] = {1, 1, 1};

    int axis = 0;
    while (decomposition[0] * decomposition[1] * decomposition[2] < target) {
        ++decomposition[axis];
        axis = (axis + 1) % 3;
    }

    const int found = decomposition[0] * decomposition[1] * decomposition[2];
    if (found != target) {
        fprintf(stderr, "Invalid number of processes! Cannot decompose the problem domain!\n");
        fprintf(stderr, "Target nprocs: %d. Next allowed: %d\n", target, found);
        ERROR("Invalid nprocs");
        return (int3){-1, -1, -1};
    }
    else {
        return (int3){decomposition[0], decomposition[1], decomposition[2]};
    }
}

static PackedData
acCreatePackedData(const int3 dims)
{
    PackedData data = {};

    data.dims = dims;

    const size_t bytes = dims.x * dims.y * dims.z * sizeof(data.data[0]) * NUM_VTXBUF_HANDLES;
    ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&data.data, bytes));

    return data;
}

static AcResult
acDestroyPackedData(PackedData* data)
{
    data->dims = (int3){-1, -1, -1};
    cudaFree(data->data);
    data->data = NULL;

    return AC_SUCCESS;
}

static PackedData
acCreatePackedDataHost(const int3 dims)
{
    PackedData data = {};

    data.dims = dims;

    const size_t bytes = dims.x * dims.y * dims.z * sizeof(data.data[0]) * NUM_VTXBUF_HANDLES;
    data.data          = (AcReal*)malloc(bytes);
    ERRCHK_ALWAYS(data.data);

    return data;
}

static void
acTransferPackedDataToHost(const Device device, const cudaStream_t stream, const PackedData ddata,
                           PackedData* hdata)
{
    cudaSetDevice(device->id);

    const size_t bytes = ddata.dims.x * ddata.dims.y * ddata.dims.z * sizeof(ddata.data[0]) *
                         NUM_VTXBUF_HANDLES;
    ERRCHK_CUDA(cudaMemcpyAsync(hdata->data, ddata.data, bytes, cudaMemcpyDeviceToHost, stream));
}

static void
acTransferPackedDataToDevice(const Device device, const cudaStream_t stream, const PackedData hdata,
                             PackedData* ddata)
{
    cudaSetDevice(device->id);

    const size_t bytes = hdata.dims.x * hdata.dims.y * hdata.dims.z * sizeof(hdata.data[0]) *
                         NUM_VTXBUF_HANDLES;
    ERRCHK_CUDA(cudaMemcpyAsync(ddata->data, hdata.data, bytes, cudaMemcpyHostToDevice, stream));
}

static AcResult
acDestroyPackedDataHost(PackedData* data)
{
    data->dims = (int3){-1, -1, -1};
    free(data->data);
    data->data = NULL;

    return AC_SUCCESS;
}

// TODO: do with packed data
static AcResult
acDeviceDistributeMeshMPI(const AcMesh src, const int3 decomposition, AcMesh* dst)
{
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Distributing mesh...\n");
    fflush(stdout);

    MPI_Datatype datatype = MPI_FLOAT;
    if (sizeof(AcReal) == 8)
        datatype = MPI_DOUBLE;

    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    ERRCHK_ALWAYS(dst);

    // Submesh nn
    const int3 nn = (int3){
        dst->info.int_params[AC_nx],
        dst->info.int_params[AC_ny],
        dst->info.int_params[AC_nz],
    };

    // Send to self
    if (pid == 0) {
        for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
            // For pencils
            for (int k = NGHOST; k < NGHOST + nn.z; ++k) {
                for (int j = NGHOST; j < NGHOST + nn.y; ++j) {
                    const int i       = NGHOST;
                    const int count   = nn.x;
                    const int src_idx = acVertexBufferIdx(i, j, k, src.info);
                    const int dst_idx = acVertexBufferIdx(i, j, k, dst->info);
                    memcpy(&dst->vertex_buffer[vtxbuf][dst_idx], //
                           &src.vertex_buffer[vtxbuf][src_idx],  //
                           count * sizeof(src.vertex_buffer[i][0]));
                }
            }
        }
    }

    for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
        // For pencils
        for (int k = NGHOST; k < NGHOST + nn.z; ++k) {
            for (int j = NGHOST; j < NGHOST + nn.y; ++j) {
                const int i     = NGHOST;
                const int count = nn.x;

                if (pid != 0) {
                    const int dst_idx = acVertexBufferIdx(i, j, k, dst->info);
                    // Recv
                    MPI_Status status;
                    MPI_Recv(&dst->vertex_buffer[vtxbuf][dst_idx], count, datatype, 0, 0,
                             MPI_COMM_WORLD, &status);
                }
                else {
                    for (int tgt_pid = 1; tgt_pid < nprocs; ++tgt_pid) {
                        const int3 tgt_pid3d = getPid3D(tgt_pid, decomposition);
                        const int src_idx    = acVertexBufferIdx(i + tgt_pid3d.x * nn.x, //
                                                              j + tgt_pid3d.y * nn.y, //
                                                              k + tgt_pid3d.z * nn.z, //
                                                              src.info);

                        // Send
                        MPI_Send(&src.vertex_buffer[vtxbuf][src_idx], count, datatype, tgt_pid, 0,
                                 MPI_COMM_WORLD);
                    }
                }
            }
        }
    }
    return AC_SUCCESS;
}

// TODO: do with packed data
static AcResult
acDeviceGatherMeshMPI(const AcMesh src, const int3 decomposition, AcMesh* dst)
{
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Gathering mesh...\n");
    fflush(stdout);

    MPI_Datatype datatype = MPI_FLOAT;
    if (sizeof(AcReal) == 8)
        datatype = MPI_DOUBLE;

    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (pid == 0)
        ERRCHK_ALWAYS(dst);

    // Submesh nn
    const int3 nn = (int3){
        src.info.int_params[AC_nx],
        src.info.int_params[AC_ny],
        src.info.int_params[AC_nz],
    };

    // Submesh mm
    const int3 mm = (int3){
        src.info.int_params[AC_mx],
        src.info.int_params[AC_my],
        src.info.int_params[AC_mz],
    };

    // Send to self
    if (pid == 0) {
        for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
            // For pencils
            for (int k = 0; k < mm.z; ++k) {
                for (int j = 0; j < mm.y; ++j) {
                    const int i       = 0;
                    const int count   = mm.x;
                    const int src_idx = acVertexBufferIdx(i, j, k, src.info);
                    const int dst_idx = acVertexBufferIdx(i, j, k, dst->info);
                    memcpy(&dst->vertex_buffer[vtxbuf][dst_idx], //
                           &src.vertex_buffer[vtxbuf][src_idx],  //
                           count * sizeof(src.vertex_buffer[i][0]));
                }
            }
        }
    }

    for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
        // For pencils
        for (int k = 0; k < mm.z; ++k) {
            for (int j = 0; j < mm.y; ++j) {
                const int i     = 0;
                const int count = mm.x;

                if (pid != 0) {
                    // Send
                    const int src_idx = acVertexBufferIdx(i, j, k, src.info);
                    MPI_Send(&src.vertex_buffer[vtxbuf][src_idx], count, datatype, 0, 0,
                             MPI_COMM_WORLD);
                }
                else {
                    for (int tgt_pid = 1; tgt_pid < nprocs; ++tgt_pid) {
                        const int3 tgt_pid3d = getPid3D(tgt_pid, decomposition);
                        const int dst_idx    = acVertexBufferIdx(i + tgt_pid3d.x * nn.x, //
                                                              j + tgt_pid3d.y * nn.y, //
                                                              k + tgt_pid3d.z * nn.z, //
                                                              dst->info);

                        // Recv
                        MPI_Status status;
                        MPI_Recv(&dst->vertex_buffer[vtxbuf][dst_idx], count, datatype, tgt_pid, 0,
                                 MPI_COMM_WORLD, &status);
                    }
                }
            }
        }
    }
    return AC_SUCCESS;
}

static AcResult
acDeviceCommunicateBlocksMPI(const Device device,        //
                             const int3* a0s,            // Src idx inside comp. domain
                             const int3* b0s,            // Dst idx inside bound zone
                             const size_t mapping_count, // Num a0s and b0s
                             const int3 dims)            // Block size
{
    cudaSetDevice(device->id);
    acDeviceSynchronizeStream(device, STREAM_ALL);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Datatype datatype = MPI_FLOAT;
    if (sizeof(AcReal) == 8)
        datatype = MPI_DOUBLE;

    int nprocs, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    const int3 decomp = decompose(nprocs);

    const int3 nn = (int3){
        device->local_config.int_params[AC_nx],
        device->local_config.int_params[AC_ny],
        device->local_config.int_params[AC_nz],
    };

    for (int k = -1; k <= 1; ++k) {
        for (int j = -1; j <= 1; ++j) {
            for (int i = -1; i <= 1; ++i) {
                if (i == 0 && j == 0 && k == 0)
                    continue;

                for (size_t a_idx = 0; a_idx < mapping_count; ++a_idx) {
                    for (size_t b_idx = 0; b_idx < mapping_count; ++b_idx) {
                        const int3 neighbor = (int3){i, j, k};

                        const int3 a0 = a0s[a_idx];
                        // const int3 a1 = a0 + dims;

                        const int3 b0 = a0 - neighbor * nn;
                        // const int3 b1 = a1 - neighbor * nn;

                        if (b0s[b_idx].x == b0.x && b0s[b_idx].y == b0.y && b0s[b_idx].z == b0.z) {

                            const size_t count = dims.x * dims.y * dims.z * NUM_VTXBUF_HANDLES;

                            PackedData src = acCreatePackedData(dims);
                            PackedData dst = acCreatePackedData(dims);

                            const cudaStream_t stream = device->streams[STREAM_DEFAULT];
                            acKernelPackData(stream, device->vba, a0, src);
                            acDeviceSynchronizeStream(device, STREAM_DEFAULT);

                            // Host ////////////////////////////////////////////////
                            PackedData src_host = acCreatePackedDataHost(dims);
                            PackedData dst_host = acCreatePackedDataHost(dims);
                            acTransferPackedDataToHost(device, device->streams[STREAM_DEFAULT], src,
                                                       &src_host);
                            acDeviceSynchronizeStream(device, STREAM_ALL);
                            MPI_Barrier(MPI_COMM_WORLD);
                            ////////////////////////////////////////////////////////

                            const int3 pid3d = getPid3D(pid, decomp);
                            MPI_Request send_req, recv_req;
                            MPI_Isend(src_host.data, count, datatype,
                                      getPid(pid3d + neighbor, decomp), b_idx, MPI_COMM_WORLD,
                                      &send_req);
                            MPI_Irecv(dst_host.data, count, datatype,
                                      getPid(pid3d - neighbor, decomp), b_idx, MPI_COMM_WORLD,
                                      &recv_req);

                            MPI_Wait(&send_req, MPI_STATUS_IGNORE);
                            MPI_Wait(&recv_req, MPI_STATUS_IGNORE);

                            // Host ////////////////////////////////////////////////
                            acTransferPackedDataToDevice(device, device->streams[STREAM_DEFAULT],
                                                         dst_host, &dst);
                            acDeviceSynchronizeStream(device, STREAM_ALL);
                            acDestroyPackedDataHost(&src_host);
                            acDestroyPackedDataHost(&dst_host);
                            ////////////////////////////////////////////////////////

                            acKernelUnpackData(stream, dst, b0, device->vba);
                            acDeviceSynchronizeStream(device, STREAM_DEFAULT);

                            acDestroyPackedData(&src);
                            acDestroyPackedData(&dst);
                        }
                    }
                }
            }
        }
    }

    return AC_SUCCESS;
}

typedef struct {
    PackedData* srcs;
    PackedData* dsts;
    PackedData* srcs_host;
    PackedData* dsts_host;
    int3 dims;
    size_t count;

    cudaStream_t* streams;
    MPI_Request* send_reqs;
    MPI_Request* recv_reqs;
} CommData;

static CommData
acCreateCommData(const Device device, const int3 dims, const size_t count)
{
    cudaSetDevice(device->id);

    CommData data = {};

    data.srcs      = (PackedData*)malloc(count * sizeof(PackedData));
    data.dsts      = (PackedData*)malloc(count * sizeof(PackedData));
    data.srcs_host = (PackedData*)malloc(count * sizeof(PackedData));
    data.dsts_host = (PackedData*)malloc(count * sizeof(PackedData));
    data.dims      = dims;
    data.count     = count;

    data.streams   = (cudaStream_t*)malloc(count * sizeof(cudaStream_t));
    data.send_reqs = (MPI_Request*)malloc(count * sizeof(MPI_Request));
    data.recv_reqs = (MPI_Request*)malloc(count * sizeof(MPI_Request));

    ERRCHK_ALWAYS(data.srcs);
    ERRCHK_ALWAYS(data.dsts);
    ERRCHK_ALWAYS(data.srcs_host);
    ERRCHK_ALWAYS(data.dsts_host);
    ERRCHK_ALWAYS(data.send_reqs);
    ERRCHK_ALWAYS(data.recv_reqs);

    for (size_t i = 0; i < count; ++i) {
        data.srcs[i]      = acCreatePackedData(dims);
        data.dsts[i]      = acCreatePackedData(dims);
        data.srcs_host[i] = acCreatePackedDataHost(dims);
        data.dsts_host[i] = acCreatePackedDataHost(dims);

        cudaStreamCreate(&data.streams[i]);
    }

    return data;
}

static void
acDestroyCommData(const Device device, CommData* data)
{
    cudaSetDevice(device->id);

    for (size_t i = 0; i < data->count; ++i) {
        acDestroyPackedData(&data->srcs[i]);
        acDestroyPackedData(&data->dsts[i]);
        acDestroyPackedDataHost(&data->srcs_host[i]);
        acDestroyPackedDataHost(&data->dsts_host[i]);

        cudaStreamDestroy(data->streams[i]);
    }

    free(data->srcs);
    free(data->dsts);
    free(data->srcs_host);
    free(data->dsts_host);

    free(data->streams);
    free(data->send_reqs);
    free(data->recv_reqs);

    data->count = -1;
    data->dims  = (int3){-1, -1, -1};
}

static void
acPackCommData(const Device device, const int3* a0s, CommData* data)
{
    cudaSetDevice(device->id);
    for (size_t i = 0; i < data->count; ++i)
        acKernelPackData(data->streams[i], device->vba, a0s[i], data->srcs[i]);
}

static void
acTransferCommDataToHost(const Device device, CommData* data)
{
    cudaSetDevice(device->id);
    for (size_t i = 0; i < data->count; ++i)
        acTransferPackedDataToHost(device, data->streams[i], data->srcs[i], &data->srcs_host[i]);
}

static void
acUnpackCommData(const Device device, const int3* b0s, CommData* data)
{
    cudaSetDevice(device->id);

    for (size_t i = 0; i < data->count; ++i)
        acKernelUnpackData(data->streams[i], data->dsts[i], b0s[i], device->vba);
}

static void
acTransferCommDataToDevice(const Device device, CommData* data)
{
    cudaSetDevice(device->id);
    for (size_t i = 0; i < data->count; ++i)
        acTransferPackedDataToDevice(device, data->streams[i], data->dsts_host[i], &data->dsts[i]);
}

static AcResult
acTransferCommData(const Device device, //
                   const int3* a0s,     // Src idx inside comp. domain
                   const int3* b0s,     // Dst idx inside bound zone
                   CommData* data)
{
    cudaSetDevice(device->id);

    MPI_Datatype datatype = MPI_FLOAT;
    if (sizeof(AcReal) == 8)
        datatype = MPI_DOUBLE;

    int nprocs, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    const int3 decomp = decompose(nprocs);

    const int3 nn = (int3){
        device->local_config.int_params[AC_nx],
        device->local_config.int_params[AC_ny],
        device->local_config.int_params[AC_nz],
    };

    const int3 dims         = data->dims;
    const size_t blockcount = data->count;

    for (int k = -1; k <= 1; ++k) {
        for (int j = -1; j <= 1; ++j) {
            for (int i = -1; i <= 1; ++i) {
                if (i == 0 && j == 0 && k == 0)
                    continue;

                for (size_t a_idx = 0; a_idx < blockcount; ++a_idx) {
                    for (size_t b_idx = 0; b_idx < blockcount; ++b_idx) {
                        const int3 neighbor = (int3){i, j, k};

                        const int3 a0 = a0s[a_idx];
                        // const int3 a1 = a0 + dims;

                        const int3 b0 = a0 - neighbor * nn;
                        // const int3 b1 = a1 - neighbor * nn;

                        if (b0s[b_idx].x == b0.x && b0s[b_idx].y == b0.y && b0s[b_idx].z == b0.z) {

                            const size_t count = dims.x * dims.y * dims.z * NUM_VTXBUF_HANDLES;

                            // PackedData src = data->srcs[a_idx];
                            // PackedData dst = data->dsts[b_idx];
                            // PackedData src = data->srcs_host[a_idx];
                            PackedData dst = data->dsts_host[b_idx];

                            const int3 pid3d = getPid3D(pid, decomp);
                            MPI_Irecv(dst.data, count, datatype, getPid(pid3d - neighbor, decomp),
                                      b_idx, MPI_COMM_WORLD, &data->recv_reqs[b_idx]);
                        }
                    }
                }
            }
        }
    }

    for (int k = -1; k <= 1; ++k) {
        for (int j = -1; j <= 1; ++j) {
            for (int i = -1; i <= 1; ++i) {
                if (i == 0 && j == 0 && k == 0)
                    continue;

                for (size_t a_idx = 0; a_idx < blockcount; ++a_idx) {
                    for (size_t b_idx = 0; b_idx < blockcount; ++b_idx) {
                        const int3 neighbor = (int3){i, j, k};

                        const int3 a0 = a0s[a_idx];
                        // const int3 a1 = a0 + dims;

                        const int3 b0 = a0 - neighbor * nn;
                        // const int3 b1 = a1 - neighbor * nn;

                        if (b0s[b_idx].x == b0.x && b0s[b_idx].y == b0.y && b0s[b_idx].z == b0.z) {

                            const size_t count = dims.x * dims.y * dims.z * NUM_VTXBUF_HANDLES;

                            // PackedData src = data->srcs[a_idx];
                            // PackedData dst = data->dsts[b_idx];
                            PackedData src = data->srcs_host[a_idx];
                            // PackedData dst = data->dsts_host[b_idx];

                            const int3 pid3d = getPid3D(pid, decomp);

                            cudaStreamSynchronize(data->streams[a_idx]);
                            MPI_Isend(src.data, count, datatype, getPid(pid3d + neighbor, decomp),
                                      b_idx, MPI_COMM_WORLD, &data->send_reqs[b_idx]);
                        }
                    }
                }
            }
        }
    }

    return AC_SUCCESS;
}

static void
acTransferCommDataWait(const CommData data)
{
    for (size_t i = 0; i < data.count; ++i) {
        MPI_Wait(&data.send_reqs[i], MPI_STATUS_IGNORE);
        MPI_Wait(&data.recv_reqs[i], MPI_STATUS_IGNORE);
    }
}

static AcResult
acDeviceCommunicateHalosMPI(const Device device)
{
    // Configure
    const int3 nn = (int3){
        device->local_config.int_params[AC_nx],
        device->local_config.int_params[AC_ny],
        device->local_config.int_params[AC_nz],
    };

    // Corners
    const int3 corner_a0s[] = {
        (int3){NGHOST, NGHOST, NGHOST}, //
        (int3){nn.x, NGHOST, NGHOST},   //
        (int3){NGHOST, nn.y, NGHOST},   //
        (int3){nn.x, nn.y, NGHOST},     //

        (int3){NGHOST, NGHOST, nn.z}, //
        (int3){nn.x, NGHOST, nn.z},   //
        (int3){NGHOST, nn.y, nn.z},   //
        (int3){nn.x, nn.y, nn.z},
    };
    const int3 corner_b0s[] = {
        (int3){0, 0, 0},
        (int3){NGHOST + nn.x, 0, 0},
        (int3){0, NGHOST + nn.y, 0},
        (int3){NGHOST + nn.x, NGHOST + nn.y, 0},

        (int3){0, 0, NGHOST + nn.z},
        (int3){NGHOST + nn.x, 0, NGHOST + nn.z},
        (int3){0, NGHOST + nn.y, NGHOST + nn.z},
        (int3){NGHOST + nn.x, NGHOST + nn.y, NGHOST + nn.z},
    };
    const int3 corner_dims = (int3){NGHOST, NGHOST, NGHOST};

    // Edges X
    const int3 edgex_a0s[] = {
        (int3){NGHOST, NGHOST, NGHOST}, //
        (int3){NGHOST, nn.y, NGHOST},   //

        (int3){NGHOST, NGHOST, nn.z}, //
        (int3){NGHOST, nn.y, nn.z},   //
    };
    const int3 edgex_b0s[] = {
        (int3){NGHOST, 0, 0},
        (int3){NGHOST, NGHOST + nn.y, 0},

        (int3){NGHOST, 0, NGHOST + nn.z},
        (int3){NGHOST, NGHOST + nn.y, NGHOST + nn.z},
    };
    const int3 edgex_dims = (int3){nn.x, NGHOST, NGHOST};

    // Edges Y
    const int3 edgey_a0s[] = {
        (int3){NGHOST, NGHOST, NGHOST}, //
        (int3){nn.x, NGHOST, NGHOST},   //

        (int3){NGHOST, NGHOST, nn.z}, //
        (int3){nn.x, NGHOST, nn.z},   //
    };
    const int3 edgey_b0s[] = {
        (int3){0, NGHOST, 0},
        (int3){NGHOST + nn.x, NGHOST, 0},

        (int3){0, NGHOST, NGHOST + nn.z},
        (int3){NGHOST + nn.x, NGHOST, NGHOST + nn.z},
    };
    const int3 edgey_dims = (int3){NGHOST, nn.y, NGHOST};

    // Edges Z
    const int3 edgez_a0s[] = {
        (int3){NGHOST, NGHOST, NGHOST}, //
        (int3){nn.x, NGHOST, NGHOST},   //

        (int3){NGHOST, nn.y, NGHOST}, //
        (int3){nn.x, nn.y, NGHOST},   //
    };
    const int3 edgez_b0s[] = {
        (int3){0, 0, NGHOST},
        (int3){NGHOST + nn.x, 0, NGHOST},

        (int3){0, NGHOST + nn.y, NGHOST},
        (int3){NGHOST + nn.x, NGHOST + nn.y, NGHOST},
    };

    const int3 edgez_dims = (int3){NGHOST, NGHOST, nn.z};

    // Sides XY
    const int3 sidexy_a0s[] = {
        (int3){NGHOST, NGHOST, NGHOST}, //
        (int3){NGHOST, NGHOST, nn.z},   //
    };
    const int3 sidexy_b0s[] = {
        (int3){NGHOST, NGHOST, 0},             //
        (int3){NGHOST, NGHOST, NGHOST + nn.z}, //
    };
    const int3 sidexy_dims = (int3){nn.x, nn.y, NGHOST};

    // Sides XZ
    const int3 sidexz_a0s[] = {
        (int3){NGHOST, NGHOST, NGHOST}, //
        (int3){NGHOST, nn.y, NGHOST},   //
    };
    const int3 sidexz_b0s[] = {
        (int3){NGHOST, 0, NGHOST},             //
        (int3){NGHOST, NGHOST + nn.y, NGHOST}, //
    };
    const int3 sidexz_dims = (int3){nn.x, NGHOST, nn.z};

    // Sides YZ
    const int3 sideyz_a0s[] = {
        (int3){NGHOST, NGHOST, NGHOST}, //
        (int3){nn.x, NGHOST, NGHOST},   //
    };
    const int3 sideyz_b0s[] = {
        (int3){0, NGHOST, NGHOST},             //
        (int3){NGHOST + nn.x, NGHOST, NGHOST}, //
    };
    const int3 sideyz_dims = (int3){NGHOST, nn.y, nn.z};

    // Alloc
    CommData corner_data = acCreateCommData(device, corner_dims, ARRAY_SIZE(corner_a0s));
    CommData edgex_data  = acCreateCommData(device, edgex_dims, ARRAY_SIZE(edgex_a0s));
    CommData edgey_data  = acCreateCommData(device, edgey_dims, ARRAY_SIZE(edgey_a0s));
    CommData edgez_data  = acCreateCommData(device, edgez_dims, ARRAY_SIZE(edgez_a0s));
    CommData sidexy_data = acCreateCommData(device, sidexy_dims, ARRAY_SIZE(sidexy_a0s));
    CommData sidexz_data = acCreateCommData(device, sidexz_dims, ARRAY_SIZE(sidexz_a0s));
    CommData sideyz_data = acCreateCommData(device, sideyz_dims, ARRAY_SIZE(sideyz_a0s));

    // Warmup
    for (int i = 0; i < 10; ++i) {
        acPackCommData(device, corner_a0s, &corner_data);
        acPackCommData(device, edgex_a0s, &edgex_data);
        acPackCommData(device, edgey_a0s, &edgey_data);
        acPackCommData(device, edgez_a0s, &edgez_data);
        acPackCommData(device, sidexy_a0s, &sidexy_data);
        acPackCommData(device, sidexz_a0s, &sidexz_data);
        acPackCommData(device, sideyz_a0s, &sideyz_data);

        acTransferCommDataToHost(device, &corner_data);
        acTransferCommDataToHost(device, &edgex_data);
        acTransferCommDataToHost(device, &edgey_data);
        acTransferCommDataToHost(device, &edgez_data);
        acTransferCommDataToHost(device, &sidexy_data);
        acTransferCommDataToHost(device, &sidexz_data);
        acTransferCommDataToHost(device, &sideyz_data);

        acTransferCommData(device, corner_a0s, corner_b0s, &corner_data);
        acTransferCommData(device, edgex_a0s, edgex_b0s, &edgex_data);
        acTransferCommData(device, edgey_a0s, edgey_b0s, &edgey_data);
        acTransferCommData(device, edgez_a0s, edgez_b0s, &edgez_data);
        acTransferCommData(device, sidexy_a0s, sidexy_b0s, &sidexy_data);
        acTransferCommData(device, sidexz_a0s, sidexz_b0s, &sidexz_data);
        acTransferCommData(device, sideyz_a0s, sideyz_b0s, &sideyz_data);

        acTransferCommDataWait(corner_data);
        acTransferCommDataWait(edgex_data);
        acTransferCommDataWait(edgey_data);
        acTransferCommDataWait(edgez_data);
        acTransferCommDataWait(sidexy_data);
        acTransferCommDataWait(sidexz_data);
        acTransferCommDataWait(sideyz_data);

        acTransferCommDataToDevice(device, &corner_data);
        acTransferCommDataToDevice(device, &edgex_data);
        acTransferCommDataToDevice(device, &edgey_data);
        acTransferCommDataToDevice(device, &edgez_data);
        acTransferCommDataToDevice(device, &sidexy_data);
        acTransferCommDataToDevice(device, &sidexz_data);
        acTransferCommDataToDevice(device, &sideyz_data);

        acUnpackCommData(device, corner_b0s, &corner_data);
        acUnpackCommData(device, edgex_b0s, &edgex_data);
        acUnpackCommData(device, edgey_b0s, &edgey_data);
        acUnpackCommData(device, edgez_b0s, &edgez_data);
        acUnpackCommData(device, sidexy_b0s, &sidexy_data);
        acUnpackCommData(device, sidexz_b0s, &sidexz_data);
        acUnpackCommData(device, sideyz_b0s, &sideyz_data);
    }

    // Communicate
    Timer ttot;
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    timer_reset(&ttot);

    acPackCommData(device, corner_a0s, &corner_data);
    acPackCommData(device, edgex_a0s, &edgex_data);
    acPackCommData(device, edgey_a0s, &edgey_data);
    acPackCommData(device, edgez_a0s, &edgez_data);
    acPackCommData(device, sidexy_a0s, &sidexy_data);
    acPackCommData(device, sidexz_a0s, &sidexz_data);
    acPackCommData(device, sideyz_a0s, &sideyz_data);

    acTransferCommDataToHost(device, &corner_data);
    acTransferCommDataToHost(device, &edgex_data);
    acTransferCommDataToHost(device, &edgey_data);
    acTransferCommDataToHost(device, &edgez_data);
    acTransferCommDataToHost(device, &sidexy_data);
    acTransferCommDataToHost(device, &sidexz_data);
    acTransferCommDataToHost(device, &sideyz_data);

    acTransferCommData(device, corner_a0s, corner_b0s, &corner_data);
    acTransferCommData(device, edgex_a0s, edgex_b0s, &edgex_data);
    acTransferCommData(device, edgey_a0s, edgey_b0s, &edgey_data);
    acTransferCommData(device, edgez_a0s, edgez_b0s, &edgez_data);
    acTransferCommData(device, sidexy_a0s, sidexy_b0s, &sidexy_data);
    acTransferCommData(device, sidexz_a0s, sidexz_b0s, &sidexz_data);
    acTransferCommData(device, sideyz_a0s, sideyz_b0s, &sideyz_data);

    acTransferCommDataWait(corner_data);
    acTransferCommDataWait(edgex_data);
    acTransferCommDataWait(edgey_data);
    acTransferCommDataWait(edgez_data);
    acTransferCommDataWait(sidexy_data);
    acTransferCommDataWait(sidexz_data);
    acTransferCommDataWait(sideyz_data);

    acTransferCommDataToDevice(device, &corner_data);
    acTransferCommDataToDevice(device, &edgex_data);
    acTransferCommDataToDevice(device, &edgey_data);
    acTransferCommDataToDevice(device, &edgez_data);
    acTransferCommDataToDevice(device, &sidexy_data);
    acTransferCommDataToDevice(device, &sidexz_data);
    acTransferCommDataToDevice(device, &sideyz_data);

    acUnpackCommData(device, corner_b0s, &corner_data);
    acUnpackCommData(device, edgex_b0s, &edgex_data);
    acUnpackCommData(device, edgey_b0s, &edgey_data);
    acUnpackCommData(device, edgez_b0s, &edgez_data);
    acUnpackCommData(device, sidexy_b0s, &sidexy_data);
    acUnpackCommData(device, sidexz_b0s, &sidexz_data);
    acUnpackCommData(device, sideyz_b0s, &sideyz_data);

    /*
    acPackCommData(device, corner_a0s, &corner_data);
    acPackCommData(device, edgex_a0s, &edgex_data);
    acPackCommData(device, edgey_a0s, &edgey_data);
    acPackCommData(device, edgez_a0s, &edgez_data);
    acPackCommData(device, sidexy_a0s, &sidexy_data);
    acPackCommData(device, sidexz_a0s, &sidexz_data);
    acPackCommData(device, sideyz_a0s, &sideyz_data);

    acTransferCommDataToHost(device, &corner_data);
    acTransferCommDataToHost(device, &edgex_data);
    acTransferCommDataToHost(device, &edgey_data);
    acTransferCommDataToHost(device, &edgez_data);
    acTransferCommDataToHost(device, &sidexy_data);
    acTransferCommDataToHost(device, &sidexz_data);
    acTransferCommDataToHost(device, &sideyz_data);

    acTransferCommData(device, corner_a0s, corner_b0s, &corner_data);
    acTransferCommData(device, edgex_a0s, edgex_b0s, &edgex_data);
    acTransferCommData(device, edgey_a0s, edgey_b0s, &edgey_data);
    acTransferCommData(device, edgez_a0s, edgez_b0s, &edgez_data);
    acTransferCommData(device, sidexy_a0s, sidexy_b0s, &sidexy_data);
    acTransferCommData(device, sidexz_a0s, sidexz_b0s, &sidexz_data);
    acTransferCommData(device, sideyz_a0s, sideyz_b0s, &sideyz_data);

    acTransferCommDataWait(corner_data);
    acTransferCommDataWait(edgex_data);
    acTransferCommDataWait(edgey_data);
    acTransferCommDataWait(edgez_data);
    acTransferCommDataWait(sidexy_data);
    acTransferCommDataWait(sidexz_data);
    acTransferCommDataWait(sideyz_data);

    acTransferCommDataToDevice(device, &corner_data);
    acTransferCommDataToDevice(device, &edgex_data);
    acTransferCommDataToDevice(device, &edgey_data);
    acTransferCommDataToDevice(device, &edgez_data);
    acTransferCommDataToDevice(device, &sidexy_data);
    acTransferCommDataToDevice(device, &sidexz_data);
    acTransferCommDataToDevice(device, &sideyz_data);

    acUnpackCommData(device, corner_b0s, &corner_data);
    acUnpackCommData(device, edgex_b0s, &edgex_data);
    acUnpackCommData(device, edgey_b0s, &edgey_data);
    acUnpackCommData(device, edgez_b0s, &edgez_data);
    acUnpackCommData(device, sidexy_b0s, &sidexy_data);
    acUnpackCommData(device, sidexz_b0s, &sidexz_data);
    acUnpackCommData(device, sideyz_b0s, &sideyz_data);
    */

    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    int pid;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    if (!pid) {
        printf("---------------------------Total: ");
        timer_diff_print(ttot);
    }

    // Dealloc
    acDestroyCommData(device, &corner_data);
    acDestroyCommData(device, &edgex_data);
    acDestroyCommData(device, &edgey_data);
    acDestroyCommData(device, &edgez_data);
    acDestroyCommData(device, &sidexy_data);
    acDestroyCommData(device, &sidexz_data);
    acDestroyCommData(device, &sideyz_data);

    return AC_SUCCESS;
}

AcResult
acDeviceRunMPITest(void)
{
    MPI_Init(NULL, NULL);

    int nprocs, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    printf("Processor %s. Process %d of %d.\n", processor_name, pid, nprocs);

    // Create model and candidate meshes
    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);
    // Some real params must be calculated (for the MHD case) // TODO DANGEROUS
    info.real_params[AC_inv_dsx]   = AcReal(1.0) / info.real_params[AC_dsx];
    info.real_params[AC_inv_dsy]   = AcReal(1.0) / info.real_params[AC_dsy];
    info.real_params[AC_inv_dsz]   = AcReal(1.0) / info.real_params[AC_dsz];
    info.real_params[AC_cs2_sound] = info.real_params[AC_cs_sound] * info.real_params[AC_cs_sound];

    AcMesh model, candidate;

    // Master CPU
    if (pid == 0) {
        acMeshCreate(info, &model);
        acMeshCreate(info, &candidate);

        acMeshRandomize(&model);
        acMeshRandomize(&candidate);
    }

    /// DECOMPOSITION & SUBMESH ///////////////////////////////////
    AcMeshInfo submesh_info  = info;
    const int3 decomposition = decompose(nprocs);
    const int3 pid3d         = getPid3D(pid, decomposition);

    printf("Decomposition: %d, %d, %d\n", decomposition.x, decomposition.y, decomposition.z);
    printf("Process %d: (%d, %d, %d)\n", pid, pid3d.x, pid3d.y, pid3d.z);
    ERRCHK_ALWAYS(info.int_params[AC_nx] % decomposition.x == 0);
    ERRCHK_ALWAYS(info.int_params[AC_ny] % decomposition.y == 0);
    ERRCHK_ALWAYS(info.int_params[AC_nz] % decomposition.z == 0);

    submesh_info.int_params[AC_nx]             = info.int_params[AC_nx] / decomposition.x;
    submesh_info.int_params[AC_ny]             = info.int_params[AC_ny] / decomposition.y;
    submesh_info.int_params[AC_nz]             = info.int_params[AC_nz] / decomposition.z;
    submesh_info.int3_params[AC_global_grid_n] = (int3){
        info.int_params[AC_nx],
        info.int_params[AC_ny],
        info.int_params[AC_nz],
    };
    submesh_info.int3_params[AC_multigpu_offset] = (int3){-1, -1, -1}; // TODO
    WARNING("AC_multigpu_offset not yet implemented");
    acUpdateConfig(&submesh_info);
    ERRCHK_ALWAYS(is_valid(submesh_info.real_params[AC_inv_dsx]));
    ERRCHK_ALWAYS(is_valid(submesh_info.real_params[AC_cs2_sound]));

    AcMesh submesh;
    acMeshCreate(submesh_info, &submesh);
    acMeshRandomize(&submesh);
    ////////////////////////////////////////////////////////////////

    // GPU INIT ////////////////////////////////////////////////////
    int devices_per_node = -1;
    cudaGetDeviceCount(&devices_per_node);

    Device device;
    acDeviceCreate(pid % devices_per_node, submesh_info, &device);
    // TODO enable peer access
    ////////////////////////////////////////////////////////////////

    // DISTRIBUTE & LOAD //////////////////////////////////////////
    acDeviceDistributeMeshMPI(model, decomposition, &submesh);
    acDeviceLoadMesh(device, STREAM_DEFAULT, submesh);
    ///////////////////////////////////////////////////////////////

    // SYNC //////////////////////////////////////////////////////
    acDeviceSynchronizeStream(device, STREAM_ALL);
    MPI_Barrier(MPI_COMM_WORLD);
    //////////////////////////////////////////////////////////////

    // TIMING START //////////////////////////////////////////////
    acDeviceSynchronizeStream(device, STREAM_ALL);
    MPI_Barrier(MPI_COMM_WORLD);
    Timer t;
    timer_reset(&t);
    //////////////////////////////////////////////////////////////

    // INTEGRATION & BOUNDCONDS////////////////////////////////////
    acDeviceCommunicateHalosMPI(device);
    ///////////////////////////////////////////////////////////////

    // TIMING END //////////////////////////////////////////////
    acDeviceSynchronizeStream(device, STREAM_ALL);
    MPI_Barrier(MPI_COMM_WORLD);
    if (!pid) {
        timer_diff_print(t);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    //////////////////////////////////////////////////////////////

    // STORE & GATHER /////////////////////////////////////////////
    MPI_Barrier(MPI_COMM_WORLD);
    acDeviceStoreMesh(device, STREAM_DEFAULT, &submesh);
    acDeviceSynchronizeStream(device, STREAM_DEFAULT);
    acDeviceGatherMeshMPI(submesh, decomposition, &candidate);
    //////////////////////////////////////////////////////////////

    // VERIFY ////////////////////////////////////////////////////
    if (pid == 0) {
        acMeshApplyPeriodicBounds(&model);

        acVerifyMesh(model, candidate);
        acMeshDestroy(&model);
        acMeshDestroy(&candidate);
    }
    //////////////////////////////////////////////////////////////

    // DESTROY ///////////////////////////////////////////////////
    acDeviceDestroy(device);
    acMeshDestroy(&submesh);
    MPI_Finalize();
    //////////////////////////////////////////////////////////////
    return AC_SUCCESS;
}
#endif // MPI_ENABLED //////////////////////////////////////////////////////////////////////////////
