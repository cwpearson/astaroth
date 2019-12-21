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
#include "astaroth_device.h"

#include <string.h> // memcpy

#include "errchk.h"
#include "math_utils.h"

#include "kernels/common.cuh"

struct device_s {
    int id;
    AcMeshInfo local_config;

    // Concurrency
    cudaStream_t streams[NUM_STREAMS];

    // Memory
    VertexBufferArray vba;
    AcReal* reduce_scratchpad;
    AcReal* reduce_result;

    /*
    #if AC_MPI_ENABLED
        // Declare memory for buffers needed for packed data transfers here
        AcReal* inner[2];
        AcReal* outer[2];

        AcReal* inner_host[2];
        AcReal* outer_host[2];
    #endif
    */
};

#include "kernels/boundconds.cuh"
#include "kernels/integration.cuh"
#include "kernels/reductions.cuh"

#if PACKED_DATA_TRANSFERS // Defined in device.cuh
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

    /*
    #if AC_MPI_ENABLED
        // Allocate data required for packed transfers here (cudaMalloc)
        const size_t block_size_bytes = device_config.int_params[AC_mx] *
                                        device_config.int_params[AC_my] * NGHOST *
    NUM_VTXBUF_HANDLES * sizeof(AcReal); for (int i = 0; i < 2; ++i) {
            ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&device->inner[i], block_size_bytes));
            ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&device->outer[i], block_size_bytes));

            ERRCHK_CUDA_ALWAYS(cudaMallocHost(&device->inner_host[i], block_size_bytes));
            ERRCHK_CUDA_ALWAYS(cudaMallocHost(&device->outer_host[i], block_size_bytes));
        }
    #endif
    */

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

    /*
    #if AC_MPI_ENABLED
        // Free data required for packed tranfers here (cudaFree)
        for (int i = 0; i < 2; ++i) {
            cudaFree(device->inner[i]);
            cudaFree(device->outer[i]);

            cudaFreeHost(device->inner_host[i]);
            cudaFreeHost(device->outer_host[i]);
        }
    #endif
    */

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

////////////////////////////////////////////////////////////////////////////////////////////////////
// MPI tests
////////////////////////////////////////////////////////////////////////////////////////////////////
#if AC_MPI_ENABLED == 1
/**
    Running: mpirun -np <num processes> <executable>
*/
#include <mpi.h>

#include <assert.h>

static int
mod(const int a, const int b)
{
    const int r = a % b;
    return r < 0 ? r + b : r;
}

/*
static inline int
get_neighbor(const int3 offset)
{
    // The number of nodes is n^3 = m = num_processes
    // Require that the problem size is always equivalent among processes ((floor(cbrt(m))^3 == m)
    // Require that mesh dimension is (n 2^w), where w is some integer

    int pid, num_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    const int n = (int)cbrt((float)num_processes);
    ERRCHK_ALWAYS((int)ceil(cbrt((float)num_processes)) == n);
    ERRCHK_ALWAYS(n * n * n == num_processes);

    return mod(pid + offset.x, n) + offset.y * n + offset.z * n * n;
}
*/

/*
#ifdef DECOMP_1D
// Working 1D decomp distribute/gather
static void
acDeviceDistributeMeshMPI(const AcMesh src, AcMesh* dst)
{
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Distributing mesh...\n");

    MPI_Datatype datatype = MPI_FLOAT;
    if (sizeof(AcReal) == 8)
        datatype = MPI_DOUBLE;

    int pid, num_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    const size_t count = acVertexBufferSize(dst->info);
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {

        if (pid == 0) {
            // Communicate to self
            assert(dst);
            memcpy(&dst->vertex_buffer[i][0], //
                   &src.vertex_buffer[i][0],  //
                   count * sizeof(src.vertex_buffer[i][0]));

            // Communicate to others
            for (int j = 1; j < num_processes; ++j) {
                const size_t src_idx = acVertexBufferIdx(
                    0, 0, j * src.info.int_params[AC_nz] / num_processes, src.info);

                MPI_Send(&src.vertex_buffer[i][src_idx], count, datatype, j, 0, MPI_COMM_WORLD);
            }
        }
        else {
            assert(dst);

            // Recv
            const size_t dst_idx = 0;
            MPI_Status status;
            MPI_Recv(&dst->vertex_buffer[i][dst_idx], count, datatype, 0, 0, MPI_COMM_WORLD,
                     &status);
        }
    }
}

static void
acDeviceGatherMeshMPI(const AcMesh src, AcMesh* dst)
{
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Gathering mesh...\n");
    MPI_Datatype datatype = MPI_FLOAT;
    if (sizeof(AcReal) == 8)
        datatype = MPI_DOUBLE;

    int pid, num_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    size_t count = acVertexBufferSize(src.info);

    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        // Communicate to self
        if (pid == 0) {
            assert(dst);
            memcpy(&dst->vertex_buffer[i][0], //
                   &src.vertex_buffer[i][0],  //
                   count * sizeof(src.vertex_buffer[i][0]));

            for (int j = 1; j < num_processes; ++j) {
                // Recv
                const size_t dst_idx = acVertexBufferIdx(
                    0, 0, j * dst->info.int_params[AC_nz] / num_processes, dst->info);

                assert(dst_idx + count <= acVertexBufferSize(dst->info));
                MPI_Status status;
                MPI_Recv(&dst->vertex_buffer[i][dst_idx], count, datatype, j, 0, MPI_COMM_WORLD,
                         &status);
            }
        }
        else {
            // Send
            const size_t src_idx = 0;

            assert(src_idx + count <= acVertexBufferSize(src.info));
            MPI_Send(&src.vertex_buffer[i][src_idx], count, datatype, 0, 0, MPI_COMM_WORLD);
        }
    }
}
#endif
*/

// 1D decomp
static AcResult
acDeviceBoundStepMPI(const Device device)
{
    const int mx       = device->local_config.int_params[AC_mx];
    const int my       = device->local_config.int_params[AC_my];
    const int mz       = device->local_config.int_params[AC_mz];
    const size_t count = mx * my * NGHOST;

    acDeviceSynchronizeStream(device, STREAM_ALL);
    // Local boundconds
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        // Front plate local
        {
            const int3 start = (int3){0, 0, NGHOST};
            const int3 end   = (int3){mx, my, 2 * NGHOST};
            acDevicePeriodicBoundcondStep(device, (Stream)i, (VertexBufferHandle)i, start, end);
        }
        // Back plate local
        {
            const int3 start = (int3){0, 0, mz - 2 * NGHOST};
            const int3 end   = (int3){mx, my, mz - NGHOST};
            acDevicePeriodicBoundcondStep(device, (Stream)i, (VertexBufferHandle)i, start, end);
        }
    }
#define INNER_BOUNDCOND_STREAM ((Stream)(NUM_STREAMS - 1))
    // Inner boundconds (while waiting)
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {

        const int3 start = (int3){0, 0, 2 * NGHOST};
        const int3 end   = (int3){mx, my, mz - 2 * NGHOST};
        acDevicePeriodicBoundcondStep(device, INNER_BOUNDCOND_STREAM, (VertexBufferHandle)i, start,
                                      end);
    }

    // MPI
    MPI_Request send_requests[2 * NUM_VTXBUF_HANDLES];
    MPI_Request recv_requests[2 * NUM_VTXBUF_HANDLES];
    MPI_Datatype datatype = MPI_FLOAT;
    if (sizeof(AcReal) == 8)
        datatype = MPI_DOUBLE;

    int pid, num_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        { // Recv neighbor's front
            // ...|ooooxxx|... -> xxx|ooooooo|...
            const size_t dst_idx = acVertexBufferIdx(0, 0, 0, device->local_config);
            const int recv_pid   = (pid + num_processes - 1) % num_processes;

            MPI_Irecv(&device->vba.in[i][dst_idx], count, datatype, recv_pid, i, MPI_COMM_WORLD,
                      &recv_requests[i]);
        }
        { // Recv neighbor's back
            // ...|ooooooo|xxx <- ...|xxxoooo|...
            const size_t dst_idx = acVertexBufferIdx(0, 0, mz - NGHOST, device->local_config);
            const int recv_pid   = (pid + 1) % num_processes;

            MPI_Irecv(&device->vba.in[i][dst_idx], count, datatype, recv_pid,
                      NUM_VTXBUF_HANDLES + i, MPI_COMM_WORLD,
                      &recv_requests[i + NUM_VTXBUF_HANDLES]);
        }
    }

    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        acDeviceSynchronizeStream(device, (Stream)i);
        {
            // Send front
            // ...|ooooxxx|... -> xxx|ooooooo|...
            const size_t src_idx = acVertexBufferIdx(0, 0, mz - 2 * NGHOST, device->local_config);
            const int send_pid   = (pid + 1) % num_processes;

            MPI_Isend(&device->vba.in[i][src_idx], count, datatype, send_pid, i, MPI_COMM_WORLD,
                      &send_requests[i]);
        }
        { // Send back
            // ...|ooooooo|xxx <- ...|xxxoooo|...
            const size_t src_idx = acVertexBufferIdx(0, 0, NGHOST, device->local_config);
            const int send_pid   = (pid + num_processes - 1) % num_processes;

            MPI_Isend(&device->vba.in[i][src_idx], count, datatype, send_pid,
                      i + NUM_VTXBUF_HANDLES, MPI_COMM_WORLD,
                      &send_requests[i + NUM_VTXBUF_HANDLES]);
        }
    }
    MPI_Waitall(2 * NUM_VTXBUF_HANDLES, recv_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(2 * NUM_VTXBUF_HANDLES, send_requests, MPI_STATUSES_IGNORE);
    return AC_SUCCESS;
}

// 1D decomp
static AcResult
acDeviceIntegrateStepMPI(const Device device, const AcReal dt)
{
    const int mx       = device->local_config.int_params[AC_mx];
    const int my       = device->local_config.int_params[AC_my];
    const int mz       = device->local_config.int_params[AC_mz];
    const int nx       = device->local_config.int_params[AC_nx];
    const int ny       = device->local_config.int_params[AC_ny];
    const int nz       = device->local_config.int_params[AC_nz];
    const size_t count = mx * my * NGHOST;

    for (int isubstep = 0; isubstep < 3; ++isubstep) {
        acDeviceSynchronizeStream(device, STREAM_ALL);
        // Local boundconds
        for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
            // Front plate local
            {
                const int3 start = (int3){0, 0, NGHOST};
                const int3 end   = (int3){mx, my, 2 * NGHOST};
                acDevicePeriodicBoundcondStep(device, (Stream)i, (VertexBufferHandle)i, start, end);
            }
            // Back plate local
            {
                const int3 start = (int3){0, 0, mz - 2 * NGHOST};
                const int3 end   = (int3){mx, my, mz - NGHOST};
                acDevicePeriodicBoundcondStep(device, (Stream)i, (VertexBufferHandle)i, start, end);
            }
        }
#define INNER_BOUNDCOND_STREAM ((Stream)(NUM_STREAMS - 1))
        // Inner boundconds (while waiting)
        for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {

            const int3 start = (int3){0, 0, 2 * NGHOST};
            const int3 end   = (int3){mx, my, mz - 2 * NGHOST};
            acDevicePeriodicBoundcondStep(device, INNER_BOUNDCOND_STREAM, (VertexBufferHandle)i,
                                          start, end);
        }

        // MPI
        MPI_Request send_requests[2 * NUM_VTXBUF_HANDLES];
        MPI_Request recv_requests[2 * NUM_VTXBUF_HANDLES];
        MPI_Datatype datatype = MPI_FLOAT;
        if (sizeof(AcReal) == 8)
            datatype = MPI_DOUBLE;

        int pid, num_processes;
        MPI_Comm_rank(MPI_COMM_WORLD, &pid);
        MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

        for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
            { // Recv neighbor's front
                // ...|ooooxxx|... -> xxx|ooooooo|...
                const size_t dst_idx = acVertexBufferIdx(0, 0, 0, device->local_config);
                const int recv_pid   = (pid + num_processes - 1) % num_processes;

                MPI_Irecv(&device->vba.in[i][dst_idx], count, datatype, recv_pid, i, MPI_COMM_WORLD,
                          &recv_requests[i]);
            }
            { // Recv neighbor's back
                // ...|ooooooo|xxx <- ...|xxxoooo|...
                const size_t dst_idx = acVertexBufferIdx(0, 0, mz - NGHOST, device->local_config);
                const int recv_pid   = (pid + 1) % num_processes;

                MPI_Irecv(&device->vba.in[i][dst_idx], count, datatype, recv_pid,
                          NUM_VTXBUF_HANDLES + i, MPI_COMM_WORLD,
                          &recv_requests[i + NUM_VTXBUF_HANDLES]);
            }
        }

        for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
            acDeviceSynchronizeStream(device, (Stream)i);
            {
                // Send front
                // ...|ooooxxx|... -> xxx|ooooooo|...
                const size_t src_idx = acVertexBufferIdx(0, 0, mz - 2 * NGHOST,
                                                         device->local_config);
                const int send_pid   = (pid + 1) % num_processes;

                MPI_Isend(&device->vba.in[i][src_idx], count, datatype, send_pid, i, MPI_COMM_WORLD,
                          &send_requests[i]);
            }
            { // Send back
                // ...|ooooooo|xxx <- ...|xxxoooo|...
                const size_t src_idx = acVertexBufferIdx(0, 0, NGHOST, device->local_config);
                const int send_pid   = (pid + num_processes - 1) % num_processes;

                MPI_Isend(&device->vba.in[i][src_idx], count, datatype, send_pid,
                          i + NUM_VTXBUF_HANDLES, MPI_COMM_WORLD,
                          &send_requests[i + NUM_VTXBUF_HANDLES]);
            }
        }
        // Inner integration
        {
            ERRCHK(NUM_STREAMS - 2 >= 0);
            const int3 m1 = (int3){2 * NGHOST, 2 * NGHOST, 2 * NGHOST};
            const int3 m2 = (int3){mx, my, mz} - m1;
            acDeviceIntegrateSubstep(device, (Stream)(NUM_STREAMS - 2), isubstep, m1, m2, dt);
        }

        MPI_Waitall(2 * NUM_VTXBUF_HANDLES, recv_requests, MPI_STATUSES_IGNORE);
        MPI_Waitall(2 * NUM_VTXBUF_HANDLES, send_requests, MPI_STATUSES_IGNORE);

        acDeviceSynchronizeStream(device, INNER_BOUNDCOND_STREAM);
        { // Front
            const int3 m1 = (int3){NGHOST, NGHOST, NGHOST};
            const int3 m2 = m1 + (int3){nx, ny, NGHOST};
            acDeviceIntegrateSubstep(device, STREAM_0, isubstep, m1, m2, dt);
        }
        { // Back
            const int3 m1 = (int3){NGHOST, NGHOST, nz};
            const int3 m2 = m1 + (int3){nx, ny, NGHOST};
            acDeviceIntegrateSubstep(device, STREAM_1, isubstep, m1, m2, dt);
        }
        { // Bottom
            const int3 m1 = (int3){NGHOST, NGHOST, 2 * NGHOST};
            const int3 m2 = m1 + (int3){nx, NGHOST, nz - 2 * NGHOST};
            acDeviceIntegrateSubstep(device, STREAM_2, isubstep, m1, m2, dt);
        }
        { // Top
            const int3 m1 = (int3){NGHOST, ny, 2 * NGHOST};
            const int3 m2 = m1 + (int3){nx, NGHOST, nz - 2 * NGHOST};
            acDeviceIntegrateSubstep(device, STREAM_3, isubstep, m1, m2, dt);
        }
        { // Left
            const int3 m1 = (int3){NGHOST, 2 * NGHOST, 2 * NGHOST};
            const int3 m2 = m1 + (int3){NGHOST, ny - 2 * NGHOST, nz - 2 * NGHOST};
            acDeviceIntegrateSubstep(device, STREAM_4, isubstep, m1, m2, dt);
        }
        { // Right
            const int3 m1 = (int3){nx, 2 * NGHOST, 2 * NGHOST};
            const int3 m2 = m1 + (int3){NGHOST, ny - 2 * NGHOST, nz - 2 * NGHOST};
            acDeviceIntegrateSubstep(device, STREAM_5, isubstep, m1, m2, dt);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        acDeviceSwapBuffers(device);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    return AC_SUCCESS;
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

static int
getPid1D(const int3 pid, const int3 decomposition)
{
    return mod(pid.x, decomposition.x) +                   //
           mod(pid.y, decomposition.y) * decomposition.x + //
           mod(pid.z, decomposition.z) * decomposition.x * decomposition.y;
}

static void
acDeviceDistributeMeshMPI(const AcMesh src, const int3 decomposition, AcMesh* dst)
{
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Distributing mesh...\n");

    MPI_Datatype datatype = MPI_FLOAT;
    if (sizeof(AcReal) == 8)
        datatype = MPI_DOUBLE;

    int pid, num_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    const size_t count = acVertexBufferSize(dst->info);
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {

        if (pid == 0) {
            // Communicate to self
            assert(dst);
            memcpy(&dst->vertex_buffer[i][0], //
                   &src.vertex_buffer[i][0],  //
                   count * sizeof(src.vertex_buffer[i][0]));

            // Communicate to others
            for (int j = 1; j < num_processes; ++j) {
                const int3 target_pid = getPid3D(j, decomposition);
                const int3 offset     = (int3){
                    src.info.int_params[AC_nx] / decomposition.x,
                    src.info.int_params[AC_ny] / decomposition.y,
                    src.info.int_params[AC_nz] / decomposition.z,
                };
                const size_t src_idx = acVertexBufferIdx(target_pid.x * offset.x,
                                                         target_pid.y * offset.y,
                                                         target_pid.z * offset.z, src.info);

                MPI_Send(&src.vertex_buffer[i][src_idx], count, datatype, j, 0, MPI_COMM_WORLD);
            }
        }
        else {
            assert(dst);

            // Recv
            const size_t dst_idx = 0;
            MPI_Status status;
            MPI_Recv(&dst->vertex_buffer[i][dst_idx], count, datatype, 0, 0, MPI_COMM_WORLD,
                     &status);
        }
    }
}

static void
acDeviceGatherMeshMPI(const AcMesh src, const int3 decomposition, AcMesh* dst)
{
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Gathering mesh...\n");
    MPI_Datatype datatype = MPI_FLOAT;
    if (sizeof(AcReal) == 8)
        datatype = MPI_DOUBLE;

    int pid, num_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    size_t count = acVertexBufferSize(src.info);

    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        // Communicate to self
        if (pid == 0) {
            assert(dst);
            memcpy(&dst->vertex_buffer[i][0], //
                   &src.vertex_buffer[i][0],  //
                   count * sizeof(src.vertex_buffer[i][0]));

            for (int j = 1; j < num_processes; ++j) {
                // Recv
                // NOTE! Gathers halos also! Must be up to date!
                // Not well-defined which GPU it gets the boundary from!
                const int3 target_pid = getPid3D(j, decomposition);
                const int3 offset     = (int3){
                    dst->info.int_params[AC_nx] / decomposition.x,
                    dst->info.int_params[AC_ny] / decomposition.y,
                    dst->info.int_params[AC_nz] / decomposition.z,
                };
                const size_t dst_idx = acVertexBufferIdx(target_pid.x * offset.x,
                                                         target_pid.y * offset.y,
                                                         target_pid.z * offset.z, dst->info);

                assert(dst_idx + count <= acVertexBufferSize(dst->info));
                MPI_Status status;
                MPI_Recv(&dst->vertex_buffer[i][dst_idx], count, datatype, j, 0, MPI_COMM_WORLD,
                         &status);
            }
        }
        else {
            // Send
            const size_t src_idx = 0;

            assert(src_idx + count <= acVertexBufferSize(src.info));
            MPI_Send(&src.vertex_buffer[i][src_idx], count, datatype, 0, 0, MPI_COMM_WORLD);
        }
    }
}

static void
acDeviceCommunicateHalos(void)
{
    // TODO
    WARNING("acDeviceCommunicateHalos not yet implemented. Tests will fail (bounds must be "
            "up-to-date before calling acDeviceGatherMeshMPI)");
}

// From Astaroth Utils
#include "src/utils/config_loader.h"
#include "src/utils/memory.h"
#include "src/utils/modelsolver.h"
#include "src/utils/timer_hires.h"
#include "src/utils/verification.h"

#include <algorithm>
#include <vector>
// --smpiargs="-gpu"

// 3D decomposition
AcResult
acDeviceRunMPITest(void)
{
    // If 1, runs the strong scaling tests and verification.
    // Verification is disabled when benchmarking weak scaling because the
    // whole grid is too large to fit into memory.
#define BENCH_STRONG_SCALING (1)

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

    // Large mesh dim
    const int nn           = 64;
    info.int_params[AC_nx] = info.int_params[AC_ny] = nn;
    info.int_params[AC_nz]                          = BENCH_STRONG_SCALING ? nn : nn * nprocs;
    info.real_params[AC_inv_dsx]                    = AcReal(1.0) / info.real_params[AC_dsx];
    info.real_params[AC_inv_dsy]                    = AcReal(1.0) / info.real_params[AC_dsy];
    info.real_params[AC_inv_dsz]                    = AcReal(1.0) / info.real_params[AC_dsz];
    info.real_params[AC_cs2_sound] = info.real_params[AC_cs_sound] * info.real_params[AC_cs_sound];
    acUpdateConfig(&info);
    acPrintMeshInfo(info);

#if BENCH_STRONG_SCALING
    AcMesh model, candidate;

    // Master CPU
    if (pid == 0) {
        acMeshCreate(info, &model);
        acMeshCreate(info, &candidate);

        acMeshRandomize(&model);
        acMeshRandomize(&candidate);
    }
#endif

    /// DECOMPOSITION
    AcMeshInfo submesh_info  = info;
    const int3 decomposition = decompose(nprocs);
    const int3 pid3d         = (int3){
        pid % decomposition.x,
        (pid / decomposition.x) % decomposition.y,
        (pid / (decomposition.x * decomposition.y)),
    };

    printf("Decomposition: %d, %d, %d\n", decomposition.x, decomposition.y, decomposition.z);
    printf("Process %d: (%d, %d, %d)\n", pid, pid3d.x, pid3d.y, pid3d.z);
    ERRCHK(info.int_params[AC_nx] % decomposition.x == 0);
    ERRCHK(info.int_params[AC_ny] % decomposition.y == 0);
    ERRCHK(info.int_params[AC_nz] % decomposition.z == 0);

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
    //

    AcMesh submesh;
    acMeshCreate(submesh_info, &submesh);
    acMeshRandomize(&submesh);

#if BENCH_STRONG_SCALING
    acDeviceDistributeMeshMPI(model, decomposition, &submesh);
#endif

    // Init the GPU
    int devices_per_node = -1;
    cudaGetDeviceCount(&devices_per_node);

    Device device;
    acDeviceCreate(pid % devices_per_node, submesh_info, &device);
    acDeviceLoadMesh(device, STREAM_DEFAULT, submesh);

    MPI_Barrier(MPI_COMM_WORLD);
    // Attempt to enable peer access with all neighbors in the node
    for (int i = 0; i < devices_per_node; ++i) {
        cudaSetDevice(device->id);
        WARNCHK_CUDA_ALWAYS(cudaDeviceEnablePeerAccess(i, 0));
    }
    /*
    // Attempt to enable peer access to the most expensive neighbors (sides)
    const int left  = getPid1D(pid3d + (int3){-1, 0, 0}, decomposition);
    const int right = getPid1D(pid3d + (int3){1, 0, 0}, decomposition);
    const int up    = getPid1D(pid3d + (int3){0, 1, 0}, decomposition);
    const int down  = getPid1D(pid3d + (int3){0, -1, 0}, decomposition);
    const int front = getPid1D(pid3d + (int3){0, 0, 1}, decomposition);
    const int back  = getPid1D(pid3d + (int3){0, 0, -1}, decomposition);

    cudaSetDevice(device->id);
    WARNCHK_CUDA_ALWAYS(cudaDeviceEnablePeerAccess(left, 0));
    WARNCHK_CUDA_ALWAYS(cudaDeviceEnablePeerAccess(right, 0));
    WARNCHK_CUDA_ALWAYS(cudaDeviceEnablePeerAccess(up, 0));
    WARNCHK_CUDA_ALWAYS(cudaDeviceEnablePeerAccess(down, 0));
    WARNCHK_CUDA_ALWAYS(cudaDeviceEnablePeerAccess(front, 0));
    WARNCHK_CUDA_ALWAYS(cudaDeviceEnablePeerAccess(back, 0));
    */

// Verification start ///////////////////////////////////////////////////////////////////////
#if BENCH_STRONG_SCALING
    {
        // const float dt = FLT_EPSILON; // TODO
        // acDeviceIntegrateStepMPI(device, dt); // TODO
        // acDeviceBoundStepMPI(device); TODO
        acDeviceCommunicateHalos();
        acDeviceSynchronizeStream(device, STREAM_ALL);

        acDeviceStoreMesh(device, STREAM_DEFAULT, &submesh);
        acDeviceGatherMeshMPI(submesh, decomposition, &candidate);
        if (pid == 0) {
            // acModelIntegrateStep(model, FLT_EPSILON); // TODO
            acMeshApplyPeriodicBounds(&model);

            const bool valid = acVerifyMesh(model, candidate);
            acMeshDestroy(&model);
            acMeshDestroy(&candidate);

            // Write out
            char buf[256];
            sprintf(buf, "nprocs_%d_result_%s", nprocs, valid ? "valid" : "INVALID_CHECK_OUTPUT");
            FILE* fp = fopen(buf, "w");
            ERRCHK_ALWAYS(fp);
            fclose(fp);
        }
    }
#endif
    // Verification end ///////////////////////////////////////////////////////////////////////

    /*
    // TODO
    // Warmup
    for (int i = 0; i < 10; ++i)
        acDeviceIntegrateStepMPI(device, 0);

    acDeviceSynchronizeStream(device, STREAM_ALL);
    MPI_Barrier(MPI_COMM_WORLD);

    // Benchmark start ///////////////////////////////////////////////////////////////////////
    std::vector<double> results;
    results.reserve(num_iters);

    Timer total_time;
    timer_reset(&total_time);

    Timer step_time;
    const int num_iters    = 10;
    for (int i = 0; i < num_iters; ++i) {
        timer_reset(&step_time);

        const AcReal dt = FLT_EPSILON;
        acDeviceIntegrateStepMPI(device, dt);
        acDeviceSynchronizeStream(device, STREAM_ALL);
        MPI_Barrier(MPI_COMM_WORLD);

        results.push_back(timer_diff_nsec(step_time) / 1e6);
    }

    const double ms_elapsed     = timer_diff_nsec(total_time) / 1e6;
    const double nth_percentile = 0.90;
    std::sort(results.begin(), results.end(),
              [](const double& a, const double& b) { return a < b; });

    if (pid == 0) {
        printf("vertices: %d^3, iterations: %d\n", nn, num_iters);
        printf("Total time: %f ms\n", ms_elapsed);
        printf("Time per step: %f ms\n", ms_elapsed / num_iters);

        const size_t nth_index = int(nth_percentile * num_iters);
        printf("%dth percentile per step: %f ms\n", int(100 * nth_percentile), results[nth_index]);

        // Write out
        char buf[256];
        sprintf(buf, "nprocs_%d_result_%s.bench", nprocs, BENCH_STRONG_SCALING ? "strong" : "weak");
        FILE* fp = fopen(buf, "w");
        ERRCHK_ALWAYS(fp);
        fprintf(fp, "nprocs, percentile (%dth)\n", int(100 * nth_percentile));
        fprintf(fp, "%d, %g\n", nprocs, results[nth_index]);
        fclose(fp);
    }
    // Benchmark end ///////////////////////////////////////////////////////////////////////
    */

    // Finalize
    acDeviceDestroy(device);
    acMeshDestroy(&submesh);
    MPI_Finalize();
    return AC_SUCCESS;
}

/*
// Working, 1D decomposition
AcResult
acDeviceRunMPITest(void)
{
    // If 1, runs the strong scaling tests and verification.
    // Verification is disabled when benchmarking weak scaling because the
    // whole grid is too large to fit into memory.
#define BENCH_STRONG_SCALING (0)

    MPI_Init(NULL, NULL);

    int num_processes, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    printf("Processor %s. Process %d of %d.\n", processor_name, pid, num_processes);

    // Create model and candidate meshes
    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);

    // Large mesh dim
    const int nn           = 256;
    const int num_iters    = 10;
    info.int_params[AC_nx] = info.int_params[AC_ny] = nn;
    info.int_params[AC_nz]         = BENCH_STRONG_SCALING ? nn : nn * num_processes;
    info.real_params[AC_inv_dsx]   = AcReal(1.0) / info.real_params[AC_dsx];
    info.real_params[AC_inv_dsy]   = AcReal(1.0) / info.real_params[AC_dsy];
    info.real_params[AC_inv_dsz]   = AcReal(1.0) / info.real_params[AC_dsz];
    info.real_params[AC_cs2_sound] = info.real_params[AC_cs_sound] * info.real_params[AC_cs_sound];
    acUpdateConfig(&info);
    acPrintMeshInfo(info);
    ERRCHK_ALWAYS(info.int_params[AC_nz] % num_processes == 0);

#if BENCH_STRONG_SCALING
    AcMesh model, candidate;

    // Master CPU
    if (pid == 0) {
        acMeshCreate(info, &model);
        acMeshCreate(info, &candidate);

        acMeshRandomize(&model);
        acMeshRandomize(&candidate);
    }
#endif

    /// DECOMPOSITION
    AcMeshInfo submesh_info                    = info;
    const int submesh_nz                       = info.int_params[AC_nz] / num_processes;
    submesh_info.int_params[AC_nz]             = submesh_nz;
    submesh_info.int3_params[AC_global_grid_n] = (int3){
        info.int_params[AC_nx],
        info.int_params[AC_ny],
        info.int_params[AC_nz],
    };
    submesh_info.int3_params[AC_multigpu_offset] = (int3){0, 0, pid * submesh_nz};
    acUpdateConfig(&submesh_info);
    ERRCHK_ALWAYS(is_valid(submesh_info.real_params[AC_inv_dsx]));
    ERRCHK_ALWAYS(is_valid(submesh_info.real_params[AC_cs2_sound]));
    //

    AcMesh submesh;
    acMeshCreate(submesh_info, &submesh);
    acMeshRandomize(&submesh);

#if BENCH_STRONG_SCALING
    acDeviceDistributeMeshMPI(model, &submesh);
#endif

    // Init the GPU
    int devices_per_node = -1;
    cudaGetDeviceCount(&devices_per_node);

    Device device;
    acDeviceCreate(pid % devices_per_node, submesh_info, &device);
    acDeviceLoadMesh(device, STREAM_DEFAULT, submesh);

    // Enable peer access
    MPI_Barrier(MPI_COMM_WORLD);
    const int front = (device->id + 1) % devices_per_node;
    const int back  = (device->id + devices_per_node - 1) % devices_per_node;
    cudaSetDevice(device->id);
    WARNCHK_CUDA_ALWAYS(cudaDeviceEnablePeerAccess(front, 0));
    WARNCHK_CUDA_ALWAYS(cudaDeviceEnablePeerAccess(back, 0));

// Verification start ///////////////////////////////////////////////////////////////////////
#if BENCH_STRONG_SCALING
    {
        const float dt = FLT_EPSILON;
        acDeviceIntegrateStepMPI(device, dt);
        acDeviceBoundStepMPI(device);
        acDeviceSynchronizeStream(device, STREAM_ALL);

        acDeviceStoreMesh(device, STREAM_DEFAULT, &submesh);
        acDeviceGatherMeshMPI(submesh, &candidate);
        if (pid == 0) {
            acModelIntegrateStep(model, FLT_EPSILON);
            acMeshApplyPeriodicBounds(&model);

            const bool valid = acVerifyMesh(model, candidate);
            acMeshDestroy(&model);
            acMeshDestroy(&candidate);

            // Write out
            char buf[256];
            sprintf(buf, "nprocs_%d_result_%s", num_processes,
                    valid ? "valid" : "INVALID_CHECK_OUTPUT");
            FILE* fp = fopen(buf, "w");
            ERRCHK_ALWAYS(fp);
            fclose(fp);
        }
    }
#endif
    // Verification end ///////////////////////////////////////////////////////////////////////

    // Warmup
    for (int i = 0; i < 10; ++i)
        acDeviceIntegrateStepMPI(device, 0);

    acDeviceSynchronizeStream(device, STREAM_ALL);
    MPI_Barrier(MPI_COMM_WORLD);

    // Benchmark start ///////////////////////////////////////////////////////////////////////
    std::vector<double> results;
    results.reserve(num_iters);

    Timer total_time;
    timer_reset(&total_time);

    Timer step_time;
    for (int i = 0; i < num_iters; ++i) {
        timer_reset(&step_time);

        const AcReal dt = FLT_EPSILON;
        acDeviceIntegrateStepMPI(device, dt);
        acDeviceSynchronizeStream(device, STREAM_ALL);
        MPI_Barrier(MPI_COMM_WORLD);

        results.push_back(timer_diff_nsec(step_time) / 1e6);
    }

    const double ms_elapsed     = timer_diff_nsec(total_time) / 1e6;
    const double nth_percentile = 0.90;
    std::sort(results.begin(), results.end(),
              [](const double& a, const double& b) { return a < b; });

    if (pid == 0) {
        printf("vertices: %d^3, iterations: %d\n", nn, num_iters);
        printf("Total time: %f ms\n", ms_elapsed);
        printf("Time per step: %f ms\n", ms_elapsed / num_iters);

        const size_t nth_index = int(nth_percentile * num_iters);
        printf("%dth percentile per step: %f ms\n", int(100 * nth_percentile), results[nth_index]);

        // Write out
        char buf[256];
        sprintf(buf, "nprocs_%d_result_%s.bench", num_processes,
                BENCH_STRONG_SCALING ? "strong" : "weak");
        FILE* fp = fopen(buf, "w");
        ERRCHK_ALWAYS(fp);
        fprintf(fp, "num_processes, percentile (%dth)\n", int(100 * nth_percentile));
        fprintf(fp, "%d, %g\n", num_processes, results[nth_index]);
        fclose(fp);
    }
    // Benchmark end ///////////////////////////////////////////////////////////////////////

    // Finalize
    acDeviceDestroy(device);
    acMeshDestroy(&submesh);
    MPI_Finalize();
    return AC_SUCCESS;
}
*/

#else
AcResult
acDeviceRunMPITest(void)
{
    WARNING("MPI was not enabled but acDeviceRunMPITest() was called");
    return AC_FAILURE;
}
#endif

#if PACKED_DATA_TRANSFERS // DEPRECATED, see AC_MPI_ENABLED instead
// Functions for calling packed data transfers
#endif
