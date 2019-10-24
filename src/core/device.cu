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

#include "errchk.h"

// Device info
#define REGISTERS_PER_THREAD (255)
#define MAX_REGISTERS_PER_BLOCK (65536)
#define MAX_THREADS_PER_BLOCK (1024)
#define WARP_SIZE (32)

typedef struct {
    AcReal* in[NUM_VTXBUF_HANDLES];
    AcReal* out[NUM_VTXBUF_HANDLES];

    AcReal* profiles[NUM_SCALARARRAY_HANDLES];
} VertexBufferArray;

struct device_s {
    int id;
    AcMeshInfo local_config;

    // Concurrency
    cudaStream_t streams[NUM_STREAMS];

    // Memory
    VertexBufferArray vba;
    AcReal* reduce_scratchpad;
    AcReal* reduce_result;

#if AC_MPI_ENABLED
    // Declare memory for buffers needed for packed data transfers here
    AcReal* inner[2];
    AcReal* outer[2];

    AcReal* inner_host[2];
    AcReal* outer_host[2];
#endif
};

__constant__ AcMeshInfo d_mesh_info;
static int __device__ __forceinline__
DCONST(const AcIntParam param)
{
    return d_mesh_info.int_params[param];
}
static int3 __device__ __forceinline__
DCONST(const AcInt3Param param)
{
    return d_mesh_info.int3_params[param];
}
static AcReal __device__ __forceinline__
DCONST(const AcRealParam param)
{
    return d_mesh_info.real_params[param];
}
static AcReal3 __device__ __forceinline__
DCONST(const AcReal3Param param)
{
    return d_mesh_info.real3_params[param];
}
static __device__ constexpr VertexBufferHandle
DCONST(const VertexBufferHandle handle)
{
    return handle;
}
#define DEVICE_VTXBUF_IDX(i, j, k) ((i) + (j)*DCONST(AC_mx) + (k)*DCONST(AC_mxy))
#define DEVICE_1D_COMPDOMAIN_IDX(i, j, k) ((i) + (j)*DCONST(AC_nx) + (k)*DCONST(AC_nxy))
#define globalGridN (d_mesh_info.int3_params[AC_global_grid_n])
//#define globalMeshM // Placeholder
//#define localMeshN // Placeholder
//#define localMeshM // Placeholder
//#define localMeshN_min // Placeholder
//#define globalMeshN_min // Placeholder
#define d_multigpu_offset (d_mesh_info.int3_params[AC_multigpu_offset])
//#define d_multinode_offset (d_mesh_info.int3_params[AC_multinode_offset]) // Placeholder
//#include <thrust/complex.h>
// using namespace thrust;
#include <cuComplex.h>
#if AC_DOUBLE_PRECISION == 1
typedef cuDoubleComplex acComplex;
#define acComplex(x, y) make_cuDoubleComplex(x, y)
#else
typedef cuFloatComplex acComplex;
#define acComplex(x, y) make_cuFloatComplex(x, y)
#endif
static __device__ inline acComplex
exp(const acComplex& val)
{
    return acComplex(exp(val.x) * cos(val.y), exp(val.x) * sin(val.y));
}
static __device__ inline acComplex operator*(const AcReal& a, const acComplex& b)
{
    return (acComplex){a * b.x, a * b.y};
}

static __device__ inline acComplex operator*(const acComplex& b, const AcReal& a)
{
    return (acComplex){a * b.x, a * b.y};
}

static __device__ inline acComplex operator*(const acComplex& a, const acComplex& b)
{
    return (acComplex){a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}
//#include <complex>

#include "kernels/boundconds.cuh"
#include "kernels/integration.cuh"
#include "kernels/reductions.cuh"

static dim3 rk3_tpb(32, 1, 4);

#if PACKED_DATA_TRANSFERS // Defined in device.cuh
// #include "kernels/pack_unpack.cuh"
#endif

static __global__ void
dummy_kernel(void)
{
    DCONST((AcIntParam)0);
    DCONST((AcInt3Param)0);
    DCONST((AcRealParam)0);
    DCONST((AcReal3Param)0);
    acComplex a = exp(AcReal(1) * acComplex(1, 1) * AcReal(1));
    a* a;
}

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
    dummy_kernel<<<1, 1>>>();
    ERRCHK_CUDA_KERNEL_ALWAYS();
    printf("Success!\n");

    // Concurrency
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreateWithPriority(&device->streams[i], cudaStreamNonBlocking, i);
    }

    // Memory
    // VBA in/out
    const size_t vba_size_bytes = acVertexBufferSizeBytes(device_config);
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        ERRCHK_CUDA_ALWAYS(cudaMalloc(&device->vba.in[i], vba_size_bytes));
        ERRCHK_CUDA_ALWAYS(cudaMalloc(&device->vba.out[i], vba_size_bytes));
    }
    // VBA Profiles
    const size_t profile_size_bytes = sizeof(AcReal) * max(device_config.int_params[AC_mx],
                                                           max(device_config.int_params[AC_my],
                                                               device_config.int_params[AC_mz]));
    for (int i = 0; i < NUM_SCALARARRAY_HANDLES; ++i) {
        ERRCHK_CUDA_ALWAYS(cudaMalloc(&device->vba.profiles[i], profile_size_bytes));
    }

    // Reductions
    ERRCHK_CUDA_ALWAYS(
        cudaMalloc(&device->reduce_scratchpad, acVertexBufferCompdomainSizeBytes(device_config)));
    ERRCHK_CUDA_ALWAYS(cudaMalloc(&device->reduce_result, sizeof(AcReal)));

#if AC_MPI_ENABLED
    // Allocate data required for packed transfers here (cudaMalloc)
    const size_t block_size_bytes = device_config.int_params[AC_mx] *
                                    device_config.int_params[AC_my] * NGHOST * NUM_VTXBUF_HANDLES *
                                    sizeof(AcReal);
    for (int i = 0; i < 2; ++i) {
        ERRCHK_CUDA_ALWAYS(cudaMalloc(&device->inner[i], block_size_bytes));
        ERRCHK_CUDA_ALWAYS(cudaMalloc(&device->outer[i], block_size_bytes));

        ERRCHK_CUDA_ALWAYS(cudaMallocHost(&device->inner_host[i], block_size_bytes));
        ERRCHK_CUDA_ALWAYS(cudaMallocHost(&device->outer_host[i], block_size_bytes));
    }
#endif

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

#if AC_MPI_ENABLED
    // Free data required for packed tranfers here (cudaFree)
    for (int i = 0; i < 2; ++i) {
        cudaFree(device->inner[i]);
        cudaFree(device->outer[i]);

        cudaFreeHost(device->inner_host[i]);
        cudaFreeHost(device->outer_host[i]);
    }
#endif

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

                // #ifdef AC_dt
                acDeviceLoadScalarUniform(device, STREAM_DEFAULT, AC_dt, FLT_EPSILON);
                /*#else
                                ERROR("FATAL ERROR: acDeviceAutoOptimize() or
                acDeviceIntegrateSubstep() was " "called, but AC_dt was not defined. Either define
                it or call the generated " "device function acDeviceKernel_<kernel name> which does
                not require the " "timestep to be defined.\n"); #endif*/

                cudaEventRecord(tstart); // ---------------------------------------- Timing start
                for (int i = 0; i < num_iterations; ++i)
                    solve<2><<<bpg, tpb>>>(start, end, device->vba);

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
    const size_t offset = (size_t)&d_mesh_info.real_params[param] - (size_t)&d_mesh_info;
    ERRCHK_CUDA(cudaMemcpyToSymbolAsync(d_mesh_info, &value, sizeof(value), offset,
                                        cudaMemcpyHostToDevice, device->streams[stream]));
    return AC_SUCCESS;
}

AcResult
acDeviceLoadVectorUniform(const Device device, const Stream stream, const AcReal3Param param,
                          const AcReal3 value)
{
    cudaSetDevice(device->id);
    const size_t offset = (size_t)&d_mesh_info.real3_params[param] - (size_t)&d_mesh_info;
    ERRCHK_CUDA(cudaMemcpyToSymbolAsync(d_mesh_info, &value, sizeof(value), offset,
                                        cudaMemcpyHostToDevice, device->streams[stream]));
    return AC_SUCCESS;
}

AcResult
acDeviceLoadIntUniform(const Device device, const Stream stream, const AcIntParam param,
                       const int value)
{
    cudaSetDevice(device->id);
    const size_t offset = (size_t)&d_mesh_info.int_params[param] - (size_t)&d_mesh_info;
    ERRCHK_CUDA(cudaMemcpyToSymbolAsync(d_mesh_info, &value, sizeof(value), offset,
                                        cudaMemcpyHostToDevice, device->streams[stream]));
    return AC_SUCCESS;
}

AcResult
acDeviceLoadInt3Uniform(const Device device, const Stream stream, const AcInt3Param param,
                        const int3 value)
{
    cudaSetDevice(device->id);
    const size_t offset = (size_t)&d_mesh_info.int3_params[param] - (size_t)&d_mesh_info;
    ERRCHK_CUDA(cudaMemcpyToSymbolAsync(d_mesh_info, &value, sizeof(value), offset,
                                        cudaMemcpyHostToDevice, device->streams[stream]));
    return AC_SUCCESS;
}

AcResult
acDeviceLoadScalarArray(const Device device, const Stream stream, const ScalarArrayHandle handle,
                        const size_t start, const AcReal* data, const size_t num)
{
    cudaSetDevice(device->id);

    ERRCHK((int)(start + num) <= max(device->local_config.int_params[AC_mx],
                                     max(device->local_config.int_params[AC_my],
                                         device->local_config.int_params[AC_mz])));

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

    ERRCHK_CUDA_ALWAYS(cudaMemcpyToSymbolAsync(d_mesh_info, &device_config, sizeof(device_config),
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

    const dim3 tpb = rk3_tpb;

    const int3 n = end - start;
    const dim3 bpg((unsigned int)ceil(n.x / AcReal(tpb.x)), //
                   (unsigned int)ceil(n.y / AcReal(tpb.y)), //
                   (unsigned int)ceil(n.z / AcReal(tpb.z)));

    //#ifdef AC_dt
    acDeviceLoadScalarUniform(device, stream, AC_dt, dt);
    /*#else
        (void)dt;
        ERROR("FATAL ERROR: acDeviceAutoOptimize() or acDeviceIntegrateSubstep() was "
              "called, but AC_dt was not defined. Either define it or call the generated "
              "device function acDeviceKernel_<kernel name> which does not require the "
              "timestep to be defined.\n");
    #endif*/
    if (step_number == 0)
        solve<0><<<bpg, tpb, 0, device->streams[stream]>>>(start, end, device->vba);
    else if (step_number == 1)
        solve<1><<<bpg, tpb, 0, device->streams[stream]>>>(start, end, device->vba);
    else
        solve<2><<<bpg, tpb, 0, device->streams[stream]>>>(start, end, device->vba);

    ERRCHK_CUDA_KERNEL();

    return AC_SUCCESS;
}

AcResult
acDevicePeriodicBoundcondStep(const Device device, const Stream stream_type,
                              const VertexBufferHandle vtxbuf_handle, const int3 start,
                              const int3 end)
{
    cudaSetDevice(device->id);
    const cudaStream_t stream = device->streams[stream_type];

    const dim3 tpb(8, 2, 8);
    const dim3 bpg((unsigned int)ceil((end.x - start.x) / (float)tpb.x),
                   (unsigned int)ceil((end.y - start.y) / (float)tpb.y),
                   (unsigned int)ceil((end.z - start.z) / (float)tpb.z));

    kernel_periodic_boundconds<<<bpg, tpb, 0, stream>>>(start, end, device->vba.in[vtxbuf_handle]);
    ERRCHK_CUDA_KERNEL();

    return AC_SUCCESS;
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

    *result = reduce_scal(device->streams[stream], rtype, start, end, device->vba.in[vtxbuf_handle],
                          device->reduce_scratchpad, device->reduce_result);
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

    *result = reduce_vec(device->streams[stream], rtype, start, end, device->vba.in[vtxbuf0],
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

static int
mod(const int a, const int b)
{
    const int r = a % b;
    return r < 0 ? r + b : r;
}

static int
get_neighbor(const int3 offset)
{
    // The number of nodes is n^3 = m = num_processes
    // Require that the problem size is always equivalent among processes ((floor(cbrt(m))^3 == m)
    // Require that mesh dimension is (n 2^w), where w is some integer

    int pid, num_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    const int n = (int)cbrt(num_processes);
    ERRCHK_ALWAYS((int)ceil(cbrt(num_processes)) == n);
    ERRCHK_ALWAYS(n * n * n == num_processes);

    return mod(pid + offset.x, n) + offset.y * n + offset.z * n * n;
}

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

// 1D decomp
static AcResult
acDeviceBoundStepMPI(const Device device)
{
    const int mx       = device->local_config.int_params[AC_mx];
    const int my       = device->local_config.int_params[AC_my];
    const int mz       = device->local_config.int_params[AC_mz];
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
                const size_t dst_idx = acVertexBufferIdx(0, 0, 0, device->local_config);
                const int send_pid   = (pid + 1) % num_processes;

                MPI_Request request;
                MPI_Isend(&device->vba.in[i][src_idx], count, datatype, send_pid, i, MPI_COMM_WORLD,
                          &request);
            }
            { // Send back
                // ...|ooooooo|xxx <- ...|xxxoooo|...
                const size_t src_idx = acVertexBufferIdx(0, 0, NGHOST, device->local_config);
                const int send_pid   = (pid + num_processes - 1) % num_processes;

                MPI_Request request;
                MPI_Isend(&device->vba.in[i][src_idx], count, datatype, send_pid,
                          i + NUM_VTXBUF_HANDLES, MPI_COMM_WORLD, &request);
            }
        }
        for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
            MPI_Status status;
            MPI_Wait(&recv_requests[i], &status);
            MPI_Wait(&recv_requests[i + NUM_VTXBUF_HANDLES], &status);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        acDeviceSwapBuffers(device);
        MPI_Barrier(MPI_COMM_WORLD);
    }

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
                const size_t dst_idx = acVertexBufferIdx(0, 0, 0, device->local_config);
                const int send_pid   = (pid + 1) % num_processes;

                MPI_Request request;
                MPI_Isend(&device->vba.in[i][src_idx], count, datatype, send_pid, i, MPI_COMM_WORLD,
                          &request);
            }
            { // Send back
                // ...|ooooooo|xxx <- ...|xxxoooo|...
                const size_t src_idx = acVertexBufferIdx(0, 0, NGHOST, device->local_config);
                const int send_pid   = (pid + num_processes - 1) % num_processes;

                MPI_Request request;
                MPI_Isend(&device->vba.in[i][src_idx], count, datatype, send_pid,
                          i + NUM_VTXBUF_HANDLES, MPI_COMM_WORLD, &request);
            }
        }
        // Inner integration
        {
            ERRCHK(NUM_STREAMS - 2 >= 0);
            const int3 m1 = (int3){2 * NGHOST, 2 * NGHOST, 2 * NGHOST};
            const int3 m2 = (int3){mx, my, mz} - m1;
            acDeviceIntegrateSubstep(device, (Stream)(NUM_STREAMS - 2), isubstep, m1, m2, dt);
        }

        for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
            MPI_Status status;
            MPI_Wait(&recv_requests[i], &status);
            MPI_Wait(&recv_requests[i + NUM_VTXBUF_HANDLES], &status);
        }

        acDeviceSynchronizeStream(device, INNER_BOUNDCOND_STREAM);
        // #pragma omp parallel for
        { // Front
            const int3 m1 = (int3){NGHOST, NGHOST, NGHOST};
            const int3 m2 = m1 + (int3){nx, ny, NGHOST};
            acDeviceIntegrateSubstep(device, STREAM_0, isubstep, m1, m2, dt);
        }
        // #pragma omp parallel for
        { // Back
            const int3 m1 = (int3){NGHOST, NGHOST, nz};
            const int3 m2 = m1 + (int3){nx, ny, NGHOST};
            acDeviceIntegrateSubstep(device, STREAM_1, isubstep, m1, m2, dt);
        }
        // #pragma omp parallel for
        { // Bottom
            const int3 m1 = (int3){NGHOST, NGHOST, 2 * NGHOST};
            const int3 m2 = m1 + (int3){nx, NGHOST, nz - 2 * NGHOST};
            acDeviceIntegrateSubstep(device, STREAM_2, isubstep, m1, m2, dt);
        }
        // #pragma omp parallel for
        { // Top
            const int3 m1 = (int3){NGHOST, ny, 2 * NGHOST};
            const int3 m2 = m1 + (int3){nx, NGHOST, nz - 2 * NGHOST};
            acDeviceIntegrateSubstep(device, STREAM_3, isubstep, m1, m2, dt);
        }
        // #pragma omp parallel for
        { // Left
            const int3 m1 = (int3){NGHOST, 2 * NGHOST, 2 * NGHOST};
            const int3 m2 = m1 + (int3){NGHOST, ny - 2 * NGHOST, nz - 2 * NGHOST};
            acDeviceIntegrateSubstep(device, STREAM_4, isubstep, m1, m2, dt);
        }
        // #pragma omp parallel for
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

// From Astaroth Utils
#include "src/utils/config_loader.h"
#include "src/utils/memory.h"
#include "src/utils/timer_hires.h"
#include "src/utils/verification.h"
// --smpiargs="-gpu"
AcResult
acDeviceRunMPITest(void)
{
    int num_processes, pid;
    MPI_Init(NULL, NULL);
    // int provided;
    // MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided); // Hybrid MP + MPI
    // ERRCHK_ALWAYS(provided == MPI_THREAD_MULTIPLE);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    printf("Processor %s. Process %d of %d.\n", processor_name, pid, num_processes);

#ifdef MPIX_CUDA_AWARE_SUPPORT
    if (MPIX_Query_cuda_support())
        printf("CUDA-aware MPI supported (MPIX)\n");
    else
        WARNING("CUDA-aware MPI not supported with this MPI library (MPIX)\n");
#else
    printf("MPIX_CUDA_AWARE_SUPPORT was not defined. Do not know whether CUDA-aware MPI is "
           "supported\n");
#endif

    if (getenv("MPICH_RDMA_ENABLED_CUDA") && atoi(getenv("MPICH_RDMA_ENABLED_CUDA")))
        printf("CUDA-aware MPI supported (MPICH)\n");
    else
        WARNING("MPICH not used or this MPI library does not support CUDA-aware MPI\n");

    // Create model and candidate meshes
    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);

    // Large mesh dim
    const int nn           = 512;
    info.int_params[AC_nx] = info.int_params[AC_ny] = info.int_params[AC_nz] = nn;
    acUpdateConfig(&info);

    AcMesh model, candidate;

    // Master CPU
    if (pid == 0) {
        acMeshCreate(info, &model);
        acMeshCreate(info, &candidate);

        acMeshRandomize(&model);
    }
    assert(info.int_params[AC_nz] % num_processes == 0);

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
    //

    AcMesh submesh;
    acMeshCreate(submesh_info, &submesh);
    acMeshRandomize(&submesh);
    acDeviceDistributeMeshMPI(model, &submesh);

#define VERIFY (0)

// Master CPU
#if VERIFY
    if (pid == 0) {
        acMeshApplyPeriodicBounds(&model);
    }
#endif

    ////////////////////////////////////////////////////////////////////////////////////////////////
    Device device;
    acDeviceCreate(0, submesh_info, &device);
    acDeviceLoadMesh(device, STREAM_DEFAULT, submesh);

    // Warmup
    for (int i = 0; i < 5; ++i) {
        acDeviceIntegrateStepMPI(device, FLT_EPSILON);
    }
    acDeviceSynchronizeStream(device, STREAM_ALL);
    MPI_Barrier(MPI_COMM_WORLD);

    // Benchmark
    const int num_iters = 100;
    Timer total_time;
    timer_reset(&total_time);
    for (int i = 0; i < num_iters; ++i) {
        // acDeviceBoundStepMPI(device);
        acDeviceIntegrateStepMPI(device, FLT_EPSILON); // TODO recheck
    }
    acDeviceSynchronizeStream(device, STREAM_ALL);
    MPI_Barrier(MPI_COMM_WORLD);
    if (pid == 0) {
        const double ms_elapsed = timer_diff_nsec(total_time) / 1e6;
        printf("vertices: %d^3, iterations: %d\n", nn, num_iters);
        printf("Total time: %f ms\n", ms_elapsed);
        printf("Time per step: %f ms\n", ms_elapsed / num_iters);

        char buf[256];
        sprintf(buf, "procs_%d.bench", num_processes);
        FILE* fp = fopen(buf, "w");
        ERRCHK_ALWAYS(fp);
        fprintf(fp, "%d, %g", num_processes, ms_elapsed);
        fclose(fp);
    }
    ////////////////////////////// Timer end
    acDeviceBoundStepMPI(device);
    acDeviceStoreMesh(device, STREAM_DEFAULT, &submesh);
    acDeviceDestroy(device);
    ////////////////////////////////////////////////////////////////////////////////////////////////

    acDeviceGatherMeshMPI(submesh, &candidate);
    acMeshDestroy(&submesh);

    // Master CPU
    if (pid == 0) {
#if VERIFY
        acVerifyMesh(model, candidate);
#endif
        acMeshDestroy(&model);
        acMeshDestroy(&candidate);
    }

    MPI_Finalize();
    return AC_FAILURE;
}
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
