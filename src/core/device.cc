#include "astaroth.h"

#include <string.h>

#include "errchk.h"
#include "math_utils.h"
#include "timer_hires.h"

#include "kernels/kernels.h"

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))

#define MPI_GPUDIRECT_DISABLED (0) // Buffer through host memory, deprecated
#define MPI_DECOMPOSITION_AXES (3)
#define MPI_COMPUTE_ENABLED (1)
#define MPI_COMM_ENABLED (1)
#define MPI_INCL_CORNERS (0)
#define MPI_USE_PINNED (0)              // Do inter-node comm with pinned memory
#define MPI_USE_CUDA_DRIVER_PINNING (0) // Pin with cuPointerSetAttribute, otherwise cudaMallocHost

#include <cuda.h> // CUDA driver API (needed if MPI_USE_CUDA_DRIVER_PINNING is set)

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
#if AC_VERBOSE
    acDevicePrintInfo(device);
#endif

// Check that the code was compiled for the proper GPU architecture
#if AC_VERBOSE
    printf("Trying to run a dummy kernel. If this fails, make sure that your\n"
           "device supports the CUDA architecture you are compiling for.\n");
#endif
    printf("Testing CUDA... ");
    fflush(stdout);
    acKernelDummy();
    printf("\x1B[32m%s\x1B[0m\n", "OK!");
    fflush(stdout);

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
    acDeviceLoadDefaultUniforms(device);
    acDeviceLoadMeshInfo(device, device_config);

#if AC_VERBOSE
    printf("Created device %d (%p)\n", device->id, device);
#endif
    *device_handle = device;

    // Autoptimize
    acDeviceAutoOptimize(device);

    return AC_SUCCESS;
}

AcResult
acDeviceDestroy(Device device)
{
    cudaSetDevice(device->id);
#if AC_VERBOSE
    printf("Destroying device %d (%p)\n", device->id, device);
#endif
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

AcResult
acDeviceReduceVecScal(const Device device, const Stream stream, const ReductionType rtype,
                  const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                  const VertexBufferHandle vtxbuf2, const VertexBufferHandle vtxbuf3, AcReal* result)
{
    cudaSetDevice(device->id);

    const int3 start = (int3){device->local_config.int_params[AC_nx_min],
                              device->local_config.int_params[AC_ny_min],
                              device->local_config.int_params[AC_nz_min]};

    const int3 end = (int3){device->local_config.int_params[AC_nx_max],
                            device->local_config.int_params[AC_ny_max],
                            device->local_config.int_params[AC_nz_max]};

    *result = acKernelReduceVecScal(device->streams[stream], rtype, start, end, device->vba.in[vtxbuf0],
                                    device->vba.in[vtxbuf1], device->vba.in[vtxbuf2], device->vba.in[vtxbuf3],
                                    device->reduce_scratchpad, device->reduce_result);
    return AC_SUCCESS;
}


#if AC_MPI_ENABLED
/**
Quick overview of the MPI implementation:

The halo is partitioned into segments. The first coordinate of a segment is b0.
The array containing multiple b0s is called... "b0s".

Each b0 maps to an index in the computational domain of some neighboring process a0.
We have a0 = mod(b0 - nghost, nn) + nghost.
Intuitively, we
  1) Transform b0 into a coordinate system where (0, 0, 0) is the first index in
     the comp domain.
  2) Wrap the transformed b0 around nn (comp domain)
  3) Transform b0 back to a coordinate system where (0, 0, 0) is the first index
     in the ghost zone

struct PackedData is used for packing and unpacking. Holds the actual data in
                  the halo partition
struct CommData holds multiple PackedDatas for sending and receiving halo
                partitions
struct Grid contains information about the local GPU device, decomposition, the
            total mesh dimensions and CommDatas


Basic steps:
  1) Distribute the mesh among ranks
  2) Integrate & communicate
    - start inner integration and at the same time, pack halo data and send it to neighbors
    - once all halo data has been received, unpack and do outer integration
    - sync and start again
  3) Gather the mesh to rank 0 for postprocessing
*/
#include <mpi.h>

#include <stdint.h>

typedef struct {
    uint64_t x, y, z;
} uint3_64;

static uint3_64
operator+(const uint3_64& a, const uint3_64& b)
{
    return (uint3_64){a.x + b.x, a.y + b.y, a.z + b.z};
}

static int3
make_int3(const uint3_64 a)
{
    return (int3){(int)a.x, (int)a.y, (int)a.z};
}

static uint64_t
mod(const int a, const int b)
{
    const int r = a % b;
    return r < 0 ? r + b : r;
}

static uint3_64
morton3D(const uint64_t pid)
{
    uint64_t i, j, k;
    i = j = k = 0;

    if (MPI_DECOMPOSITION_AXES == 3) {
        for (int bit = 0; bit <= 21; ++bit) {
            const uint64_t mask = 0x1l << 3 * bit;
            k |= ((pid & (mask << 0)) >> 2 * bit) >> 0;
            j |= ((pid & (mask << 1)) >> 2 * bit) >> 1;
            i |= ((pid & (mask << 2)) >> 2 * bit) >> 2;
        }
    }
    // Just a quick copy/paste for other decomp dims
    else if (MPI_DECOMPOSITION_AXES == 2) {
        for (int bit = 0; bit <= 21; ++bit) {
            const uint64_t mask = 0x1l << 2 * bit;
            j |= ((pid & (mask << 0)) >> 1 * bit) >> 0;
            k |= ((pid & (mask << 1)) >> 1 * bit) >> 1;
        }
    }
    else if (MPI_DECOMPOSITION_AXES == 1) {
        for (int bit = 0; bit <= 21; ++bit) {
            const uint64_t mask = 0x1l << 1 * bit;
            k |= ((pid & (mask << 0)) >> 0 * bit) >> 0;
        }
    }
    else {
        fprintf(stderr, "Invalid MPI_DECOMPOSITION_AXES\n");
        ERRCHK_ALWAYS(0);
    }

    return (uint3_64){i, j, k};
}

static uint64_t
morton1D(const uint3_64 pid)
{
    uint64_t i = 0;

    if (MPI_DECOMPOSITION_AXES == 3) {
        for (int bit = 0; bit <= 21; ++bit) {
            const uint64_t mask = 0x1l << bit;
            i |= ((pid.z & mask) << 0) << 2 * bit;
            i |= ((pid.y & mask) << 1) << 2 * bit;
            i |= ((pid.x & mask) << 2) << 2 * bit;
        }
    }
    else if (MPI_DECOMPOSITION_AXES == 2) {
        for (int bit = 0; bit <= 21; ++bit) {
            const uint64_t mask = 0x1l << bit;
            i |= ((pid.y & mask) << 0) << 1 * bit;
            i |= ((pid.z & mask) << 1) << 1 * bit;
        }
    }
    else if (MPI_DECOMPOSITION_AXES == 1) {
        for (int bit = 0; bit <= 21; ++bit) {
            const uint64_t mask = 0x1l << bit;
            i |= ((pid.z & mask) << 0) << 0 * bit;
        }
    }
    else {
        fprintf(stderr, "Invalid MPI_DECOMPOSITION_AXES\n");
        ERRCHK_ALWAYS(0);
    }

    return i;
}

static uint3_64
decompose(const uint64_t target)
{
    // This is just so beautifully elegant. Complex and efficient decomposition
    // in just one line of code.
    uint3_64 p = morton3D(target - 1) + (uint3_64){1, 1, 1};

    ERRCHK_ALWAYS(p.x * p.y * p.z == target);
    return p;
}

static uint3_64
wrap(const int3 i, const uint3_64 n)
{
    return (uint3_64){
        mod(i.x, n.x),
        mod(i.y, n.y),
        mod(i.z, n.z),
    };
}

static int
getPid(const int3 pid_raw, const uint3_64 decomp)
{
    const uint3_64 pid = wrap(pid_raw, decomp);
    return (int)morton1D(pid);
}

static int3
getPid3D(const uint64_t pid, const uint3_64 decomp)
{
    const uint3_64 pid3D = morton3D(pid);
    ERRCHK_ALWAYS(getPid(make_int3(pid3D), decomp) == (int)pid);
    return (int3){(int)pid3D.x, (int)pid3D.y, (int)pid3D.z};
}

/** Assumes that contiguous pids are on the same node and there is one process per GPU. */
static inline bool
onTheSameNode(const uint64_t pid_a, const uint64_t pid_b)
{
    int devices_per_node = -1;
    cudaGetDeviceCount(&devices_per_node);

    const uint64_t node_a = pid_a / devices_per_node;
    const uint64_t node_b = pid_b / devices_per_node;

    return node_a == node_b;
}

static PackedData
acCreatePackedData(const int3 dims)
{
    PackedData data = {};

    data.dims = dims;

    const size_t bytes = dims.x * dims.y * dims.z * sizeof(data.data[0]) * NUM_VTXBUF_HANDLES;
    ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&data.data, bytes));

#if MPI_USE_CUDA_DRIVER_PINNING
    ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&data.data_pinned, bytes));

    unsigned int flag = 1;
    CUresult retval   = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                            (CUdeviceptr)data.data_pinned);
    ERRCHK_ALWAYS(retval == CUDA_SUCCESS);
#else
    ERRCHK_CUDA_ALWAYS(cudaMallocHost((void**)&data.data_pinned, bytes));
// ERRCHK_CUDA_ALWAYS(cudaMallocManaged((void**)&data.data_pinned, bytes)); // Significantly
// slower than pinned (38 ms vs. 125 ms)
#endif // USE_CUDA_DRIVER_PINNING

    return data;
}

static AcResult
acDestroyPackedData(PackedData* data)
{
    cudaFree(data->data_pinned);

    data->dims = (int3){-1, -1, -1};
    cudaFree(data->data);
    data->data = NULL;

    return AC_SUCCESS;
}

#if MPI_GPUDIRECT_DISABLED
static PackedData
acCreatePackedDataHost(const int3 dims)
{
    PackedData data = {};

    data.dims = dims;

    const size_t bytes = dims.x * dims.y * dims.z * sizeof(data.data[0]) * NUM_VTXBUF_HANDLES;
    data.data          = (AcRealPacked*)malloc(bytes);
    ERRCHK_ALWAYS(data.data);

    return data;
}

static AcResult
acDestroyPackedDataHost(PackedData* data)
{
    data->dims = (int3){-1, -1, -1};
    free(data->data);
    data->data = NULL;

    return AC_SUCCESS;
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
#endif // MPI_GPUDIRECT_DISABLED

static void
acPinPackedData(const Device device, const cudaStream_t stream, PackedData* ddata)
{
    cudaSetDevice(device->id);
    // TODO sync stream
    ddata->pinned = true;

    const size_t bytes = ddata->dims.x * ddata->dims.y * ddata->dims.z * sizeof(ddata->data[0]) *
                         NUM_VTXBUF_HANDLES;
    ERRCHK_CUDA(cudaMemcpyAsync(ddata->data_pinned, ddata->data, bytes, cudaMemcpyDefault, stream));
}

static void
acUnpinPackedData(const Device device, const cudaStream_t stream, PackedData* ddata)
{
    if (!ddata->pinned) // Unpin iff the data was pinned previously
        return;

    cudaSetDevice(device->id);
    // TODO sync stream
    ddata->pinned = false;

    const size_t bytes = ddata->dims.x * ddata->dims.y * ddata->dims.z * sizeof(ddata->data[0]) *
                         NUM_VTXBUF_HANDLES;
    ERRCHK_CUDA(cudaMemcpyAsync(ddata->data, ddata->data_pinned, bytes, cudaMemcpyDefault, stream));
}

// TODO: do with packed data
static AcResult
acDeviceDistributeMeshMPI(const AcMesh src, const uint3_64 decomposition, AcMesh* dst)
{
    MPI_Barrier(MPI_COMM_WORLD);
#if AC_VERBOSE
    printf("Distributing mesh...\n");
    fflush(stdout);
#endif

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
acDeviceGatherMeshMPI(const AcMesh src, const uint3_64 decomposition, AcMesh* dst)
{
    MPI_Barrier(MPI_COMM_WORLD);
#if AC_VERBOSE
    printf("Gathering mesh...\n");
    fflush(stdout);
#endif

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

typedef struct {
    PackedData* srcs;
    PackedData* dsts;
#if MPI_GPUDIRECT_DISABLED
    PackedData* srcs_host;
    PackedData* dsts_host;
#endif
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

    data.srcs  = (PackedData*)malloc(count * sizeof(PackedData));
    data.dsts  = (PackedData*)malloc(count * sizeof(PackedData));
    data.dims  = dims;
    data.count = count;

    data.streams   = (cudaStream_t*)malloc(count * sizeof(cudaStream_t));
    data.send_reqs = (MPI_Request*)malloc(count * sizeof(MPI_Request));
    data.recv_reqs = (MPI_Request*)malloc(count * sizeof(MPI_Request));

    ERRCHK_ALWAYS(data.srcs);
    ERRCHK_ALWAYS(data.dsts);
    ERRCHK_ALWAYS(data.send_reqs);
    ERRCHK_ALWAYS(data.recv_reqs);

#if MPI_GPUDIRECT_DISABLED
    data.srcs_host = (PackedData*)malloc(count * sizeof(PackedData));
    data.dsts_host = (PackedData*)malloc(count * sizeof(PackedData));
    ERRCHK_ALWAYS(data.srcs_host);
    ERRCHK_ALWAYS(data.dsts_host);
#endif

    for (size_t i = 0; i < count; ++i) {
        data.srcs[i] = acCreatePackedData(dims);
        data.dsts[i] = acCreatePackedData(dims);

#if MPI_GPUDIRECT_DISABLED
        data.srcs_host[i] = acCreatePackedDataHost(dims);
        data.dsts_host[i] = acCreatePackedDataHost(dims);
#endif

        int low_prio, high_prio;
        cudaDeviceGetStreamPriorityRange(&low_prio, &high_prio);
        cudaStreamCreateWithPriority(&data.streams[i], cudaStreamNonBlocking, high_prio);
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

#if MPI_GPUDIRECT_DISABLED
        acDestroyPackedDataHost(&data->srcs_host[i]);
        acDestroyPackedDataHost(&data->dsts_host[i]);
#endif

        cudaStreamDestroy(data->streams[i]);
    }

    free(data->srcs);
    free(data->dsts);

#if MPI_GPUDIRECT_DISABLED
    free(data->srcs_host);
    free(data->dsts_host);
#endif

    free(data->streams);
    free(data->send_reqs);
    free(data->recv_reqs);

    data->count = -1;
    data->dims  = (int3){-1, -1, -1};
}

static void
acSyncCommData(const CommData data)
{
    for (size_t i = 0; i < data.count; ++i)
        cudaStreamSynchronize(data.streams[i]);
}

static int3
mod(const int3 a, const int3 n)
{
    return (int3){(int)mod(a.x, n.x), (int)mod(a.y, n.y), (int)mod(a.z, n.z)};
}

static void
acPackCommData(const Device device, const int3* b0s, CommData* data)
{
    cudaSetDevice(device->id);

    const int3 nn = (int3){
        device->local_config.int_params[AC_nx],
        device->local_config.int_params[AC_ny],
        device->local_config.int_params[AC_nz],
    };
    const int3 nghost = (int3){NGHOST, NGHOST, NGHOST};

    for (size_t i = 0; i < data->count; ++i) {
        const int3 a0 = mod(b0s[i] - nghost, nn) + nghost;
        acKernelPackData(data->streams[i], device->vba, a0, data->srcs[i]);
    }
}

static void
acUnpackCommData(const Device device, const int3* b0s, CommData* data)
{
    cudaSetDevice(device->id);

    for (size_t i = 0; i < data->count; ++i)
        acKernelUnpackData(data->streams[i], data->dsts[i], b0s[i], device->vba);
}

#if MPI_GPUDIRECT_DISABLED
static void
acTransferCommDataToHost(const Device device, CommData* data)
{
    cudaSetDevice(device->id);
    for (size_t i = 0; i < data->count; ++i)
        acTransferPackedDataToHost(device, data->streams[i], data->srcs[i], &data->srcs_host[i]);
}

static void
acTransferCommDataToDevice(const Device device, CommData* data)
{
    cudaSetDevice(device->id);
    for (size_t i = 0; i < data->count; ++i)
        acTransferPackedDataToDevice(device, data->streams[i], data->dsts_host[i], &data->dsts[i]);
}
#endif

static inline void
acPinCommData(const Device device, CommData* data)
{
    cudaSetDevice(device->id);
    for (size_t i = 0; i < data->count; ++i)
        acPinPackedData(device, data->streams[i], &data->srcs[i]);
}

static void
acUnpinCommData(const Device device, CommData* data)
{
    cudaSetDevice(device->id);

    // Clear pin flags from src
    for (size_t i = 0; i < data->count; ++i)
        data->srcs[i].pinned = false;

    // Transfer from pinned to gmem
    for (size_t i = 0; i < data->count; ++i)
        acUnpinPackedData(device, data->streams[i], &data->dsts[i]);
}

static AcResult
acTransferCommData(const Device device, //
                   const int3* b0s,     // Halo partition coordinates
                   CommData* data)
{
    cudaSetDevice(device->id);

    MPI_Datatype datatype = MPI_FLOAT;
    if (sizeof(data->srcs[0].data[0]) == 2) {
        datatype = MPI_SHORT; // TODO CONFIRM THAT IS CORRECTLY CAST TO HALF
    } else if (sizeof(data->srcs[0].data[0]) == 4) {
        datatype = MPI_FLOAT;
    } else {
        datatype = MPI_DOUBLE;
    }

    int nprocs, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    const uint3_64 decomp = decompose(nprocs);

    const int3 nn = (int3){
        device->local_config.int_params[AC_nx],
        device->local_config.int_params[AC_ny],
        device->local_config.int_params[AC_nz],
    };

    const int3 pid3d        = getPid3D(pid, decomp);
    const int3 dims         = data->dims;
    const size_t blockcount = data->count;
    const size_t count      = dims.x * dims.y * dims.z * NUM_VTXBUF_HANDLES;

    for (size_t b0_idx = 0; b0_idx < blockcount; ++b0_idx) {

        const int3 b0       = b0s[b0_idx];
        const int3 neighbor = (int3){
            b0.x < NGHOST ? -1 : b0.x >= NGHOST + nn.x ? 1 : 0,
            b0.y < NGHOST ? -1 : b0.y >= NGHOST + nn.y ? 1 : 0,
            b0.z < NGHOST ? -1 : b0.z >= NGHOST + nn.z ? 1 : 0,
        };
        const int npid = getPid(pid3d + neighbor, decomp);

        PackedData* dst = &data->dsts[b0_idx];
        if (onTheSameNode(pid, npid) || !MPI_USE_PINNED) {
            MPI_Irecv(dst->data, count, datatype, npid, b0_idx, //
                      MPI_COMM_WORLD, &data->recv_reqs[b0_idx]);
            dst->pinned = false;
        }
        else {
            MPI_Irecv(dst->data_pinned, count, datatype, npid, b0_idx, //
                      MPI_COMM_WORLD, &data->recv_reqs[b0_idx]);
            dst->pinned = true;
        }
    }

    for (size_t b0_idx = 0; b0_idx < blockcount; ++b0_idx) {
        const int3 b0       = b0s[b0_idx];
        const int3 neighbor = (int3){
            b0.x < NGHOST ? -1 : b0.x >= NGHOST + nn.x ? 1 : 0,
            b0.y < NGHOST ? -1 : b0.y >= NGHOST + nn.y ? 1 : 0,
            b0.z < NGHOST ? -1 : b0.z >= NGHOST + nn.z ? 1 : 0,
        };
        const int npid = getPid(pid3d - neighbor, decomp);

        PackedData* src = &data->srcs[b0_idx];
        if (onTheSameNode(pid, npid) || !MPI_USE_PINNED) {
            cudaStreamSynchronize(data->streams[b0_idx]);
            MPI_Isend(src->data, count, datatype, npid, b0_idx, //
                      MPI_COMM_WORLD, &data->send_reqs[b0_idx]);
        }
        else {
            acPinPackedData(device, data->streams[b0_idx], src);
            cudaStreamSynchronize(data->streams[b0_idx]);
            MPI_Isend(src->data_pinned, count, datatype, npid, b0_idx, //
                      MPI_COMM_WORLD, &data->send_reqs[b0_idx]);
        }
    }

    return AC_SUCCESS;
}

static void
acTransferCommDataWait(const CommData data)
{
    MPI_Waitall(data.count, data.recv_reqs, MPI_STATUSES_IGNORE);
    MPI_Waitall(data.count, data.send_reqs, MPI_STATUSES_IGNORE);
}

typedef struct {
    Device device;
    AcMesh submesh;
    uint3_64 decomposition;
    bool initialized;

    int3 nn;
    CommData corner_data;
    CommData edgex_data;
    CommData edgey_data;
    CommData edgez_data;
    CommData sidexy_data;
    CommData sidexz_data;
    CommData sideyz_data;

    // int comm_cart;
} Grid;

static Grid grid = {};

AcResult
acGridSynchronizeStream(const Stream stream)
{
    ERRCHK(grid.initialized);

    acDeviceSynchronizeStream(grid.device, stream);
    MPI_Barrier(MPI_COMM_WORLD);
    return AC_SUCCESS;
}

AcResult
acGridRandomize(void)
{
    ERRCHK(grid.initialized);

    AcMesh host;
    acMeshCreate(grid.submesh.info, &host);
    acMeshRandomize(&host);
    acDeviceLoadMesh(grid.device, STREAM_DEFAULT, host);
    acMeshDestroy(&host);

    return AC_SUCCESS;
}

AcResult
acGridInit(const AcMeshInfo info)
{
    ERRCHK(!grid.initialized);

    // Check that MPI is initialized
    int nprocs, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Decompose
    AcMeshInfo submesh_info      = info;
    const uint3_64 decomposition = decompose(nprocs);
    const int3 pid3d             = getPid3D(pid, decomposition);

    MPI_Barrier(MPI_COMM_WORLD);
    printf("Processor %s. Process %d of %d: (%d, %d, %d)\n", processor_name, pid, nprocs, pid3d.x,
           pid3d.y, pid3d.z);
    printf("Decomposition: %lu, %lu, %lu\n", decomposition.x, decomposition.y, decomposition.z);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    ERRCHK_ALWAYS(info.int_params[AC_nx] % decomposition.x == 0);
    ERRCHK_ALWAYS(info.int_params[AC_ny] % decomposition.y == 0);
    ERRCHK_ALWAYS(info.int_params[AC_nz] % decomposition.z == 0);

    const int submesh_nx                       = info.int_params[AC_nx] / decomposition.x;
    const int submesh_ny                       = info.int_params[AC_ny] / decomposition.y;
    const int submesh_nz                       = info.int_params[AC_nz] / decomposition.z;
    submesh_info.int_params[AC_nx]             = submesh_nx;
    submesh_info.int_params[AC_ny]             = submesh_ny;
    submesh_info.int_params[AC_nz]             = submesh_nz;
    submesh_info.int3_params[AC_global_grid_n] = (int3){
        info.int_params[AC_nx],
        info.int_params[AC_ny],
        info.int_params[AC_nz],
    };
    submesh_info.int3_params[AC_multigpu_offset] = pid3d *
                                                   (int3){submesh_nx, submesh_ny, submesh_nz};
    acUpdateBuiltinParams(&submesh_info);

    // GPU alloc
    int devices_per_node = -1;
    cudaGetDeviceCount(&devices_per_node);

    Device device;
    acDeviceCreate(pid % devices_per_node, submesh_info, &device);

    // CPU alloc
    AcMesh submesh;
    acMeshCreate(submesh_info, &submesh);

    // Setup the global grid structure
    grid.device        = device;
    grid.submesh       = submesh;
    grid.decomposition = decomposition;
    grid.initialized   = true;

    // Configure
    const int3 nn = (int3){
        device->local_config.int_params[AC_nx],
        device->local_config.int_params[AC_ny],
        device->local_config.int_params[AC_nz],
    };

    // Create CommData
    // We have 8 corners, 12 edges, and 6 sides
    //
    // For simplicity's sake all data blocks inside a single CommData struct
    // have the same dimensions.
    grid.nn          = nn;
    grid.corner_data = acCreateCommData(device, (int3){NGHOST, NGHOST, NGHOST}, 8);
    grid.edgex_data  = acCreateCommData(device, (int3){nn.x, NGHOST, NGHOST}, 4);
    grid.edgey_data  = acCreateCommData(device, (int3){NGHOST, nn.y, NGHOST}, 4);
    grid.edgez_data  = acCreateCommData(device, (int3){NGHOST, NGHOST, nn.z}, 4);
    grid.sidexy_data = acCreateCommData(device, (int3){nn.x, nn.y, NGHOST}, 2);
    grid.sidexz_data = acCreateCommData(device, (int3){nn.x, NGHOST, nn.z}, 2);
    grid.sideyz_data = acCreateCommData(device, (int3){NGHOST, nn.y, nn.z}, 2);

    acGridSynchronizeStream(STREAM_ALL);
    return AC_SUCCESS;
}

AcResult
acGridQuit(void)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(STREAM_ALL);

    acDestroyCommData(grid.device, &grid.corner_data);
    acDestroyCommData(grid.device, &grid.edgex_data);
    acDestroyCommData(grid.device, &grid.edgey_data);
    acDestroyCommData(grid.device, &grid.edgez_data);
    acDestroyCommData(grid.device, &grid.sidexy_data);
    acDestroyCommData(grid.device, &grid.sidexz_data);
    acDestroyCommData(grid.device, &grid.sideyz_data);

    grid.initialized   = false;
    grid.decomposition = (uint3_64){0, 0, 0};
    acMeshDestroy(&grid.submesh);
    acDeviceDestroy(grid.device);

    acGridSynchronizeStream(STREAM_ALL);
    return AC_SUCCESS;
}

AcResult
acGridLoadScalarUniform(const Stream stream, const AcRealParam param, const AcReal value)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);

#if AC_DOUBLE_PRECISION == 1
    MPI_Datatype datatype = MPI_DOUBLE;
#else
    MPI_Datatype datatype = MPI_FLOAT;
#endif

    const int root_proc = 0;
    AcReal buffer       = value;
    MPI_Bcast(&buffer, 1, datatype, root_proc, MPI_COMM_WORLD);

    acDeviceLoadScalarUniform(grid.device, stream, param, buffer);
    return AC_SUCCESS;
}

/** */
AcResult
acGridLoadVectorUniform(const Stream stream, const AcReal3Param param, const AcReal3 value)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);

#if AC_DOUBLE_PRECISION == 1
    MPI_Datatype datatype = MPI_DOUBLE;
#else
    MPI_Datatype datatype = MPI_FLOAT;
#endif

    const int root_proc = 0;
    AcReal3 buffer      = value;
    MPI_Bcast(&buffer, 3, datatype, root_proc, MPI_COMM_WORLD);

    acDeviceLoadVectorUniform(grid.device, stream, param, buffer);
    return AC_SUCCESS;
}

AcResult
acGridLoadMesh(const Stream stream, const AcMesh host_mesh)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);

    acDeviceDistributeMeshMPI(host_mesh, grid.decomposition, &grid.submesh);
    acDeviceLoadMesh(grid.device, stream, grid.submesh);

    return AC_SUCCESS;
}

AcResult
acGridStoreMesh(const Stream stream, AcMesh* host_mesh)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);

    acDeviceStoreMesh(grid.device, stream, &grid.submesh);
    acGridSynchronizeStream(stream);

    acDeviceGatherMeshMPI(grid.submesh, grid.decomposition, host_mesh);

    return AC_SUCCESS;
}

/*
// Unused
AcResult
acGridIntegratePipelined(const Stream stream, const AcReal dt)
{
    ERRCHK(grid.initialized);
    acGridLoadScalarUniform(stream, AC_dt, dt);
    acGridSynchronizeStream(stream);

    const Device device = grid.device;
    const int3 nn       = grid.nn;
#if MPI_INCL_CORNERS
    CommData corner_data = grid.corner_data; // Do not rm: required for corners
#endif                                       // MPI_INCL_CORNERS
    CommData edgex_data  = grid.edgex_data;
    CommData edgey_data  = grid.edgey_data;
    CommData edgez_data  = grid.edgez_data;
    CommData sidexy_data = grid.sidexy_data;
    CommData sidexz_data = grid.sidexz_data;
    CommData sideyz_data = grid.sideyz_data;

// Corners
#if MPI_INCL_CORNERS
    // Do not rm: required for corners
    const int3 corner_b0s[] = {
        (int3){0, 0, 0},
        (int3){NGHOST + nn.x, 0, 0},
        (int3){0, NGHOST + nn.y, 0},
        (int3){0, 0, NGHOST + nn.z},

        (int3){NGHOST + nn.x, NGHOST + nn.y, 0},
        (int3){NGHOST + nn.x, 0, NGHOST + nn.z},
        (int3){0, NGHOST + nn.y, NGHOST + nn.z},
        (int3){NGHOST + nn.x, NGHOST + nn.y, NGHOST + nn.z},
    };
#endif // MPI_INCL_CORNERS

    // Edges X
    const int3 edgex_b0s[] = {
        (int3){NGHOST, 0, 0},
        (int3){NGHOST, NGHOST + nn.y, 0},

        (int3){NGHOST, 0, NGHOST + nn.z},
        (int3){NGHOST, NGHOST + nn.y, NGHOST + nn.z},
    };

    // Edges Y
    const int3 edgey_b0s[] = {
        (int3){0, NGHOST, 0},
        (int3){NGHOST + nn.x, NGHOST, 0},

        (int3){0, NGHOST, NGHOST + nn.z},
        (int3){NGHOST + nn.x, NGHOST, NGHOST + nn.z},
    };

    // Edges Z
    const int3 edgez_b0s[] = {
        (int3){0, 0, NGHOST},
        (int3){NGHOST + nn.x, 0, NGHOST},

        (int3){0, NGHOST + nn.y, NGHOST},
        (int3){NGHOST + nn.x, NGHOST + nn.y, NGHOST},
    };

    // Sides XY
    const int3 sidexy_b0s[] = {
        (int3){NGHOST, NGHOST, 0},             //
        (int3){NGHOST, NGHOST, NGHOST + nn.z}, //
    };

    // Sides XZ
    const int3 sidexz_b0s[] = {
        (int3){NGHOST, 0, NGHOST},             //
        (int3){NGHOST, NGHOST + nn.y, NGHOST}, //
    };

    // Sides YZ
    const int3 sideyz_b0s[] = {
        (int3){0, NGHOST, NGHOST},             //
        (int3){NGHOST + nn.x, NGHOST, NGHOST}, //
    };

    for (int isubstep = 0; isubstep < 3; ++isubstep) {
        acDeviceSynchronizeStream(device, STREAM_ALL);
        MPI_Barrier(MPI_COMM_WORLD);

#if MPI_COMPUTE_ENABLED
        acPackCommData(device, sidexy_b0s, &sidexy_data);
        acPackCommData(device, sidexz_b0s, &sidexz_data);
        acPackCommData(device, sideyz_b0s, &sideyz_data);
#endif // MPI_COMPUTE_ENABLED

#if MPI_COMM_ENABLED
        acTransferCommData(device, sidexy_b0s, &sidexy_data);
        acTransferCommData(device, sidexz_b0s, &sidexz_data);
        acTransferCommData(device, sideyz_b0s, &sideyz_data);
#endif // MPI_COMM_ENABLED

#if MPI_COMPUTE_ENABLED
        //////////// INNER INTEGRATION //////////////
        {
            const int3 m1 = (int3){2 * NGHOST, 2 * NGHOST, 2 * NGHOST};
            const int3 m2 = nn;
            acKernelIntegrateSubstep(device->streams[STREAM_16], isubstep, m1, m2, device->vba);
        }

        acPackCommData(device, edgex_b0s, &edgex_data);
        acPackCommData(device, edgey_b0s, &edgey_data);
        acPackCommData(device, edgez_b0s, &edgez_data);
#endif // MPI_COMPUTE_ENABLED

#if MPI_COMM_ENABLED
        acTransferCommDataWait(sidexy_data);
        acUnpinCommData(device, &sidexy_data);
        acTransferCommDataWait(sidexz_data);
        acUnpinCommData(device, &sidexz_data);
        acTransferCommDataWait(sideyz_data);
        acUnpinCommData(device, &sideyz_data);

        acTransferCommData(device, edgex_b0s, &edgex_data);
        acTransferCommData(device, edgey_b0s, &edgey_data);
        acTransferCommData(device, edgez_b0s, &edgez_data);
#endif // MPI_COMM_ENABLED

#if MPI_COMPUTE_ENABLED
#if MPI_INCL_CORNERS
        acPackCommData(device, corner_b0s, &corner_data); // Do not rm: required for corners
#endif                                                    // MPI_INCL_CORNERS
        acUnpackCommData(device, sidexy_b0s, &sidexy_data);
        acUnpackCommData(device, sidexz_b0s, &sidexz_data);
        acUnpackCommData(device, sideyz_b0s, &sideyz_data);
#endif // MPI_COMPUTE_ENABLED

#if MPI_COMM_ENABLED
        acTransferCommDataWait(edgex_data);
        acUnpinCommData(device, &edgex_data);
        acTransferCommDataWait(edgey_data);
        acUnpinCommData(device, &edgey_data);
        acTransferCommDataWait(edgez_data);
        acUnpinCommData(device, &edgez_data);

#if MPI_INCL_CORNERS
        acTransferCommData(device, corner_b0s, &corner_data); // Do not rm: required for corners
#endif                                                        // MPI_INCL_CORNERS
#endif                                                        // MPI_COMM_ENABLED

#if MPI_COMPUTE_ENABLED
        acUnpackCommData(device, edgex_b0s, &edgex_data);
        acUnpackCommData(device, edgey_b0s, &edgey_data);
        acUnpackCommData(device, edgez_b0s, &edgez_data);
#endif // MPI_COMPUTE_ENABLED

#if MPI_COMM_ENABLED
#if MPI_INCL_CORNERS
        acTransferCommDataWait(corner_data);   // Do not rm: required for corners
        acUnpinCommData(device, &corner_data); // Do not rm: required for corners
#endif                                         // MPI_INCL_CORNERS
#endif                                         // MPI_COMM_ENABLED
#if MPI_COMPUTE_ENABLED
#if MPI_INCL_CORNERS
        acUnpackCommData(device, corner_b0s, &corner_data); // Do not rm: required for corners
#endif                                                      // MPI_INCL_CORNERS
#endif                                                      // MPI_COMPUTE_ENABLED

        // Wait for unpacking
        acSyncCommData(sidexy_data);
        acSyncCommData(sidexz_data);
        acSyncCommData(sideyz_data);
        acSyncCommData(edgex_data);
        acSyncCommData(edgey_data);
        acSyncCommData(edgez_data);
#if MPI_INCL_CORNERS
        acSyncCommData(corner_data); // Do not rm: required for corners
#endif                               // MPI_INCL_CORNERS

#if MPI_COMPUTE_ENABLED
        { // Front
            const int3 m1 = (int3){NGHOST, NGHOST, NGHOST};
            const int3 m2 = m1 + (int3){nn.x, nn.y, NGHOST};
            acKernelIntegrateSubstep(device->streams[STREAM_0], isubstep, m1, m2, device->vba);
        }
        { // Back
            const int3 m1 = (int3){NGHOST, NGHOST, nn.z};
            const int3 m2 = m1 + (int3){nn.x, nn.y, NGHOST};
            acKernelIntegrateSubstep(device->streams[STREAM_1], isubstep, m1, m2, device->vba);
        }
        { // Bottom
            const int3 m1 = (int3){NGHOST, NGHOST, 2 * NGHOST};
            const int3 m2 = m1 + (int3){nn.x, NGHOST, nn.z - 2 * NGHOST};
            acKernelIntegrateSubstep(device->streams[STREAM_2], isubstep, m1, m2, device->vba);
        }
        { // Top
            const int3 m1 = (int3){NGHOST, nn.y, 2 * NGHOST};
            const int3 m2 = m1 + (int3){nn.x, NGHOST, nn.z - 2 * NGHOST};
            acKernelIntegrateSubstep(device->streams[STREAM_3], isubstep, m1, m2, device->vba);
        }
        { // Left
            const int3 m1 = (int3){NGHOST, 2 * NGHOST, 2 * NGHOST};
            const int3 m2 = m1 + (int3){NGHOST, nn.y - 2 * NGHOST, nn.z - 2 * NGHOST};
            acKernelIntegrateSubstep(device->streams[STREAM_4], isubstep, m1, m2, device->vba);
        }
        { // Right
            const int3 m1 = (int3){nn.x, 2 * NGHOST, 2 * NGHOST};
            const int3 m2 = m1 + (int3){NGHOST, nn.y - 2 * NGHOST, nn.z - 2 * NGHOST};
            acKernelIntegrateSubstep(device->streams[STREAM_5], isubstep, m1, m2, device->vba);
        }
#endif // MPI_COMPUTE_ENABLED
        acDeviceSwapBuffers(device);
    }

    // Does not have to be STREAM_ALL, only the streams used with
    // acDeviceIntegrateSubstep (less likely to break this way though)
    acDeviceSynchronizeStream(device, STREAM_ALL); // Wait until inner and outer done
    return AC_SUCCESS;
}
*/

AcResult
acGridIntegrate(const Stream stream, const AcReal dt)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);

    const Device device = grid.device;
    const int3 nn       = grid.nn;
#if MPI_INCL_CORNERS
    CommData corner_data = grid.corner_data; // Do not rm: required for corners
#endif                                       // MPI_INCL_CORNERS
    CommData edgex_data  = grid.edgex_data;
    CommData edgey_data  = grid.edgey_data;
    CommData edgez_data  = grid.edgez_data;
    CommData sidexy_data = grid.sidexy_data;
    CommData sidexz_data = grid.sidexz_data;
    CommData sideyz_data = grid.sideyz_data;

    acGridLoadScalarUniform(stream, AC_dt, dt);
    acDeviceSynchronizeStream(device, stream);

// Corners
#if MPI_INCL_CORNERS
    // Do not rm: required for corners
    const int3 corner_b0s[] = {
        (int3){0, 0, 0},
        (int3){NGHOST + nn.x, 0, 0},
        (int3){0, NGHOST + nn.y, 0},
        (int3){0, 0, NGHOST + nn.z},

        (int3){NGHOST + nn.x, NGHOST + nn.y, 0},
        (int3){NGHOST + nn.x, 0, NGHOST + nn.z},
        (int3){0, NGHOST + nn.y, NGHOST + nn.z},
        (int3){NGHOST + nn.x, NGHOST + nn.y, NGHOST + nn.z},
    };
#endif // MPI_INCL_CORNERS

    // Edges X
    const int3 edgex_b0s[] = {
        (int3){NGHOST, 0, 0},
        (int3){NGHOST, NGHOST + nn.y, 0},

        (int3){NGHOST, 0, NGHOST + nn.z},
        (int3){NGHOST, NGHOST + nn.y, NGHOST + nn.z},
    };

    // Edges Y
    const int3 edgey_b0s[] = {
        (int3){0, NGHOST, 0},
        (int3){NGHOST + nn.x, NGHOST, 0},

        (int3){0, NGHOST, NGHOST + nn.z},
        (int3){NGHOST + nn.x, NGHOST, NGHOST + nn.z},
    };

    // Edges Z
    const int3 edgez_b0s[] = {
        (int3){0, 0, NGHOST},
        (int3){NGHOST + nn.x, 0, NGHOST},

        (int3){0, NGHOST + nn.y, NGHOST},
        (int3){NGHOST + nn.x, NGHOST + nn.y, NGHOST},
    };

    // Sides XY
    const int3 sidexy_b0s[] = {
        (int3){NGHOST, NGHOST, 0},             //
        (int3){NGHOST, NGHOST, NGHOST + nn.z}, //
    };

    // Sides XZ
    const int3 sidexz_b0s[] = {
        (int3){NGHOST, 0, NGHOST},             //
        (int3){NGHOST, NGHOST + nn.y, NGHOST}, //
    };

    // Sides YZ
    const int3 sideyz_b0s[] = {
        (int3){0, NGHOST, NGHOST},             //
        (int3){NGHOST + nn.x, NGHOST, NGHOST}, //
    };

    for (int isubstep = 0; isubstep < 3; ++isubstep) {

#if MPI_COMM_ENABLED
#if MPI_INCL_CORNERS
        acPackCommData(device, corner_b0s, &corner_data); // Do not rm: required for corners
#endif                                                    // MPI_INCL_CORNERS
        acPackCommData(device, edgex_b0s, &edgex_data);
        acPackCommData(device, edgey_b0s, &edgey_data);
        acPackCommData(device, edgez_b0s, &edgez_data);
        acPackCommData(device, sidexy_b0s, &sidexy_data);
        acPackCommData(device, sidexz_b0s, &sidexz_data);
        acPackCommData(device, sideyz_b0s, &sideyz_data);
#endif

#if MPI_COMM_ENABLED
        MPI_Barrier(MPI_COMM_WORLD);

#if MPI_GPUDIRECT_DISABLED
#if MPI_INCL_CORNERS
        acTransferCommDataToHost(device, &corner_data); // Do not rm: required for corners
#endif                                                  // MPI_INCL_CORNERS
        acTransferCommDataToHost(device, &edgex_data);
        acTransferCommDataToHost(device, &edgey_data);
        acTransferCommDataToHost(device, &edgez_data);
        acTransferCommDataToHost(device, &sidexy_data);
        acTransferCommDataToHost(device, &sidexz_data);
        acTransferCommDataToHost(device, &sideyz_data);
#endif
#if MPI_INCL_CORNERS
        acTransferCommData(device, corner_b0s, &corner_data); // Do not rm: required for corners
#endif                                                        // MPI_INCL_CORNERS
        acTransferCommData(device, edgex_b0s, &edgex_data);
        acTransferCommData(device, edgey_b0s, &edgey_data);
        acTransferCommData(device, edgez_b0s, &edgez_data);
        acTransferCommData(device, sidexy_b0s, &sidexy_data);
        acTransferCommData(device, sidexz_b0s, &sidexz_data);
        acTransferCommData(device, sideyz_b0s, &sideyz_data);
#endif // MPI_COMM_ENABLED

#if MPI_COMPUTE_ENABLED
        //////////// INNER INTEGRATION //////////////
        {
            const int3 m1 = (int3){2 * NGHOST, 2 * NGHOST, 2 * NGHOST};
            const int3 m2 = nn;
            acKernelIntegrateSubstep(device->streams[STREAM_16], isubstep, m1, m2, device->vba);
        }
////////////////////////////////////////////
#endif // MPI_COMPUTE_ENABLED

#if MPI_COMM_ENABLED
#if MPI_INCL_CORNERS
        acTransferCommDataWait(corner_data); // Do not rm: required for corners
#endif                                       // MPI_INCL_CORNERS
        acTransferCommDataWait(edgex_data);
        acTransferCommDataWait(edgey_data);
        acTransferCommDataWait(edgez_data);
        acTransferCommDataWait(sidexy_data);
        acTransferCommDataWait(sidexz_data);
        acTransferCommDataWait(sideyz_data);

#if MPI_INCL_CORNERS
        acUnpinCommData(device, &corner_data); // Do not rm: required for corners
#endif                                         // MPI_INCL_CORNERS
        acUnpinCommData(device, &edgex_data);
        acUnpinCommData(device, &edgey_data);
        acUnpinCommData(device, &edgez_data);
        acUnpinCommData(device, &sidexy_data);
        acUnpinCommData(device, &sidexz_data);
        acUnpinCommData(device, &sideyz_data);

#if MPI_INCL_CORNERS
        acUnpackCommData(device, corner_b0s, &corner_data);
#endif // MPI_INCL_CORNERS
        acUnpackCommData(device, edgex_b0s, &edgex_data);
        acUnpackCommData(device, edgey_b0s, &edgey_data);
        acUnpackCommData(device, edgez_b0s, &edgez_data);
        acUnpackCommData(device, sidexy_b0s, &sidexy_data);
        acUnpackCommData(device, sidexz_b0s, &sidexz_data);
        acUnpackCommData(device, sideyz_b0s, &sideyz_data);
//////////// OUTER INTEGRATION //////////////

// Wait for unpacking
#if MPI_INCL_CORNERS
        acSyncCommData(corner_data); // Do not rm: required for corners
#endif                               // MPI_INCL_CORNERS
        acSyncCommData(edgex_data);
        acSyncCommData(edgey_data);
        acSyncCommData(edgez_data);
        acSyncCommData(sidexy_data);
        acSyncCommData(sidexz_data);
        acSyncCommData(sideyz_data);
#endif // MPI_COMM_ENABLED
#if MPI_COMPUTE_ENABLED
        { // Front
            const int3 m1 = (int3){NGHOST, NGHOST, NGHOST};
            const int3 m2 = m1 + (int3){nn.x, nn.y, NGHOST};
            acKernelIntegrateSubstep(device->streams[STREAM_0], isubstep, m1, m2, device->vba);
        }
        { // Back
            const int3 m1 = (int3){NGHOST, NGHOST, nn.z};
            const int3 m2 = m1 + (int3){nn.x, nn.y, NGHOST};
            acKernelIntegrateSubstep(device->streams[STREAM_1], isubstep, m1, m2, device->vba);
        }
        { // Bottom
            const int3 m1 = (int3){NGHOST, NGHOST, 2 * NGHOST};
            const int3 m2 = m1 + (int3){nn.x, NGHOST, nn.z - 2 * NGHOST};
            acKernelIntegrateSubstep(device->streams[STREAM_2], isubstep, m1, m2, device->vba);
        }
        { // Top
            const int3 m1 = (int3){NGHOST, nn.y, 2 * NGHOST};
            const int3 m2 = m1 + (int3){nn.x, NGHOST, nn.z - 2 * NGHOST};
            acKernelIntegrateSubstep(device->streams[STREAM_3], isubstep, m1, m2, device->vba);
        }
        { // Left
            const int3 m1 = (int3){NGHOST, 2 * NGHOST, 2 * NGHOST};
            const int3 m2 = m1 + (int3){NGHOST, nn.y - 2 * NGHOST, nn.z - 2 * NGHOST};
            acKernelIntegrateSubstep(device->streams[STREAM_4], isubstep, m1, m2, device->vba);
        }
        { // Right
            const int3 m1 = (int3){nn.x, 2 * NGHOST, 2 * NGHOST};
            const int3 m2 = m1 + (int3){NGHOST, nn.y - 2 * NGHOST, nn.z - 2 * NGHOST};
            acKernelIntegrateSubstep(device->streams[STREAM_5], isubstep, m1, m2, device->vba);
        }
#endif // MPI_COMPUTE_ENABLED
        acDeviceSwapBuffers(device);
        acDeviceSynchronizeStream(device, STREAM_ALL); // Wait until inner and outer done
        ////////////////////////////////////////////
    }

    return AC_SUCCESS;
}

AcResult
acGridPeriodicBoundconds(const Stream stream)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);

    const Device device  = grid.device;
    const int3 nn        = grid.nn;
    CommData corner_data = grid.corner_data;
    CommData edgex_data  = grid.edgex_data;
    CommData edgey_data  = grid.edgey_data;
    CommData edgez_data  = grid.edgez_data;
    CommData sidexy_data = grid.sidexy_data;
    CommData sidexz_data = grid.sidexz_data;
    CommData sideyz_data = grid.sideyz_data;

    // Corners
    const int3 corner_b0s[] = {
        (int3){0, 0, 0},
        (int3){NGHOST + nn.x, 0, 0},
        (int3){0, NGHOST + nn.y, 0},
        (int3){0, 0, NGHOST + nn.z},

        (int3){NGHOST + nn.x, NGHOST + nn.y, 0},
        (int3){NGHOST + nn.x, 0, NGHOST + nn.z},
        (int3){0, NGHOST + nn.y, NGHOST + nn.z},
        (int3){NGHOST + nn.x, NGHOST + nn.y, NGHOST + nn.z},
    };

    // Edges X
    const int3 edgex_b0s[] = {
        (int3){NGHOST, 0, 0},
        (int3){NGHOST, NGHOST + nn.y, 0},

        (int3){NGHOST, 0, NGHOST + nn.z},
        (int3){NGHOST, NGHOST + nn.y, NGHOST + nn.z},
    };

    // Edges Y
    const int3 edgey_b0s[] = {
        (int3){0, NGHOST, 0},
        (int3){NGHOST + nn.x, NGHOST, 0},

        (int3){0, NGHOST, NGHOST + nn.z},
        (int3){NGHOST + nn.x, NGHOST, NGHOST + nn.z},
    };

    // Edges Z
    const int3 edgez_b0s[] = {
        (int3){0, 0, NGHOST},
        (int3){NGHOST + nn.x, 0, NGHOST},

        (int3){0, NGHOST + nn.y, NGHOST},
        (int3){NGHOST + nn.x, NGHOST + nn.y, NGHOST},
    };

    // Sides XY
    const int3 sidexy_b0s[] = {
        (int3){NGHOST, NGHOST, 0},             //
        (int3){NGHOST, NGHOST, NGHOST + nn.z}, //
    };

    // Sides XZ
    const int3 sidexz_b0s[] = {
        (int3){NGHOST, 0, NGHOST},             //
        (int3){NGHOST, NGHOST + nn.y, NGHOST}, //
    };

    // Sides YZ
    const int3 sideyz_b0s[] = {
        (int3){0, NGHOST, NGHOST},             //
        (int3){NGHOST + nn.x, NGHOST, NGHOST}, //
    };

    acPackCommData(device, corner_b0s, &corner_data);
    acPackCommData(device, edgex_b0s, &edgex_data);
    acPackCommData(device, edgey_b0s, &edgey_data);
    acPackCommData(device, edgez_b0s, &edgez_data);
    acPackCommData(device, sidexy_b0s, &sidexy_data);
    acPackCommData(device, sidexz_b0s, &sidexz_data);
    acPackCommData(device, sideyz_b0s, &sideyz_data);

    MPI_Barrier(MPI_COMM_WORLD);

#if MPI_GPUDIRECT_DISABLED
    acTransferCommDataToHost(device, &corner_data);
    acTransferCommDataToHost(device, &edgex_data);
    acTransferCommDataToHost(device, &edgey_data);
    acTransferCommDataToHost(device, &edgez_data);
    acTransferCommDataToHost(device, &sidexy_data);
    acTransferCommDataToHost(device, &sidexz_data);
    acTransferCommDataToHost(device, &sideyz_data);
#endif

    acTransferCommData(device, corner_b0s, &corner_data);
    acTransferCommData(device, edgex_b0s, &edgex_data);
    acTransferCommData(device, edgey_b0s, &edgey_data);
    acTransferCommData(device, edgez_b0s, &edgez_data);
    acTransferCommData(device, sidexy_b0s, &sidexy_data);
    acTransferCommData(device, sidexz_b0s, &sidexz_data);
    acTransferCommData(device, sideyz_b0s, &sideyz_data);

    acTransferCommDataWait(corner_data);
    acTransferCommDataWait(edgex_data);
    acTransferCommDataWait(edgey_data);
    acTransferCommDataWait(edgez_data);
    acTransferCommDataWait(sidexy_data);
    acTransferCommDataWait(sidexz_data);
    acTransferCommDataWait(sideyz_data);

#if MPI_GPUDIRECT_DISABLED
    acTransferCommDataToDevice(device, &corner_data);
    acTransferCommDataToDevice(device, &edgex_data);
    acTransferCommDataToDevice(device, &edgey_data);
    acTransferCommDataToDevice(device, &edgez_data);
    acTransferCommDataToDevice(device, &sidexy_data);
    acTransferCommDataToDevice(device, &sidexz_data);
    acTransferCommDataToDevice(device, &sideyz_data);
#endif

    acUnpinCommData(device, &corner_data);
    acUnpinCommData(device, &edgex_data);
    acUnpinCommData(device, &edgey_data);
    acUnpinCommData(device, &edgez_data);
    acUnpinCommData(device, &sidexy_data);
    acUnpinCommData(device, &sidexz_data);
    acUnpinCommData(device, &sideyz_data);

    acUnpackCommData(device, corner_b0s, &corner_data);
    acUnpackCommData(device, edgex_b0s, &edgex_data);
    acUnpackCommData(device, edgey_b0s, &edgey_data);
    acUnpackCommData(device, edgez_b0s, &edgez_data);
    acUnpackCommData(device, sidexy_b0s, &sidexy_data);
    acUnpackCommData(device, sidexz_b0s, &sidexz_data);
    acUnpackCommData(device, sideyz_b0s, &sideyz_data);

    // Wait for unpacking
    acSyncCommData(corner_data);
    acSyncCommData(edgex_data);
    acSyncCommData(edgey_data);
    acSyncCommData(edgez_data);
    acSyncCommData(sidexy_data);
    acSyncCommData(sidexz_data);
    acSyncCommData(sideyz_data);
    return AC_SUCCESS;
}

static AcResult
acMPIReduceScal(const AcReal local_result, const ReductionType rtype, AcReal* result)
{

    MPI_Op op;
    if (rtype == RTYPE_MAX) {
        op = MPI_MAX;
    }
    else if (rtype == RTYPE_MIN) {
        op = MPI_MIN;
    }
    else if (rtype == RTYPE_RMS || rtype == RTYPE_RMS_EXP || rtype == RTYPE_SUM) {
        op = MPI_SUM;
    }
    else {
        ERROR("Unrecognised rtype");
    }

#if AC_DOUBLE_PRECISION == 1
    MPI_Datatype datatype = MPI_DOUBLE;
#else
    MPI_Datatype datatype = MPI_FLOAT;
#endif

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    AcReal mpi_res;
    MPI_Reduce(&local_result, &mpi_res, 1, datatype, op, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        if (rtype == RTYPE_RMS || rtype == RTYPE_RMS_EXP) {
            const AcReal inv_n = AcReal(1.) /
                                 (grid.nn.x * grid.decomposition.x * grid.nn.y *
                                  grid.decomposition.y * grid.nn.z * grid.decomposition.z);
            mpi_res = sqrt(inv_n * mpi_res);
        }
        *result = mpi_res;
    }
    return AC_SUCCESS;
}

AcResult
acGridReduceScal(const Stream stream, const ReductionType rtype,
                 const VertexBufferHandle vtxbuf_handle, AcReal* result)
{
    ERRCHK(grid.initialized);

    const Device device = grid.device;

    acGridSynchronizeStream(STREAM_ALL);
    // MPI_Barrier(MPI_COMM_WORLD);

    AcReal local_result;
    acDeviceReduceScal(device, stream, rtype, vtxbuf_handle, &local_result);

    return acMPIReduceScal(local_result, rtype, result);
}

AcResult
acGridReduceVec(const Stream stream, const ReductionType rtype, const VertexBufferHandle vtxbuf0,
                const VertexBufferHandle vtxbuf1, const VertexBufferHandle vtxbuf2, AcReal* result)
{
    ERRCHK(grid.initialized);

    const Device device = grid.device;

    acGridSynchronizeStream(STREAM_ALL);
    // MPI_Barrier(MPI_COMM_WORLD);

    AcReal local_result;
    acDeviceReduceVec(device, stream, rtype, vtxbuf0, vtxbuf1, vtxbuf2, &local_result);

    return acMPIReduceScal(local_result, rtype, result);
}

//MV: for MPI we will need acGridReduceVecScal() to get Alfven speeds etc. TODO 

#endif // AC_MPI_ENABLED
