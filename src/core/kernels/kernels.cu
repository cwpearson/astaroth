#include "kernels.h"

#include <assert.h>
#include <cuComplex.h>

#include "errchk.h"
#include "math_utils.h"

__device__ __constant__ AcMeshInfo d_mesh_info;

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

static __device__ constexpr int
IDX(const int i)
{
    return i;
}

static __device__ __forceinline__ int
IDX(const int i, const int j, const int k)
{
    return DEVICE_VTXBUF_IDX(i, j, k);
}

static __device__ __forceinline__ int
IDX(const int3 idx)
{
    return DEVICE_VTXBUF_IDX(idx.x, idx.y, idx.z);
}

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

// Kernels /////////////////////////////////////////////////////////////////////
#include "boundconds.cuh"
#include "integration.cuh"
#include "packing.cuh"
#include "reductions.cuh"

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
acDeviceLoadScalarUniform(const Device device, const Stream stream, const AcRealParam param,
                          const AcReal value)
{
    cudaSetDevice(device->id);

    if (param >= NUM_REAL_PARAMS)
        return AC_FAILURE;

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

    if (param >= NUM_REAL3_PARAMS || !NUM_REAL3_PARAMS)
        return AC_FAILURE;

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

    if (param >= NUM_INT_PARAMS)
        return AC_FAILURE;

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

    if (param >= NUM_INT3_PARAMS)
        return AC_FAILURE;

    const size_t offset = (size_t)&d_mesh_info.int3_params[param] - (size_t)&d_mesh_info;
    ERRCHK_CUDA(cudaMemcpyToSymbolAsync(d_mesh_info, &value, sizeof(value), offset,
                                        cudaMemcpyHostToDevice, device->streams[stream]));
    return AC_SUCCESS;
}
