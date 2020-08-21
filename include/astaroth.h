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
#pragma once
#include <cuda_runtime_api.h> // Vector types
#include <float.h>            // FLT_EPSILON, etc
#include <stdio.h>            // printf
#include <stdlib.h>           // size_t

// Library flags
#define STENCIL_ORDER (6)
#define NGHOST (STENCIL_ORDER / 2)

// Built-in types and parameters
#if AC_DOUBLE_PRECISION == 1
typedef double AcReal;
typedef double3 AcReal3;
#define AC_REAL_MAX (DBL_MAX)
#define AC_REAL_MIN (DBL_MIN)
#define AC_REAL_EPSILON (DBL_EPSILON)
#else
typedef float AcReal;
typedef float3 AcReal3;
#define AC_REAL_MAX (FLT_MAX)
#define AC_REAL_MIN (FLT_MIN)
#define AC_REAL_EPSILON (FLT_EPSILON)
#endif

typedef struct {
    AcReal3 row[3];
} AcMatrix;

#include "user_defines.h" // Autogenerated defines from the DSL

typedef enum { AC_SUCCESS = 0, AC_FAILURE = 1 } AcResult;

#define AC_GEN_ID(X) X,
typedef enum {
    AC_FOR_RTYPES(AC_GEN_ID) //
    NUM_RTYPES
} ReductionType;

typedef enum {
    AC_FOR_USER_INT_PARAM_TYPES(AC_GEN_ID) //
    NUM_INT_PARAMS
} AcIntParam;

typedef enum {
    AC_FOR_USER_INT3_PARAM_TYPES(AC_GEN_ID) //
    NUM_INT3_PARAMS
} AcInt3Param;

typedef enum {
    AC_FOR_USER_REAL_PARAM_TYPES(AC_GEN_ID) //
    NUM_REAL_PARAMS
} AcRealParam;

typedef enum {
    AC_FOR_USER_REAL3_PARAM_TYPES(AC_GEN_ID) //
    NUM_REAL3_PARAMS
} AcReal3Param;

typedef enum {
    AC_FOR_SCALARARRAY_HANDLES(AC_GEN_ID) //
    NUM_SCALARARRAY_HANDLES
} ScalarArrayHandle;

typedef enum {
    AC_FOR_VTXBUF_HANDLES(AC_GEN_ID) //
    NUM_VTXBUF_HANDLES
} VertexBufferHandle;
#undef AC_GEN_ID

#define _UNUSED __attribute__((unused)) // Does not give a warning if unused
#define AC_GEN_STR(X) #X,
static const char* rtype_names[] _UNUSED      = {AC_FOR_RTYPES(AC_GEN_STR) "-end-"};
static const char* intparam_names[] _UNUSED   = {AC_FOR_USER_INT_PARAM_TYPES(AC_GEN_STR) "-end-"};
static const char* int3param_names[] _UNUSED  = {AC_FOR_USER_INT3_PARAM_TYPES(AC_GEN_STR) "-end-"};
static const char* realparam_names[] _UNUSED  = {AC_FOR_USER_REAL_PARAM_TYPES(AC_GEN_STR) "-end-"};
static const char* real3param_names[] _UNUSED = {AC_FOR_USER_REAL3_PARAM_TYPES(AC_GEN_STR) "-end-"};
static const char* scalararray_names[] _UNUSED = {AC_FOR_SCALARARRAY_HANDLES(AC_GEN_STR) "-end-"};
static const char* vtxbuf_names[] _UNUSED      = {AC_FOR_VTXBUF_HANDLES(AC_GEN_STR) "-end-"};
#undef AC_GEN_STR
#undef _UNUSED

typedef struct {
    int int_params[NUM_INT_PARAMS];
    int3 int3_params[NUM_INT3_PARAMS];
    AcReal real_params[NUM_REAL_PARAMS];
    AcReal3 real3_params[NUM_REAL3_PARAMS];
} AcMeshInfo;

typedef struct {
    AcReal* vertex_buffer[NUM_VTXBUF_HANDLES];
    AcMeshInfo info;
} AcMesh;

// Device
typedef struct device_s* Device; // Opaque pointer to device_s. Analogous to dispatchable handles
                                 // in Vulkan, f.ex. VkDevice

// Node
typedef struct node_s* Node; // Opaque pointer to node_s.

// Grid
// typedef struct grid_s* Grid; // Opaque pointer to grid_s

typedef struct {
    int3 m;
    int3 n;
} GridDims;

typedef struct {
    int num_devices;
    Device* devices;

    GridDims grid;
    GridDims subgrid;
} DeviceConfiguration;

#ifdef __cplusplus
extern "C" {
#endif

/*
 * =============================================================================
 * Helper functions
 * =============================================================================
 */
static inline size_t
acVertexBufferSize(const AcMeshInfo info)
{
    return info.int_params[AC_mx] * info.int_params[AC_my] * info.int_params[AC_mz];
}

static inline size_t
acVertexBufferSizeBytes(const AcMeshInfo info)
{
    return sizeof(AcReal) * acVertexBufferSize(info);
}

static inline size_t
acVertexBufferCompdomainSize(const AcMeshInfo info)
{
    return info.int_params[AC_nx] * info.int_params[AC_ny] * info.int_params[AC_nz];
}

static inline size_t
acVertexBufferCompdomainSizeBytes(const AcMeshInfo info)
{
    return sizeof(AcReal) * acVertexBufferCompdomainSize(info);
}

static inline size_t
acVertexBufferIdx(const int i, const int j, const int k, const AcMeshInfo info)
{
    return i +                          //
           j * info.int_params[AC_mx] + //
           k * info.int_params[AC_mx] * info.int_params[AC_my];
}

/** Prints all parameters inside AcMeshInfo */
static inline void
acPrintMeshInfo(const AcMeshInfo config)
{
    for (int i = 0; i < NUM_INT_PARAMS; ++i)
        printf("[%s]: %d\n", intparam_names[i], config.int_params[i]);
    for (int i = 0; i < NUM_INT3_PARAMS; ++i)
        printf("[%s]: (%d, %d, %d)\n", int3param_names[i], config.int3_params[i].x,
               config.int3_params[i].y, config.int3_params[i].z);
    for (int i = 0; i < NUM_REAL_PARAMS; ++i)
        printf("[%s]: %g\n", realparam_names[i], (double)(config.real_params[i]));
    for (int i = 0; i < NUM_REAL3_PARAMS; ++i)
        printf("[%s]: (%g, %g, %g)\n", real3param_names[i], (double)(config.real3_params[i].x),
               (double)(config.real3_params[i].y), (double)(config.real3_params[i].z));
}

/*
 * =============================================================================
 * Legacy interface
 * =============================================================================
 */
/** Allocates all memory and initializes the devices visible to the caller. Should be
 * called before any other function in this interface. */
AcResult acInit(const AcMeshInfo mesh_info);

/** Frees all GPU allocations and resets all devices in the node. Should be
 * called at exit. */
AcResult acQuit(void);

/** Checks whether there are any CUDA devices available. Returns AC_SUCCESS if there is 1 or more,
 * AC_FAILURE otherwise. */
AcResult acCheckDeviceAvailability(void);

/** Synchronizes a specific stream. All streams are synchronized if STREAM_ALL is passed as a
 * parameter*/
AcResult acSynchronizeStream(const Stream stream);

/** */
AcResult acSynchronizeMesh(void);

/** Loads a constant to the memories of the devices visible to the caller */
AcResult acLoadDeviceConstant(const AcRealParam param, const AcReal value);

/** Loads an AcMesh to the devices visible to the caller */
AcResult acLoad(const AcMesh host_mesh);

/** Stores the AcMesh distributed among the devices visible to the caller back to the host*/
AcResult acStore(AcMesh* host_mesh);

/** Performs Runge-Kutta 3 integration. Note: Boundary conditions are not applied after the final
 * substep and the user is responsible for calling acBoundcondStep before reading the data. */
AcResult acIntegrate(const AcReal dt);

/** Applies periodic boundary conditions for the Mesh distributed among the devices visible to
 * the caller*/
AcResult acBoundcondStep(void);

/** Does a scalar reduction with the data stored in some vertex buffer */
AcReal acReduceScal(const ReductionType rtype, const VertexBufferHandle vtxbuf_handle);

/** Does a vector reduction with vertex buffers where the vector components are (a, b, c) */
AcReal acReduceVec(const ReductionType rtype, const VertexBufferHandle a,
                   const VertexBufferHandle b, const VertexBufferHandle c);

/** Stores a subset of the mesh stored across the devices visible to the caller back to host memory.
 */
AcResult acStoreWithOffset(const int3 dst, const size_t num_vertices, AcMesh* host_mesh);

/** Will potentially be deprecated in later versions. Added only to fix backwards compatibility with
 * PC for now.*/
AcResult acIntegrateStep(const int isubstep, const AcReal dt);
AcResult acIntegrateStepWithOffset(const int isubstep, const AcReal dt, const int3 start,
                                   const int3 end);
AcResult acSynchronize(void);
AcResult acLoadWithOffset(const AcMesh host_mesh, const int3 src, const int num_vertices);

/** */
int acGetNumDevicesPerNode(void);

/** */
Node acGetNode(void);

/*
 * =============================================================================
 * Grid interface
 * =============================================================================
 */
#if AC_MPI_ENABLED
/**
Initializes all available devices.

Must compile and run the code with MPI.

Must allocate exactly one process per GPU. And the same number of processes
per node as there are GPUs on that node.

Devices in the grid are configured based on the contents of AcMesh.
 */
AcResult acGridInit(const AcMeshInfo info);

/**
Resets all devices on the current grid.
 */
AcResult acGridQuit(void);

/** */
AcResult acGridSynchronizeStream(const Stream stream);

/** */
AcResult acGridLoadMesh(const Stream stream, const AcMesh host_mesh);

/** */
AcResult acGridStoreMesh(const Stream stream, AcMesh* host_mesh);

/** */
AcResult acGridIntegrate(const Stream stream, const AcReal dt);

/** */
AcResult acGridPeriodicBoundconds(const Stream stream);

/** TODO */
AcResult acGridReduceScal(const Stream stream, const ReductionType rtype,
                          const VertexBufferHandle vtxbuf_handle, AcReal* result);

/** TODO */
AcResult acGridReduceVec(const Stream stream, const ReductionType rtype,
                         const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                         const VertexBufferHandle vtxbuf2, AcReal* result);
#endif // AC_MPI_ENABLED

/*
 * =============================================================================
 * Node interface
 * =============================================================================
 */
/**
Initializes all devices on the current node.

Devices on the node are configured based on the contents of AcMesh.

@return Exit status. Places the newly created handle in the output parameter.
@see AcMeshInfo


Usage example:
@code
AcMeshInfo info;
acLoadConfig(AC_DEFAULT_CONFIG, &info);

Node node;
acNodeCreate(0, info, &node);
acNodeDestroy(node);
@endcode
 */
AcResult acNodeCreate(const int id, const AcMeshInfo node_config, Node* node);

/**
Resets all devices on the current node.

@see acNodeCreate()
 */
AcResult acNodeDestroy(Node node);

/**
Prints information about the devices available on the current node.

Requires that Node has been initialized with
@See acNodeCreate().
*/
AcResult acNodePrintInfo(const Node node);

/**



@see DeviceConfiguration
*/
AcResult acNodeQueryDeviceConfiguration(const Node node, DeviceConfiguration* config);

/** */
AcResult acNodeAutoOptimize(const Node node);

/** */
AcResult acNodeSynchronizeStream(const Node node, const Stream stream);

/** Deprecated ? */
AcResult acNodeSynchronizeVertexBuffer(const Node node, const Stream stream,
                                       const VertexBufferHandle vtxbuf_handle); // Not in Device

/** */
AcResult acNodeSynchronizeMesh(const Node node, const Stream stream); // Not in Device

/** */
AcResult acNodeSwapBuffers(const Node node);

/** */
AcResult acNodeLoadConstant(const Node node, const Stream stream, const AcRealParam param,
                            const AcReal value);

/** Deprecated ? Might be useful though if the user wants to load only one vtxbuf. But in this case
 * the user should supply a AcReal* instead of vtxbuf_handle */
AcResult acNodeLoadVertexBufferWithOffset(const Node node, const Stream stream,
                                          const AcMesh host_mesh,
                                          const VertexBufferHandle vtxbuf_handle, const int3 src,
                                          const int3 dst, const int num_vertices);

/** */
AcResult acNodeLoadMeshWithOffset(const Node node, const Stream stream, const AcMesh host_mesh,
                                  const int3 src, const int3 dst, const int num_vertices);

/** Deprecated ? */
AcResult acNodeLoadVertexBuffer(const Node node, const Stream stream, const AcMesh host_mesh,
                                const VertexBufferHandle vtxbuf_handle);

/** */
AcResult acNodeLoadMesh(const Node node, const Stream stream, const AcMesh host_mesh);

/** Deprecated ? */
AcResult acNodeStoreVertexBufferWithOffset(const Node node, const Stream stream,
                                           const VertexBufferHandle vtxbuf_handle, const int3 src,
                                           const int3 dst, const int num_vertices,
                                           AcMesh* host_mesh);

/** */
AcResult acNodeStoreMeshWithOffset(const Node node, const Stream stream, const int3 src,
                                   const int3 dst, const int num_vertices, AcMesh* host_mesh);

/** Deprecated ? */
AcResult acNodeStoreVertexBuffer(const Node node, const Stream stream,
                                 const VertexBufferHandle vtxbuf_handle, AcMesh* host_mesh);

/** */
AcResult acNodeStoreMesh(const Node node, const Stream stream, AcMesh* host_mesh);

/** */
AcResult acNodeIntegrateSubstep(const Node node, const Stream stream, const int step_number,
                                const int3 start, const int3 end, const AcReal dt);

/** */
AcResult acNodeIntegrate(const Node node, const AcReal dt);

/** */
AcResult acNodePeriodicBoundcondStep(const Node node, const Stream stream,
                                     const VertexBufferHandle vtxbuf_handle);

/** */
AcResult acNodePeriodicBoundconds(const Node node, const Stream stream);

/** */
AcResult acNodeReduceScal(const Node node, const Stream stream, const ReductionType rtype,
                          const VertexBufferHandle vtxbuf_handle, AcReal* result);
/** */
AcResult acNodeReduceVec(const Node node, const Stream stream_type, const ReductionType rtype,
                         const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                         const VertexBufferHandle vtxbuf2, AcReal* result);

/*
 * =============================================================================
 * Device interface
 * =============================================================================
 */
/** */
AcResult acDeviceCreate(const int id, const AcMeshInfo device_config, Device* device);

/** */
AcResult acDeviceDestroy(Device device);

/** */
AcResult acDevicePrintInfo(const Device device);

/** */
AcResult acDeviceAutoOptimize(const Device device);

/** */
AcResult acDeviceSynchronizeStream(const Device device, const Stream stream);

/** */
AcResult acDeviceSwapBuffers(const Device device);

/** */
AcResult acDeviceLoadScalarUniform(const Device device, const Stream stream,
                                   const AcRealParam param, const AcReal value);

/** */
AcResult acDeviceLoadVectorUniform(const Device device, const Stream stream,
                                   const AcReal3Param param, const AcReal3 value);

/** */
AcResult acDeviceLoadIntUniform(const Device device, const Stream stream, const AcIntParam param,
                                const int value);

/** */
AcResult acDeviceLoadInt3Uniform(const Device device, const Stream stream, const AcInt3Param param,
                                 const int3 value);

/** */
AcResult acDeviceLoadScalarArray(const Device device, const Stream stream,
                                 const ScalarArrayHandle handle, const size_t start,
                                 const AcReal* data, const size_t num);

/** */
AcResult acDeviceLoadMeshInfo(const Device device, const AcMeshInfo device_config);

/** */
AcResult acDeviceLoadDefaultUniforms(const Device device);

/** */
AcResult acDeviceLoadVertexBufferWithOffset(const Device device, const Stream stream,
                                            const AcMesh host_mesh,
                                            const VertexBufferHandle vtxbuf_handle, const int3 src,
                                            const int3 dst, const int num_vertices);

/** Deprecated */
AcResult acDeviceLoadMeshWithOffset(const Device device, const Stream stream,
                                    const AcMesh host_mesh, const int3 src, const int3 dst,
                                    const int num_vertices);

/** */
AcResult acDeviceLoadVertexBuffer(const Device device, const Stream stream, const AcMesh host_mesh,
                                  const VertexBufferHandle vtxbuf_handle);

/** */
AcResult acDeviceLoadMesh(const Device device, const Stream stream, const AcMesh host_mesh);

/** */
AcResult acDeviceStoreVertexBufferWithOffset(const Device device, const Stream stream,
                                             const VertexBufferHandle vtxbuf_handle, const int3 src,
                                             const int3 dst, const int num_vertices,
                                             AcMesh* host_mesh);

/** Deprecated */
AcResult acDeviceStoreMeshWithOffset(const Device device, const Stream stream, const int3 src,
                                     const int3 dst, const int num_vertices, AcMesh* host_mesh);

/** */
AcResult acDeviceStoreVertexBuffer(const Device device, const Stream stream,
                                   const VertexBufferHandle vtxbuf_handle, AcMesh* host_mesh);

/** */
AcResult acDeviceStoreMesh(const Device device, const Stream stream, AcMesh* host_mesh);

/** */
AcResult acDeviceTransferVertexBufferWithOffset(const Device src_device, const Stream stream,
                                                const VertexBufferHandle vtxbuf_handle,
                                                const int3 src, const int3 dst,
                                                const int num_vertices, Device dst_device);

/** Deprecated */
AcResult acDeviceTransferMeshWithOffset(const Device src_device, const Stream stream,
                                        const int3 src, const int3 dst, const int num_vertices,
                                        Device* dst_device);

/** */
AcResult acDeviceTransferVertexBuffer(const Device src_device, const Stream stream,
                                      const VertexBufferHandle vtxbuf_handle, Device dst_device);

/** */
AcResult acDeviceTransferMesh(const Device src_device, const Stream stream, Device dst_device);

/** */
AcResult acDeviceIntegrateSubstep(const Device device, const Stream stream, const int step_number,
                                  const int3 start, const int3 end, const AcReal dt);
/** */
AcResult acDevicePeriodicBoundcondStep(const Device device, const Stream stream,
                                       const VertexBufferHandle vtxbuf_handle, const int3 start,
                                       const int3 end);

/** */
AcResult acDevicePeriodicBoundconds(const Device device, const Stream stream, const int3 start,
                                    const int3 end);

/** */
AcResult acDeviceReduceScal(const Device device, const Stream stream, const ReductionType rtype,
                            const VertexBufferHandle vtxbuf_handle, AcReal* result);
/** */
AcResult acDeviceReduceVec(const Device device, const Stream stream_type, const ReductionType rtype,
                           const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                           const VertexBufferHandle vtxbuf2, AcReal* result);
/** */
AcResult acDeviceRunMPITest(void);

/*
 * =============================================================================
 * Helper functions
 * =============================================================================
 */
/** Updates the built-in parameters based on nx, ny and nz */
AcResult acUpdateBuiltinParams(AcMeshInfo* config);

/** Creates a mesh stored in host memory */
AcResult acMeshCreate(const AcMeshInfo mesh_info, AcMesh* mesh);

/** Destroys a mesh stored in host memory */
AcResult acMeshDestroy(AcMesh* mesh);

#ifdef __cplusplus
} // extern "C"
#endif
