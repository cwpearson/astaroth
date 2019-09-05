# Astaroth specification and user manual

Copyright (C) 2014-2019, Johannes Pekkila, Miikka Vaisala.

	   Astaroth is free software: you can redistribute it and/or modify
	   it under the terms of the GNU General Public License as published by
	   the Free Software Foundation, either version 3 of the License, or
	   (at your option) any later version.
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


# Introduction and background

Astaroth is a collection of tools for utilizing multiple graphics processing units (GPUs) efficiently in three-dimensional stencil computations. This document specifies the Astaroth application-programming interface (API) and domain-specific language (DSL).

Astaroth has been designed for the demands in computational sciences, where large stencils are often used to attain sufficient accuracy. The majority of previous work focuses on stencil computations with low-order stencils for which several efficient algorithms have been proposed, whereas work on high-order stencils is more limited. In addition, in computational physics multiple fields interact with each other, such as the velocity and magnetic fields of electrically conducting fluids. Such computations are especially challenging to solve efficiently because of the problem's relatively low operational intensity and the small caches provided by GPUs. Efficient methods for computations with several coupled fields and large stencils have not been addressed sufficiently in prior work.

With Astaroth, we have taken inspiration of image processing and graphics pipelines which rely on holding intermediate data in caches for the duration of computations, and extended the idea to work efficiently also with large three-dimensional stencils and an arbitrary number of coupled fields. As programming GPUs efficiently is relatively verbose and requires deep knowledge of the underlying hardware and execution model, we have created a high-level domain-specific language for expressing a wide range of tasks in computational sciences and provide a source-to-source compiler for translating stencil problems expressed in our language into efficient CUDA kernels.

The kernels generated from the Astaroth DSL are embedded in the Astaroth Core library, which is usable via the Astaroth API. While the Astaroth library is written in C++/CUDA, the API conforms to the C99 standard.


# Publications

The foundational work was done in (Väisälä, Pekkilä, 2017) and the library, API and DSL described in this document were introduced in (Pekkilä, 2019). We kindly wish the users of Astaroth to cite to these publications in their work.

> J. Pekkilä, Astaroth: A Library for Stencil Computations on Graphics Processing Units. Master's thesis, Aalto University School of Science, Espoo, Finland, 2019.

> M. S. Väisälä, Magnetic Phenomena of the Interstellar Medium in Theory and Observation. PhD thesis, University of Helsinki, Finland, 2017.

> J. Pekkilä, M. S. Väisälä, M. Käpylä, P. J. Käpylä, and O. Anjum, “Methods for compressible fluid simulation on GPUs using high-order finite differences, ”Computer Physics Communications, vol. 217, pp. 11–22, Aug. 2017.



# Astaroth API

The Astroth application-programming interface (API) provides the means for controlling execution of user-defined and built-in functions on multiple graphics processing units. Functions in the API are prefixed with lower case ```ac```, while structures and data types are prefixed with capitalized ```Ac```. Compile-time constants, such as definitions and enumerations, have the prefix ```AC_```. All of the API functions return an AcResult value indicating either success or failure. The return codes are
```C
typedef enum { 
    AC_SUCCESS = 0, 
    AC_FAILURE = 1
} AcResult;
```

The API is divided into layers which differ in the level of control provided over the execution. There are two primary layers:

* Device layer
    > Functions start with acDevice*.
    > Provides control over a single GPU.
    > All functions are asynchronous.

* Node layer
    > Functions start with acNode*.
    > Provides control over multiple devices in a single node.
    > All functions are asynchronous and executed concurrently on all devices in the node.
    > Subsequent functions called in the same stream (see Section #Streams and synchronization) are guaranteed to be synchronous.

Finally, a third layer is provided for convenience and backwards compatibility. 

* Astaroth layer (deprecated)
    > Functions start with ac* without Node or Device, f.ex. acInit().
    > Provided for backwards compatibility.
    > Essentially a wrapper for the Node layer. 
    > All functions are guaranteed to be synchronous.

There are also several helper functions defined in `include/astaroth_defines.h`, which can be used for, say, determining the size or performing index calculations within the simulation domain.


## List of Astaroth API functions

Here's a non-exhaustive list of astaroth API functions. For more info and an up-to-date list, see the corresponding header files in `include/astaroth_defines.h`, `include/astaroth.h`, `include/astaroth_node.h`, `include/astaroth_device.h`.

### Initialization, quitting and helper functions

Device layer.
```C
AcResult acDeviceCreate(const int id, const AcMeshInfo device_config, Device* device);
AcResult acDeviceDestroy(Device device);
AcResult acDevicePrintInfo(const Device device);
AcResult acDeviceAutoOptimize(const Device device);
```

Node layer.
```C
AcResult acNodeCreate(const int id, const AcMeshInfo node_config, Node* node);
AcResult acNodeDestroy(Node node);
AcResult acNodePrintInfo(const Node node);
AcResult acNodeQueryDeviceConfiguration(const Node node, DeviceConfiguration* config);
AcResult acNodeAutoOptimize(const Node node);
```

General helper functions.
```C
size_t acVertexBufferSize(const AcMeshInfo info);
size_t acVertexBufferSizeBytes(const AcMeshInfo info);
size_t acVertexBufferCompdomainSize(const AcMeshInfo info);
size_t acVertexBufferCompdomainSizeBytes(const AcMeshInfo info);
size_t acVertexBufferIdx(const int i, const int j, const int k, const AcMeshInfo info);
```

### Loading and storing

Loading meshes and vertex buffers to device memory.
```C
AcResult acDeviceLoadMesh(const Device device, const Stream stream, const AcMesh host_mesh);
AcResult acDeviceLoadMeshWithOffset(const Device device, const Stream stream,
                                    const AcMesh host_mesh, const int3 src, const int3 dst,
                                    const int num_vertices);
AcResult acDeviceLoadVertexBuffer(const Device device, const Stream stream, const AcMesh host_mesh,
                                  const VertexBufferHandle vtxbuf_handle);
AcResult acDeviceLoadVertexBufferWithOffset(const Device device, const Stream stream,
                                            const AcMesh host_mesh,
                                            const VertexBufferHandle vtxbuf_handle, const int3 src,
                                            const int3 dst, const int num_vertices);

AcResult acNodeLoadMesh(const Node node, const Stream stream, const AcMesh host_mesh);
AcResult acNodeLoadMeshWithOffset(const Node node, const Stream stream, const AcMesh host_mesh,
                                  const int3 src, const int3 dst, const int num_vertices);
AcResult acNodeLoadVertexBuffer(const Node node, const Stream stream, const AcMesh host_mesh,
                                const VertexBufferHandle vtxbuf_handle);
AcResult acNodeLoadVertexBufferWithOffset(const Node node, const Stream stream,
                                          const AcMesh host_mesh,
                                          const VertexBufferHandle vtxbuf_handle, const int3 src,
                                          const int3 dst, const int num_vertices);
```

Storing meshes and vertex buffer to host memory.
```C
AcResult acDeviceStoreMesh(const Device device, const Stream stream, AcMesh* host_mesh);
AcResult acDeviceStoreMeshWithOffset(const Device device, const Stream stream, const int3 src,
                                     const int3 dst, const int num_vertices, AcMesh* host_mesh);
AcResult acDeviceStoreVertexBuffer(const Device device, const Stream stream,
                                   const VertexBufferHandle vtxbuf_handle, AcMesh* host_mesh);
AcResult acDeviceStoreMeshWithOffset(const Device device, const Stream stream, const int3 src,
                                     const int3 dst, const int num_vertices, AcMesh* host_mesh);

AcResult acNodeStoreMesh(const Node node, const Stream stream, AcMesh* host_mesh);
AcResult acNodeStoreMeshWithOffset(const Node node, const Stream stream, const int3 src,
                                   const int3 dst, const int num_vertices, AcMesh* host_mesh);
AcResult acNodeStoreVertexBuffer(const Node node, const Stream stream,
                                 const VertexBufferHandle vtxbuf_handle, AcMesh* host_mesh);
AcResult acNodeStoreVertexBufferWithOffset(const Node node, const Stream stream,
                                           const VertexBufferHandle vtxbuf_handle, const int3 src,
                                           const int3 dst, const int num_vertices,
                                           AcMesh* host_mesh);
```

Transferring data between devices
```C
AcResult acDeviceTransferMesh(const Device src_device, const Stream stream, Device dst_device);
AcResult acDeviceTransferMeshWithOffset(const Device src_device, const Stream stream,
                                        const int3 src, const int3 dst, const int num_vertices,
                                        Device* dst_device);
AcResult acDeviceTransferVertexBuffer(const Device src_device, const Stream stream,
                                      const VertexBufferHandle vtxbuf_handle, Device dst_device);
AcResult acDeviceTransferVertexBufferWithOffset(const Device src_device, const Stream stream,
                                                const VertexBufferHandle vtxbuf_handle,
                                                const int3 src, const int3 dst,
                                                const int num_vertices, Device dst_device);
```

Loading uniforms (device constants)
```C
AcResult acDeviceLoadScalarConstant(const Device device, const Stream stream,
                                    const AcRealParam param, const AcReal value);
AcResult acDeviceLoadVectorConstant(const Device device, const Stream stream,
                                    const AcReal3Param param, const AcReal3 value);
AcResult acDeviceLoadIntConstant(const Device device, const Stream stream, const AcIntParam param,
                                 const int value);
AcResult acDeviceLoadInt3Constant(const Device device, const Stream stream, const AcInt3Param param,
                                  const int3 value);
AcResult acDeviceLoadScalarArray(const Device device, const Stream stream,
                                 const ScalarArrayHandle handle, const AcReal* data,
                                 const size_t num);
AcResult acDeviceLoadMeshInfo(const Device device, const Stream stream,
                              const AcMeshInfo device_config);
```


### Computation

```C
AcResult acDeviceIntegrateSubstep(const Device device, const Stream stream, const int step_number,
                                  const int3 start, const int3 end, const AcReal dt);
AcResult acDevicePeriodicBoundcondStep(const Device device, const Stream stream,
                                       const VertexBufferHandle vtxbuf_handle, const int3 start,
                                       const int3 end);
AcResult acDevicePeriodicBoundconds(const Device device, const Stream stream, const int3 start,
                                    const int3 end);
AcResult acDeviceReduceScal(const Device device, const Stream stream, const ReductionType rtype,
                            const VertexBufferHandle vtxbuf_handle, AcReal* result);
AcResult acDeviceReduceVec(const Device device, const Stream stream_type, const ReductionType rtype,
                           const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                           const VertexBufferHandle vtxbuf2, AcReal* result);

AcResult acNodeIntegrateSubstep(const Node node, const Stream stream, const int step_number,
                                const int3 start, const int3 end, const AcReal dt);
AcResult acNodeIntegrate(const Node node, const AcReal dt);
AcResult acNodePeriodicBoundcondStep(const Node node, const Stream stream,
                                     const VertexBufferHandle vtxbuf_handle);
AcResult acNodePeriodicBoundconds(const Node node, const Stream stream);
AcResult acNodeReduceScal(const Node node, const Stream stream, const ReductionType rtype,
                          const VertexBufferHandle vtxbuf_handle, AcReal* result);
AcResult acNodeReduceVec(const Node node, const Stream stream_type, const ReductionType rtype,
                         const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                         const VertexBufferHandle vtxbuf2, AcReal* result);
```

### Synchronization

The functions for synchronizing with certain stream and swapping input and output buffers are
defined as follows. Reductions and boundary conditions declared in the previous subsection operate
in-place on the input array, while integration places the result in the output buffer.
Therefore ac*SwapBuffers() should always be called after an integration substep has been completed.

The node layer introduces two new functions in addition to synchronization and swapping functions,
acNodeSynchronizeVertexBuffer and acNodeSynchronizeMesh. These functions communicate the overlapping
vertices between neighboring devices.
Note that part of the overlapping ghost zone is also communicated when synchronizing vertex buffers
and meshes. Therefore the data in this part of the ghost zone must be up to date before calling
the communication functions.

```C
AcResult acDeviceSynchronizeStream(const Device device, const Stream stream);
AcResult acDeviceSwapBuffers(const Device device);

AcResult acNodeSynchronizeStream(const Node node, const Stream stream);
AcResult acNodeSwapBuffers(const Node node);
AcResult acNodeSynchronizeVertexBuffer(const Node node, const Stream stream,
                                       const VertexBufferHandle vtxbuf_handle);
AcResult acNodeSynchronizeMesh(const Node node, const Stream stream);
```

## Devices

The data type Device is a handle to some single device and is used in the Device layer functions to specify which device should execute the function. A device is created and destroyed with the following interface functions.
```C
AcResult acDeviceCreate(const int device_id, const AcMeshInfo device_config, Device* device);

AcResult acDeviceDestroy(Device device);
```

## Nodes

The data type Node is a handle to some node, which consists of multiple devices. The Node handle is used to specify which node the Node layer functions should operate in. A node is created and destroyed with the following interface functions.
```C
AcResult acNodeCreate(const int id, const AcMeshInfo node_config, Node* node);

AcResult acNodeDestroy(Node node);
```

The function acNodeCreate calls acDeviceCreate for all devices that are visible from the current process. After a node has been created, the devices in it can be retrived with the function
```C
AcResult acNodeQueryDeviceConfiguration(const Node node, DeviceConfiguration* config);
```
where DeviceConfiguration is defined as
```C
typedef struct {
    int num_devices;
    Device* devices; // Todo make this predefined s.t. the user/us do not have to alloc/free

    Grid grid;
    Grid subgrid;
} DeviceConfiguration;
```

## Meshes

Meshes are the primary structures for passing information to the library and kernels. The definition
of a mesh is declared as
```C
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
```

Several built-in parameters, such as the dimensions of the mesh, and all user-defined parameters
are defined in the `AcMeshInfo` structure. Before passing AcMeshInfo to an initialization function,
such as `acDeviceCreate()`, the built-in parameters `AC_nx, AC_ny, AC_nz` must be set. These
parameters define the dimensions of the computational domain of the mesh. For example,
```C
AcMeshInfo info;
info.int_params[AC_nx] = 128;
info.int_params[AC_ny] = 64;
info.int_params[AC_nz] = 32;

Device device;
acDeviceCreate(0, info, &device);
```

AcMesh is used in loading and storing data from host to device and vice versa. Before calling for
example `acDeviceLoadMesh()`, one must ensure that all `NUM_VTXBUF_HANDLES` are pointers to valid
arrays in host memory consisting of `mx * my * mz` elements and stored in order
`i + j * mx + k * mx * my`, where `i`, `j` and `k` correspond to `x`, `y` and `z` coordinate
indices, respectively. The mesh dimensions can be queried with 
```C
int mx = info.int_params[AC_mx];
int my = info.int_params[AC_my];
int mz = info.int_params[AC_mz];
```
after initialization.




### Decomposition
Grids and subgrids contain the dimensions of the the mesh decomposed to multiple devices.
```C
typedef struct {
    int3 m; // Size of the simulation domain (includes the ghost zones)
    int3 n; // Size of the computational domain (without ghost zones)
} Grid;
```
Each device is assigned a block of data 


## Streams and synchronization

## Integration, reductions and boundary conditions


# Astaroth DSL

## Uniforms

### Control flow and implementing switches
// Runtime constants are as fast as compile-time constants as long as
// 1) They are not placed in tight loops, especially those that inlcude global memory accesses, that could be unrolled
// 2) They are not multiplied with each other
// 3) At least 32 neighboring threads in the x-axis access the same constant

// Safe and efficient to use as switches

## Vertex buffers

### Input and output buffers

## Built-in variables and functions

## Functions

### Kernel

### Preprocessed




