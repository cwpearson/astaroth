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
#pragma once
#include "astaroth.h"

typedef struct {
    int3 m;
    int3 n;
} Grid;

typedef struct device_s* Device; // Opaque pointer to device_s. Analogous to dispatchable handles
                                 // in Vulkan, f.ex. VkDevice

/** */
AcResult printDeviceInfo(const Device device);

/** */
AcResult createDevice(const int id, const AcMeshInfo device_config, Device* device);

/** */
AcResult destroyDevice(Device device);

/** */
AcResult boundcondStep(const Device device, const StreamType stream_type, const int3& start,
                       const int3& end);

/** */
AcResult reduceScal(const Device device, const StreamType stream_type, const ReductionType rtype,
                    const VertexBufferHandle vtxbuf_handle, AcReal* result);

/** */
AcResult reduceVec(const Device device, const StreamType stream_type, const ReductionType rtype,
                   const VertexBufferHandle vec0, const VertexBufferHandle vec1,
                   const VertexBufferHandle vec2, AcReal* result);

/** */
AcResult rkStep(const Device device, const StreamType stream_type, const int step_number,
                const int3& start, const int3& end, const AcReal dt);

/** Sychronizes the device with respect to stream_type. If STREAM_ALL is given as
    a StreamType, the function synchronizes all streams on the device. */
AcResult synchronize(const Device device, const StreamType stream_type);

/** */
AcResult copyMeshToDevice(const Device device, const StreamType stream_type,
                          const AcMesh& host_mesh, const int3& src, const int3& dst,
                          const int num_vertices);

/** */
AcResult copyMeshToHost(const Device device, const StreamType stream_type, const int3& src,
                        const int3& dst, const int num_vertices, AcMesh* host_mesh);

/** */
AcResult copyMeshDeviceToDevice(const Device src, const StreamType stream_type, const int3& src_idx,
                                Device dst, const int3& dst_idx, const int num_vertices);

/** Swaps the input/output buffers used in computations */
AcResult swapBuffers(const Device device);

/** */
AcResult loadDeviceConstant(const Device device, const StreamType stream_type,
                            const AcIntParam param, const int value);

/** */
AcResult loadDeviceConstant(const Device device, const StreamType stream_type,
                            const AcRealParam param, const AcReal value);

/** */
AcResult loadGlobalGrid(const Device device, const Grid grid);

/** */
AcResult autoOptimize(const Device device);

// #define PACKED_DATA_TRANSFERS (1) %JP: placeholder for optimized ghost zone packing and transfers
#if PACKED_DATA_TRANSFERS
// Declarations used for packed data transfers
#endif
