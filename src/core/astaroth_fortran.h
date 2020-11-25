#pragma once
#include "astaroth.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Utils
 */
void achostupdatebuiltinparams_(AcMeshInfo* info);

void acgetdevicecount_(int* count);

/**
 * Device
 */
void acdevicecreate_(const int* id, const AcMeshInfo* info, Device* handle);

void acdevicedestroy_(Device* device);

void acdeviceprintinfo_(const Device* device);

void acdeviceloadmeshinfo_(const Device* device, const AcMeshInfo* info);

void acdeviceloadmesh_(const Device* device, const Stream* stream, const AcMeshInfo* info,
                       const int* num_farrays, AcReal* farray);

void acdevicestoremesh_(const Device* device, const Stream* stream, const AcMeshInfo* info,
                        const int* num_farrays, AcReal* farray);

void acdeviceintegratesubstep_(const Device* device, const Stream* stream, const int* step_number,
                               const int3* start, const int3* end, const AcReal* dt);
void acdeviceperiodicboundconds_(const Device* device, const Stream* stream, const int3* start,
                                 const int3* end);

void acdeviceswapbuffers_(const Device* device);

void acdevicereducescal_(const Device* device, const Stream* stream, const ReductionType* rtype,
                         const VertexBufferHandle* vtxbuf_handle, AcReal* result);

void acdevicereducevec_(const Device* device, const Stream* stream, const ReductionType* rtype,
                        const VertexBufferHandle* vtxbuf0, const VertexBufferHandle* vtxbuf1,
                        const VertexBufferHandle* vtxbuf2, AcReal* result);

void acdevicesynchronizestream_(const Device* device, const Stream* stream);

#ifdef __cplusplus
} // extern "C"
#endif
