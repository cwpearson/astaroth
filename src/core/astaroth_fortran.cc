#include "astaroth_fortran.h"

#include "astaroth.h"
#include "astaroth_utils.h"

void
acdevicecreate_(const int* id, const AcMeshInfo* info, Device* handle)
{
    // TODO errorcheck
    acDeviceCreate(*id, *info, handle);
}

void
acdevicedestroy_(Device* device)
{
    // TODO errorcheck
    acDeviceDestroy(*device);
}

void
acdeviceprintinfo_(const Device* device)
{
    // TODO errorcheck
    acDevicePrintInfo(*device);
}

void
acupdatebuiltinparams_(AcMeshInfo* info)
{
    // TODO errorcheck
    acUpdateBuiltinParams(info);
}

void
acdeviceswapbuffers_(const Device* device)
{
    acDeviceSwapBuffers(*device);
}

void
acdeviceloadmesh_(const Device* device, const Stream* stream, const AcMesh* host_mesh)
{
    // TODO construct AcMesh from fortran farray
    acDeviceLoadMesh(*device, *stream, *host_mesh);
}

void
acdevicestoremesh_(const Device* device, const Stream* stream, AcMesh* host_mesh)
{
    // TODO construct AcMesh from fortran farray
    acDeviceStoreMesh(*device, *stream, host_mesh);
}

void
acdeviceintegratesubstep_(const Device* device, const Stream* stream, const int* step_number,
                          const int3* start, const int3* end, const AcReal* dt)
{
    acDeviceIntegrateSubstep(*device, *stream, *step_number, *start, *end, *dt);
}

void
acdeviceperiodicboundconds_(const Device* device, const Stream* stream, const int3* start,
                            const int3* end)
{

    acDevicePeriodicBoundconds(*device, *stream, *start, *end);
}

void
acdevicereducescal_(const Device* device, const Stream* stream, const ReductionType* rtype,
                    const VertexBufferHandle* vtxbuf_handle, AcReal* result)
{
    acDeviceReduceScal(*device, *stream, *rtype, *vtxbuf_handle, result);
}

void
acdevicereducevec_(const Device* device, const Stream* stream, const ReductionType* rtype,
                   const VertexBufferHandle* vtxbuf0, const VertexBufferHandle* vtxbuf1,
                   const VertexBufferHandle* vtxbuf2, AcReal* result)
{
    acDeviceReduceVec(*device, *stream, *rtype, *vtxbuf0, *vtxbuf1, *vtxbuf2, result);
}

void
acdevicesynchronizestream_(const Device* device, const Stream* stream)
{
    acDeviceSynchronizeStream(*device, *stream);
}

void
acdeviceloadmeshinfo_(const Device* device, const AcMeshInfo* info)
{
    acDeviceLoadMeshInfo(*device, *info);
}
