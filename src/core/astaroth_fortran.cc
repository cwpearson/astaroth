#include "astaroth_fortran.h"

#include "astaroth.h"
#include "errchk.h"

/**
 * Utils
 */
void
acupdatebuiltinparams_(AcMeshInfo* info)
{
    acUpdateBuiltinParams(info);
}

void
acgetdevicecount_(int* count)
{
    ERRCHK_CUDA_ALWAYS(cudaGetDeviceCount(count));
}

/**
 * Device
 */
void
acdevicecreate_(const int* id, const AcMeshInfo* info, Device* handle)
{
    acDeviceCreate(*id, *info, handle);
}

void
acdevicedestroy_(Device* device)
{
    acDeviceDestroy(*device);
}

void
acdeviceprintinfo_(const Device* device)
{
    acDevicePrintInfo(*device);
}

void
acdeviceloadmeshinfo_(const Device* device, const AcMeshInfo* info)
{
    acDeviceLoadMeshInfo(*device, *info);
}

void
acdeviceloadmesh_(const Device* device, const Stream* stream, const AcMeshInfo* info,
                  const int* num_farrays, AcReal* farray)
{
    ERRCHK_ALWAYS(*num_farrays >= NUM_VTXBUF_HANDLES);
    WARNCHK_ALWAYS(*num_farrays == NUM_VTXBUF_HANDLES);
    const size_t mxyz = info->int_params[AC_mx] * info->int_params[AC_mx] * info->int_params[AC_mx];

    AcMesh mesh;
    mesh.info = *info;
    for (int i = 0; i < *num_farrays; ++i)
        mesh.vertex_buffer[i] = &farray[i * mxyz];

    acDeviceLoadMesh(*device, *stream, mesh);
}

void
acdevicestoremesh_(const Device* device, const Stream* stream, const AcMeshInfo* info,
                   const int* num_farrays, AcReal* farray)
{
    ERRCHK_ALWAYS(*num_farrays >= NUM_VTXBUF_HANDLES);
    WARNCHK_ALWAYS(*num_farrays == NUM_VTXBUF_HANDLES);
    AcMesh mesh;
    mesh.info = *info;

    const size_t mxyz = info->int_params[AC_mx] * info->int_params[AC_mx] * info->int_params[AC_mx];
    for (int i = 0; i < *num_farrays; ++i)
        mesh.vertex_buffer[i] = &farray[i * mxyz];

    acDeviceStoreMesh(*device, *stream, &mesh);
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
acdeviceswapbuffers_(const Device* device)
{
    acDeviceSwapBuffers(*device);
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
