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
