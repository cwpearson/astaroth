#pragma once
#include "astaroth.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Utils
 */
void acupdatebuiltinparams_(AcMeshInfo* info);

/**
 * Device
 */
void acdevicecreate_(const int* id, const AcMeshInfo* info, Device* handle);

void acdevicedestroy_(Device* device);

void acdeviceprintinfo_(const Device* device);

#ifdef __cplusplus
} // extern "C"
#endif
