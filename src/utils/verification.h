#pragma once
#include <stdbool.h>

#include "memory.h"

#ifdef __cplusplus
extern "C" {
#endif

bool acVerifyMesh(const AcMesh model, const AcMesh candidate);

#ifdef __cplusplus
} // extern "C"
#endif
