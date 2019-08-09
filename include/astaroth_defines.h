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
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <float.h>  // FLT_EPSILON, etc
#include <stdlib.h> // size_t
//#include <vector_types.h> // CUDA vector types (float4, etc)

#ifndef __CUDACC__
typedef struct {
    int x, y, z;
} int3;

typedef struct {
    float x, y;
} float2;

typedef struct {
    float x, y, z;
} float3;

typedef struct {
    double x, y, z;
} double3;
#endif // __CUDACC__

// Library flags
#define VERBOSE_PRINTING (1)

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

#include "stencil_defines.h" // User-defined header

// clang-format off
#define AC_FOR_BUILTIN_INT_PARAM_TYPES(FUNC)\
        FUNC(AC_nx), \
        FUNC(AC_ny), \
        FUNC(AC_nz), \
        FUNC(AC_mx), \
        FUNC(AC_my), \
        FUNC(AC_mz), \
        FUNC(AC_nx_min), \
        FUNC(AC_ny_min), \
        FUNC(AC_nz_min), \
        FUNC(AC_nx_max), \
        FUNC(AC_ny_max), \
        FUNC(AC_nz_max), \
        FUNC(AC_mxy),\
        FUNC(AC_nxy),\
        FUNC(AC_nxyz),\

#define AC_FOR_BUILTIN_INT3_PARAM_TYPES(FUNC)\
        FUNC(AC_global_grid_n),\
        FUNC(AC_multigpu_offset),

#define AC_FOR_BUILTIN_REAL_PARAM_TYPES(FUNC)

#define AC_FOR_BUILTIN_REAL3_PARAM_TYPES(FUNC)
// clang-format on

typedef enum { AC_SUCCESS = 0, AC_FAILURE = 1 } AcResult;

typedef enum {
    RTYPE_MAX,
    RTYPE_MIN,
    RTYPE_RMS,
    RTYPE_RMS_EXP,
    RTYPE_SUM,
    NUM_REDUCTION_TYPES
} ReductionType;

typedef enum {
    STREAM_DEFAULT,
    STREAM_0,
    STREAM_1,
    STREAM_2,
    STREAM_3,
    STREAM_4,
    STREAM_5,
    STREAM_6,
    STREAM_7,
    STREAM_8,
    STREAM_9,
    STREAM_10,
    STREAM_11,
    STREAM_12,
    STREAM_13,
    STREAM_14,
    STREAM_15,
    STREAM_16,
    NUM_STREAM_TYPES
} Stream;
#define STREAM_ALL (NUM_STREAM_TYPES)

#define AC_GEN_ID(X) X
typedef enum {
    AC_FOR_BUILTIN_INT_PARAM_TYPES(AC_GEN_ID) //
    AC_FOR_USER_INT_PARAM_TYPES(AC_GEN_ID)    //
    NUM_INT_PARAMS
} AcIntParam;

typedef enum {
    AC_FOR_BUILTIN_INT3_PARAM_TYPES(AC_GEN_ID) //
    AC_FOR_USER_INT3_PARAM_TYPES(AC_GEN_ID)    //
    NUM_INT3_PARAMS
} AcInt3Param;

typedef enum {
    AC_FOR_BUILTIN_REAL_PARAM_TYPES(AC_GEN_ID) //
    AC_FOR_USER_REAL_PARAM_TYPES(AC_GEN_ID)    //
    NUM_REAL_PARAMS
} AcRealParam;

typedef enum {
    AC_FOR_BUILTIN_REAL3_PARAM_TYPES(AC_GEN_ID) //
    AC_FOR_USER_REAL3_PARAM_TYPES(AC_GEN_ID)    //
    NUM_REAL3_PARAMS
} AcReal3Param;

typedef enum {
    AC_FOR_VTXBUF_HANDLES(AC_GEN_ID) //
    NUM_VTXBUF_HANDLES
} VertexBufferHandle;
#undef AC_GEN_ID

extern const char* intparam_names[];
extern const char* int3param_names[];
extern const char* realparam_names[];
extern const char* real3param_names[];
extern const char* vtxbuf_names[];

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

/*
static inline int
acGetParam(const AcMeshInfo info, const AcIntParam param)
{
    return info.int_params[param];
}

static inline int3
acGetParam(const AcMeshInfo info, const AcInt3Param param)
{
    return info.int3_params[param];
}

static inline AcReal
acGetParam(const AcMeshInfo info, const AcRealParam param)
{
    return info.real_params[param];
}

static inline AcReal3
acGetParam(const AcMeshInfo info, const AcReal3Param param)
{
    return info.real3_params[param];
}

static inline void
acSetParam(const AcIntParam param, const int value, AcMeshInfo* info)
{
    info->int_params[param] = value;
}

static inline void
acSetParam(const AcInt3Param param, const int3 value, AcMeshInfo* info)
{
    info->int3_params[param] = value;
}

static inline void
acSetParam(const AcRealParam param, const AcReal value, AcMeshInfo* info)
{
    info->real_params[param] = value;
}

static inline void
acSetParam(const AcReal3Param param, const AcReal3 value, AcMeshInfo* info)
{
    info->real3_params[param] = value;
}
*/

#ifdef __cplusplus
} // extern "C"
#endif
