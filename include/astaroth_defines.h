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
#include <float.h>        // FLT_EPSILON, etc
#include <stdlib.h>       // size_t
#include <vector_types.h> // CUDA vector types (float4, etc)

#include "stencil_defines.h"

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

#define AC_FOR_BUILTIN_INT3_PARAM_TYPES(FUNC)

#define AC_FOR_BUILTIN_REAL_PARAM_TYPES(FUNC)

#define AC_FOR_BUILTIN_REAL3_PARAM_TYPES(FUNC)
// clang-format on

typedef enum { AC_SUCCESS = 0, AC_FAILURE = 1 } AcResult;

typedef enum { RTYPE_MAX, RTYPE_MIN, RTYPE_RMS, RTYPE_RMS_EXP, NUM_REDUCTION_TYPES } ReductionType;

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

typedef enum { AC_FOR_VTXBUF_HANDLES(AC_GEN_ID) NUM_VTXBUF_HANDLES } VertexBufferHandle;
#undef AC_GEN_ID

extern const char* intparam_names[];
extern const char* int3param_names[];
extern const char* realparam_names[];
extern const char* real3param_names[];
extern const char* vtxbuf_names[];
