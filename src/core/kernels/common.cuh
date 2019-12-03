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

extern __constant__ AcMeshInfo d_mesh_info;

typedef struct {
    AcReal* in[NUM_VTXBUF_HANDLES];
    AcReal* out[NUM_VTXBUF_HANDLES];

    AcReal* profiles[NUM_SCALARARRAY_HANDLES];
} VertexBufferArray;

static int __device__ __forceinline__
DCONST(const AcIntParam param)
{
    return d_mesh_info.int_params[param];
}
static int3 __device__ __forceinline__
DCONST(const AcInt3Param param)
{
    return d_mesh_info.int3_params[param];
}
static AcReal __device__ __forceinline__
DCONST(const AcRealParam param)
{
    return d_mesh_info.real_params[param];
}
static AcReal3 __device__ __forceinline__
DCONST(const AcReal3Param param)
{
    return d_mesh_info.real3_params[param];
}
static __device__ constexpr VertexBufferHandle
DCONST(const VertexBufferHandle handle)
{
    return handle;
}
#define DEVICE_VTXBUF_IDX(i, j, k) ((i) + (j)*DCONST(AC_mx) + (k)*DCONST(AC_mxy))
#define DEVICE_1D_COMPDOMAIN_IDX(i, j, k) ((i) + (j)*DCONST(AC_nx) + (k)*DCONST(AC_nxy))
#define globalGridN (d_mesh_info.int3_params[AC_global_grid_n])
//#define globalMeshM // Placeholder
//#define localMeshN // Placeholder
//#define localMeshM // Placeholder
//#define localMeshN_min // Placeholder
//#define globalMeshN_min // Placeholder
#define d_multigpu_offset (d_mesh_info.int3_params[AC_multigpu_offset])
//#define d_multinode_offset (d_mesh_info.int3_params[AC_multinode_offset]) // Placeholder

static __device__ constexpr int
IDX(const int i)
{
    return i;
}

static __device__ __forceinline__ int
IDX(const int i, const int j, const int k)
{
    return DEVICE_VTXBUF_IDX(i, j, k);
}

static __device__ __forceinline__ int
IDX(const int3 idx)
{
    return DEVICE_VTXBUF_IDX(idx.x, idx.y, idx.z);
}

//#include <thrust/complex.h>
// using namespace thrust;

#include <cuComplex.h>
#if AC_DOUBLE_PRECISION == 1
typedef cuDoubleComplex acComplex;
#define acComplex(x, y) make_cuDoubleComplex(x, y)
#else
typedef cuFloatComplex acComplex;
#define acComplex(x, y) make_cuFloatComplex(x, y)
#endif
static __device__ inline acComplex
exp(const acComplex& val)
{
    return acComplex(exp(val.x) * cos(val.y), exp(val.x) * sin(val.y));
}
static __device__ inline acComplex operator*(const AcReal& a, const acComplex& b)
{
    return (acComplex){a * b.x, a * b.y};
}

static __device__ inline acComplex operator*(const acComplex& b, const AcReal& a)
{
    return (acComplex){a * b.x, a * b.y};
}

static __device__ inline acComplex operator*(const acComplex& a, const acComplex& b)
{
    return (acComplex){a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}
//#include <complex>
