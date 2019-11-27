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
#include "src/core/math_utils.h"

#include <assert.h>

static_assert(NUM_VTXBUF_HANDLES > 0, "ERROR: At least one uniform ScalarField must be declared.");

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

#define make_int3(a, b, c)                                                                         \
    (int3) { (int)a, (int)b, (int)c }

template <int step_number>
static __device__ __forceinline__ AcReal
rk3_integrate(const AcReal state_previous, const AcReal state_current, const AcReal rate_of_change,
              const AcReal dt)
{
    // Williamson (1980)
    const AcReal alpha[] = {0, AcReal(.0), AcReal(-5. / 9.), AcReal(-153. / 128.)};
    const AcReal beta[]  = {0, AcReal(1. / 3.), AcReal(15. / 16.), AcReal(8. / 15.)};

    // Note the indexing: +1 to avoid an unnecessary warning about "out-of-bounds"
    // access (when accessing beta[step_number-1] even when step_number >= 1)
    switch (step_number) {
    case 0:
        return state_current + beta[step_number + 1] * rate_of_change * dt;
    case 1: // Fallthrough
    case 2:
        return state_current +
               beta[step_number + 1] * (alpha[step_number + 1] * (AcReal(1.) / beta[step_number]) *
                                            (state_current - state_previous) +
                                        rate_of_change * dt);
    default:
        return NAN;
    }
}

template <int step_number>
static __device__ __forceinline__ AcReal3
rk3_integrate(const AcReal3 state_previous, const AcReal3 state_current,
              const AcReal3 rate_of_change, const AcReal dt)
{
    return (AcReal3){
        rk3_integrate<step_number>(state_previous.x, state_current.x, rate_of_change.x, dt),
        rk3_integrate<step_number>(state_previous.y, state_current.y, rate_of_change.y, dt),
        rk3_integrate<step_number>(state_previous.z, state_current.z, rate_of_change.z, dt)};
}

#define rk3(state_previous, state_current, rate_of_change, dt)                                     \
    rk3_integrate<step_number>(state_previous, value(state_current), rate_of_change, dt)

static __device__ void
write(AcReal* __restrict__ out[], const int handle, const int idx, const AcReal value)
{
    out[handle][idx] = value;
}

static __device__ __forceinline__ void
write(AcReal* __restrict__ out[], const int3 vec, const int idx, const AcReal3 value)
{
    write(out, vec.x, idx, value.x);
    write(out, vec.y, idx, value.y);
    write(out, vec.z, idx, value.z);
}

static __device__ __forceinline__ AcReal
read_out(const int idx, AcReal* __restrict__ field[], const int handle)
{
    return field[handle][idx];
}

static __device__ __forceinline__ AcReal3
read_out(const int idx, AcReal* __restrict__ field[], const int3 handle)
{
    return (AcReal3){read_out(idx, field, handle.x), read_out(idx, field, handle.y),
                     read_out(idx, field, handle.z)};
}

#define WRITE_OUT(handle, value) (write(buffer.out, handle, idx, value))
#define READ(handle) (read_data(vertexIdx, globalVertexIdx, buffer.in, handle))
#define READ_OUT(handle) (read_out(idx, buffer.out, handle))

#define GEN_PREPROCESSED_PARAM_BOILERPLATE const int3 &vertexIdx, const int3 &globalVertexIdx
#define GEN_KERNEL_PARAM_BOILERPLATE const int3 start, const int3 end, VertexBufferArray buffer

#define GEN_KERNEL_BUILTIN_VARIABLES_BOILERPLATE()                                                 \
    const int3 vertexIdx       = (int3){threadIdx.x + blockIdx.x * blockDim.x + start.x,           \
                                  threadIdx.y + blockIdx.y * blockDim.y + start.y,           \
                                  threadIdx.z + blockIdx.z * blockDim.z + start.z};          \
    const int3 globalVertexIdx = (int3){d_multigpu_offset.x + vertexIdx.x,                         \
                                        d_multigpu_offset.y + vertexIdx.y,                         \
                                        d_multigpu_offset.z + vertexIdx.z};                        \
    (void)globalVertexIdx;                                                                         \
    if (vertexIdx.x >= end.x || vertexIdx.y >= end.y || vertexIdx.z >= end.z)                      \
        return;                                                                                    \
                                                                                                   \
    assert(vertexIdx.x < DCONST(AC_nx_max) && vertexIdx.y < DCONST(AC_ny_max) &&           \
           vertexIdx.z < DCONST(AC_nz_max));                                                   \
                                                                                                   \
    assert(vertexIdx.x >= DCONST(AC_nx_min) && vertexIdx.y >= DCONST(AC_ny_min) &&         \
           vertexIdx.z >= DCONST(AC_nz_min));                                                  \
                                                                                                   \
    const int idx = IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z);

// clang-format off
#define GEN_DEVICE_FUNC_HOOK(identifier)                                                           \
    template <int step_number>                                                                     \
    AcResult acDeviceKernel_##identifier(const Device device, const Stream stream,                 \
                                         const int3 start, const int3 end)                         \
    {                                                                                              \
        cudaSetDevice(device->id);                                                                 \
                                                                                                   \
        const dim3 tpb(32, 1, 4);                                                                  \
                                                                                                   \
        const int3 n = end - start;                                                                \
        const dim3 bpg((unsigned int)ceil(n.x / AcReal(tpb.x)),                                    \
                       (unsigned int)ceil(n.y / AcReal(tpb.y)),                                    \
                       (unsigned int)ceil(n.z / AcReal(tpb.z)));                                   \
                                                                                                   \
        identifier<step_number>                                                                    \
            <<<bpg, tpb, 0, device->streams[stream]>>>(start, end, device->vba);                   \
        ERRCHK_CUDA_KERNEL();                                                                      \
                                                                                                   \
        return AC_SUCCESS;                                                                         \
    }


#include "user_kernels.h"
