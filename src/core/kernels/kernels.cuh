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

__global__ void
kernel_periodic_boundconds(const int3 start, const int3 end, AcReal* vtxbuf)
{
    const int i_dst = start.x + threadIdx.x + blockIdx.x * blockDim.x;
    const int j_dst = start.y + threadIdx.y + blockIdx.y * blockDim.y;
    const int k_dst = start.z + threadIdx.z + blockIdx.z * blockDim.z;

    // If within the start-end range (this allows threadblock dims that are not
    // divisible by end - start)
    if (i_dst >= end.x || j_dst >= end.y || k_dst >= end.z)
        return;

    // if (i_dst >= DCONST_INT(AC_mx) || j_dst >= DCONST_INT(AC_my) || k_dst >= DCONST_INT(AC_mz))
    //    return;

    // If destination index is inside the computational domain, return since
    // the boundary conditions are only applied to the ghost zones
    if (i_dst >= DCONST_INT(AC_nx_min) && i_dst < DCONST_INT(AC_nx_max) &&
        j_dst >= DCONST_INT(AC_ny_min) && j_dst < DCONST_INT(AC_ny_max) &&
        k_dst >= DCONST_INT(AC_nz_min) && k_dst < DCONST_INT(AC_nz_max))
        return;

    // Find the source index
    // Map to nx, ny, nz coordinates
    int i_src = i_dst - DCONST_INT(AC_nx_min);
    int j_src = j_dst - DCONST_INT(AC_ny_min);
    int k_src = k_dst - DCONST_INT(AC_nz_min);

    // Translate (s.t. the index is always positive)
    i_src += DCONST_INT(AC_nx);
    j_src += DCONST_INT(AC_ny);
    k_src += DCONST_INT(AC_nz);

    // Wrap
    i_src %= DCONST_INT(AC_nx);
    j_src %= DCONST_INT(AC_ny);
    k_src %= DCONST_INT(AC_nz);

    // Map to mx, my, mz coordinates
    i_src += DCONST_INT(AC_nx_min);
    j_src += DCONST_INT(AC_ny_min);
    k_src += DCONST_INT(AC_nz_min);

    const int src_idx = DEVICE_VTXBUF_IDX(i_src, j_src, k_src);
    const int dst_idx = DEVICE_VTXBUF_IDX(i_dst, j_dst, k_dst);
    vtxbuf[dst_idx]   = vtxbuf[src_idx];
}

void
periodic_boundconds(const cudaStream_t stream, const int3& start, const int3& end, AcReal* vtxbuf)
{
    const dim3 tpb(8, 2, 8);
    const dim3 bpg((unsigned int)ceil((end.x - start.x) / (float)tpb.x),
                   (unsigned int)ceil((end.y - start.y) / (float)tpb.y),
                   (unsigned int)ceil((end.z - start.z) / (float)tpb.z));

    kernel_periodic_boundconds<<<bpg, tpb, 0, stream>>>(start, end, vtxbuf);
    ERRCHK_CUDA_KERNEL();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <assert.h>

static __device__ __forceinline__ int
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

static __forceinline__ AcMatrix
create_rotz(const AcReal radians)
{
    AcMatrix mat;

    mat.row[0] = (AcReal3){cos(radians), -sin(radians), 0};
    mat.row[1] = (AcReal3){sin(radians), cos(radians), 0};
    mat.row[2] = (AcReal3){0, 0, 0};

    return mat;
}

#if AC_DOUBLE_PRECISION == 0
#define sin __sinf
#define cos __cosf
#define exp __expf
#define rsqrt rsqrtf // hardware reciprocal sqrt
#endif               // AC_DOUBLE_PRECISION == 0

/*
typedef struct {
    int i, j, k;
} int3;*/

/*
 * =============================================================================
 * Level 0 (Input Assembly Stage)
 * =============================================================================
 */

/*
 * =============================================================================
 * Level 0.1 (Read stencil elements and solve derivatives)
 * =============================================================================
 */
static __device__ __forceinline__ AcReal
first_derivative(const AcReal* __restrict__ pencil, const AcReal inv_ds)
{
#if STENCIL_ORDER == 2
    const AcReal coefficients[] = {0, 1.0 / 2.0};
#elif STENCIL_ORDER == 4
    const AcReal coefficients[] = {0, 2.0 / 3.0, -1.0 / 12.0};
#elif STENCIL_ORDER == 6
    const AcReal coefficients[] = {0, 3.0 / 4.0, -3.0 / 20.0, 1.0 / 60.0};
#elif STENCIL_ORDER == 8
    const AcReal coefficients[] = {0, 4.0 / 5.0, -1.0 / 5.0, 4.0 / 105.0, -1.0 / 280.0};
#endif

#define MID (STENCIL_ORDER / 2)
    AcReal res = 0;

#pragma unroll
    for (int i = 1; i <= MID; ++i)
        res += coefficients[i] * (pencil[MID + i] - pencil[MID - i]);

    return res * inv_ds;
}

static __device__ __forceinline__ AcReal
second_derivative(const AcReal* __restrict__ pencil, const AcReal inv_ds)
{
#if STENCIL_ORDER == 2
    const AcReal coefficients[] = {-2., 1.};
#elif STENCIL_ORDER == 4
    const AcReal coefficients[] = {-5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0};
#elif STENCIL_ORDER == 6
    const AcReal coefficients[] = {-49.0 / 18.0, 3.0 / 2.0, -3.0 / 20.0, 1.0 / 90.0};
#elif STENCIL_ORDER == 8
    const AcReal coefficients[] = {-205.0 / 72.0, 8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0};
#endif

#define MID (STENCIL_ORDER / 2)
    AcReal res = coefficients[0] * pencil[MID];

#pragma unroll
    for (int i = 1; i <= MID; ++i)
        res += coefficients[i] * (pencil[MID + i] + pencil[MID - i]);

    return res * inv_ds * inv_ds;
}

/** inv_ds: inverted mesh spacing f.ex. 1. / mesh.int_params[AC_dsx] */
static __device__ __forceinline__ AcReal
cross_derivative(const AcReal* __restrict__ pencil_a, const AcReal* __restrict__ pencil_b,
                 const AcReal inv_ds_a, const AcReal inv_ds_b)
{
#if STENCIL_ORDER == 2
    const AcReal coefficients[] = {0, 1.0 / 4.0};
#elif STENCIL_ORDER == 4
    const AcReal coefficients[] = {
        0, 1.0 / 32.0, 1.0 / 64.0}; // TODO correct coefficients, these are just placeholders
#elif STENCIL_ORDER == 6
    const AcReal fac            = (1. / 720.);
    const AcReal coefficients[] = {0.0 * fac, 270.0 * fac, -27.0 * fac, 2.0 * fac};
#elif STENCIL_ORDER == 8
    const AcReal fac            = (1. / 20160.);
    const AcReal coefficients[] = {0.0 * fac, 8064. * fac, -1008. * fac, 128. * fac, -9. * fac};
#endif

#define MID (STENCIL_ORDER / 2)
    AcReal res = AcReal(0.);

#pragma unroll
    for (int i = 1; i <= MID; ++i) {
        res += coefficients[i] *
               (pencil_a[MID + i] + pencil_a[MID - i] - pencil_b[MID + i] - pencil_b[MID - i]);
    }
    return res * inv_ds_a * inv_ds_b;
}

static __device__ __forceinline__ AcReal
derx(const int3 vertexIdx, const AcReal* __restrict__ arr)
{
    AcReal pencil[STENCIL_ORDER + 1];
#pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil[offset] = arr[IDX(vertexIdx.x + offset - STENCIL_ORDER / 2, vertexIdx.y,
                                 vertexIdx.z)];

    return first_derivative(pencil, DCONST_REAL(AC_inv_dsx));
}

static __device__ __forceinline__ AcReal
derxx(const int3 vertexIdx, const AcReal* __restrict__ arr)
{
    AcReal pencil[STENCIL_ORDER + 1];
#pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil[offset] = arr[IDX(vertexIdx.x + offset - STENCIL_ORDER / 2, vertexIdx.y,
                                 vertexIdx.z)];

    return second_derivative(pencil, DCONST_REAL(AC_inv_dsx));
}

static __device__ __forceinline__ AcReal
derxy(const int3 vertexIdx, const AcReal* __restrict__ arr)
{
    AcReal pencil_a[STENCIL_ORDER + 1];
#pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil_a[offset] = arr[IDX(vertexIdx.x + offset - STENCIL_ORDER / 2,
                                   vertexIdx.y + offset - STENCIL_ORDER / 2, vertexIdx.z)];

    AcReal pencil_b[STENCIL_ORDER + 1];
#pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil_b[offset] = arr[IDX(vertexIdx.x + offset - STENCIL_ORDER / 2,
                                   vertexIdx.y + STENCIL_ORDER / 2 - offset, vertexIdx.z)];

    return cross_derivative(pencil_a, pencil_b, DCONST_REAL(AC_inv_dsx), DCONST_REAL(AC_inv_dsy));
}

static __device__ __forceinline__ AcReal
derxz(const int3 vertexIdx, const AcReal* __restrict__ arr)
{
    AcReal pencil_a[STENCIL_ORDER + 1];
#pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil_a[offset] = arr[IDX(vertexIdx.x + offset - STENCIL_ORDER / 2, vertexIdx.y,
                                   vertexIdx.z + offset - STENCIL_ORDER / 2)];

    AcReal pencil_b[STENCIL_ORDER + 1];
#pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil_b[offset] = arr[IDX(vertexIdx.x + offset - STENCIL_ORDER / 2, vertexIdx.y,
                                   vertexIdx.z + STENCIL_ORDER / 2 - offset)];

    return cross_derivative(pencil_a, pencil_b, DCONST_REAL(AC_inv_dsx), DCONST_REAL(AC_inv_dsz));
}

static __device__ __forceinline__ AcReal
dery(const int3 vertexIdx, const AcReal* __restrict__ arr)
{
    AcReal pencil[STENCIL_ORDER + 1];
#pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil[offset] = arr[IDX(vertexIdx.x, vertexIdx.y + offset - STENCIL_ORDER / 2,
                                 vertexIdx.z)];

    return first_derivative(pencil, DCONST_REAL(AC_inv_dsy));
}

static __device__ __forceinline__ AcReal
deryy(const int3 vertexIdx, const AcReal* __restrict__ arr)
{
    AcReal pencil[STENCIL_ORDER + 1];
#pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil[offset] = arr[IDX(vertexIdx.x, vertexIdx.y + offset - STENCIL_ORDER / 2,
                                 vertexIdx.z)];

    return second_derivative(pencil, DCONST_REAL(AC_inv_dsy));
}

static __device__ __forceinline__ AcReal
deryz(const int3 vertexIdx, const AcReal* __restrict__ arr)
{
    AcReal pencil_a[STENCIL_ORDER + 1];
#pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil_a[offset] = arr[IDX(vertexIdx.x, vertexIdx.y + offset - STENCIL_ORDER / 2,
                                   vertexIdx.z + offset - STENCIL_ORDER / 2)];

    AcReal pencil_b[STENCIL_ORDER + 1];
#pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil_b[offset] = arr[IDX(vertexIdx.x, vertexIdx.y + offset - STENCIL_ORDER / 2,
                                   vertexIdx.z + STENCIL_ORDER / 2 - offset)];

    return cross_derivative(pencil_a, pencil_b, DCONST_REAL(AC_inv_dsy), DCONST_REAL(AC_inv_dsz));
}

static __device__ __forceinline__ AcReal
derz(const int3 vertexIdx, const AcReal* __restrict__ arr)
{
    AcReal pencil[STENCIL_ORDER + 1];
#pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil[offset] = arr[IDX(vertexIdx.x, vertexIdx.y,
                                 vertexIdx.z + offset - STENCIL_ORDER / 2)];

    return first_derivative(pencil, DCONST_REAL(AC_inv_dsz));
}

static __device__ __forceinline__ AcReal
derzz(const int3 vertexIdx, const AcReal* __restrict__ arr)
{
    AcReal pencil[STENCIL_ORDER + 1];
#pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil[offset] = arr[IDX(vertexIdx.x, vertexIdx.y,
                                 vertexIdx.z + offset - STENCIL_ORDER / 2)];

    return second_derivative(pencil, DCONST_REAL(AC_inv_dsz));
}

/*
 * =============================================================================
 * Level 0.2 (Caching functions)
 * =============================================================================
 */

#include "stencil_assembly.cuh"

/*
typedef struct {
    AcRealData x;
    AcRealData y;
    AcRealData z;
} AcReal3Data;

static __device__ __forceinline__ AcReal3Data
read_data(const int i, const int j, const int k,
          AcReal* __restrict__ buf[], const int3& handle)
{
    AcReal3Data data;

    data.x = read_data(i, j, k, buf, handle.x);
    data.y = read_data(i, j, k, buf, handle.y);
    data.z = read_data(i, j, k, buf, handle.z);

    return data;
}
*/

/*
 * =============================================================================
 * Level 0.3 (Built-in functions available during the Stencil Processing Stage)
 * =============================================================================
 */

static __host__ __device__ __forceinline__ AcReal3
operator-(const AcReal3& a, const AcReal3& b)
{
    return (AcReal3){a.x - b.x, a.y - b.y, a.z - b.z};
}

static __host__ __device__ __forceinline__ AcReal3
operator+(const AcReal3& a, const AcReal3& b)
{
    return (AcReal3){a.x + b.x, a.y + b.y, a.z + b.z};
}

static __host__ __device__ __forceinline__ AcReal3
operator-(const AcReal3& a)
{
    return (AcReal3){-a.x, -a.y, -a.z};
}

static __host__ __device__ __forceinline__ AcReal3 operator*(const AcReal a, const AcReal3& b)
{
    return (AcReal3){a * b.x, a * b.y, a * b.z};
}

static __host__ __device__ __forceinline__ AcReal
dot(const AcReal3& a, const AcReal3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static __host__ __device__ __forceinline__ AcReal3
mul(const AcMatrix& aa, const AcReal3& x)
{
    return (AcReal3){dot(aa.row[0], x), dot(aa.row[1], x), dot(aa.row[2], x)};
}

static __host__ __device__ __forceinline__ AcReal3
cross(const AcReal3& a, const AcReal3& b)
{
    AcReal3 c;

    c.x = a.y * b.z - a.z * b.y;
    c.y = a.z * b.x - a.x * b.z;
    c.z = a.x * b.y - a.y * b.x;

    return c;
}

static __host__ __device__ __forceinline__ bool
is_valid(const AcReal& a)
{
    return !isnan(a) && !isinf(a);
}

static __host__ __device__ __forceinline__ bool
is_valid(const AcReal3& a)
{
    return is_valid(a.x) && is_valid(a.y) && is_valid(a.z);
}

/*
 * =============================================================================
 * Level 1 (Stencil Processing Stage)
 * =============================================================================
 */

/*
 * =============================================================================
 * Level 1.1 (Terms)
 * =============================================================================
 */
static __device__ __forceinline__ AcReal
laplace(const AcRealData& data)
{
    return hessian(data).row[0].x + hessian(data).row[1].y + hessian(data).row[2].z;
}

static __device__ __forceinline__ AcReal
divergence(const AcReal3Data& vec)
{
    return gradient(vec.x).x + gradient(vec.y).y + gradient(vec.z).z;
}

static __device__ __forceinline__ AcReal3
laplace_vec(const AcReal3Data& vec)
{
    return (AcReal3){laplace(vec.x), laplace(vec.y), laplace(vec.z)};
}

static __device__ __forceinline__ AcReal3
curl(const AcReal3Data& vec)
{
    return (AcReal3){gradient(vec.z).y - gradient(vec.y).z, gradient(vec.x).z - gradient(vec.z).x,
                     gradient(vec.y).x - gradient(vec.x).y};
}

static __device__ __forceinline__ AcReal3
gradient_of_divergence(const AcReal3Data& vec)
{
    return (AcReal3){hessian(vec.x).row[0].x + hessian(vec.y).row[0].y + hessian(vec.z).row[0].z,
                     hessian(vec.x).row[1].x + hessian(vec.y).row[1].y + hessian(vec.z).row[1].z,
                     hessian(vec.x).row[2].x + hessian(vec.y).row[2].y + hessian(vec.z).row[2].z};
}

// Takes uu gradients and returns S
static __device__ __forceinline__ AcMatrix
stress_tensor(const AcReal3Data& vec)
{
    AcMatrix S;

    S.row[0].x = AcReal(2. / 3.) * gradient(vec.x).x -
                 AcReal(1. / 3.) * (gradient(vec.y).y + gradient(vec.z).z);
    S.row[0].y = AcReal(1. / 2.) * (gradient(vec.x).y + gradient(vec.y).x);
    S.row[0].z = AcReal(1. / 2.) * (gradient(vec.x).z + gradient(vec.z).x);

    S.row[1].y = AcReal(2. / 3.) * gradient(vec.y).y -
                 AcReal(1. / 3.) * (gradient(vec.x).x + gradient(vec.z).z);

    S.row[1].z = AcReal(1. / 2.) * (gradient(vec.y).z + gradient(vec.z).y);

    S.row[2].z = AcReal(2. / 3.) * gradient(vec.z).z -
                 AcReal(1. / 3.) * (gradient(vec.x).x + gradient(vec.y).y);

    S.row[1].x = S.row[0].y;
    S.row[2].x = S.row[0].z;
    S.row[2].y = S.row[1].z;

    return S;
}

static __device__ __forceinline__ AcReal
contract(const AcMatrix& mat)
{
    AcReal res = 0;

#pragma unroll
    for (int i = 0; i < 3; ++i)
        res += dot(mat.row[i], mat.row[i]);

    return res;
}

/*
 * =============================================================================
 * Level 1.2 (Equations)
 * =============================================================================
 */
static __device__ __forceinline__ AcReal
length(const AcReal3& vec)
{
    return sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

static __device__ __forceinline__ AcReal
reciprocal_len(const AcReal3& vec)
{
    return rsqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

static __device__ __forceinline__ AcReal3
normalized(const AcReal3& vec)
{
    const AcReal inv_len = reciprocal_len(vec);
    return inv_len * vec;
}

#define H_CONST (AcReal(0.0))
#define C_CONST (AcReal(0.0))

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
/*
template <int step_number>
static __device__ __forceinline__ AcReal
rk3_integrate_scal(const AcReal state_previous, const AcReal state_current,
              const AcReal rate_of_change, const AcReal dt)
{
    // Williamson (1980)
    const AcReal alpha[] = {AcReal(.0), AcReal(-5. / 9.), AcReal(-153. / 128.)};
    const AcReal beta[]  = {AcReal(1. / 3.), AcReal(15. / 16.),
                           AcReal(8. / 15.)};


    switch (step_number) {
        case 0:
            return state_current + beta[step_number] * rate_of_change * dt;
        case 1: // Fallthrough
        case 2:
            return state_current +
               beta[step_number] *
                   (alpha[step_number] * (AcReal(1.) / beta[step_number - 1]) *
                        (state_current - state_previous) +
                    rate_of_change * dt);
        default:
            return NAN;
    }
}
*/

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

/*
template <int step_number>
static __device__ __forceinline__ AcReal
rk3_integrate(const int idx, const AcReal out, const int handle,
              const AcRealData& in_cached, const AcReal rate_of_change, const AcReal dt)
{
    return rk3_integrate_scal<step_number>(out, value(in_cached), rate_of_change, dt);
}

template <int step_number>
static __device__ __forceinline__ AcReal3
rk3_integrate(const int idx, const AcReal3 out, const int3& handle,
                  const AcReal3Data& in_cached, const AcReal3& rate_of_change, const AcReal dt)
{
    return (AcReal3) {
        rk3_integrate<step_number>(idx, out, handle.x, in_cached.x, rate_of_change.x, dt),
        rk3_integrate<step_number>(idx, out, handle.y, in_cached.y, rate_of_change.y, dt),
        rk3_integrate<step_number>(idx, out, handle.z, in_cached.z, rate_of_change.z, dt)
    };
}

#define RK3(handle, in_cached, rate_of_change, dt) \
rk3_integrate<step_number>(idx, buffer.out, handle, in_cached, rate_of_change, dt)
*/

/*
 * =============================================================================
 * Level 1.3 (Kernels)
 * =============================================================================
 */

static __device__ void
write(AcReal* __restrict__ out[], const int handle, const int idx, const AcReal value)
{
    out[handle][idx] = value;
}

static __device__ void
write(AcReal* __restrict__ out[], const int3 vec, const int idx, const AcReal3 value)
{
    write(out, vec.x, idx, value.x);
    write(out, vec.y, idx, value.y);
    write(out, vec.z, idx, value.z);
}

static __device__ AcReal
read_out(const int idx, AcReal* __restrict__ field[], const int handle)
{
    return field[handle][idx];
}

static __device__ AcReal3
read_out(const int idx, AcReal* __restrict__ field[], const int3 handle)
{
    return (AcReal3){read_out(idx, field, handle.x), read_out(idx, field, handle.y),
                     read_out(idx, field, handle.z)};
}

#define WRITE_OUT(handle, value) (write(buffer.out, handle, idx, value))
#define READ(handle) (read_data(vertexIdx, globalVertexIdx, buffer.in, handle))
#define READ_OUT(handle) (read_out(idx, buffer.out, handle))

// also write for clarity here also, not for the DSL
//#define WRITE(HANDLE) (write(idx, ...)) s.t. we don't have to insert boilerplat in the mid of the
// function

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
    assert(vertexIdx.x < DCONST_INT(AC_nx_max) && vertexIdx.y < DCONST_INT(AC_ny_max) &&           \
           vertexIdx.z < DCONST_INT(AC_nz_max));                                                   \
                                                                                                   \
    assert(vertexIdx.x >= DCONST_INT(AC_nx_min) && vertexIdx.y >= DCONST_INT(AC_ny_min) &&         \
           vertexIdx.z >= DCONST_INT(AC_nz_min));                                                  \
                                                                                                   \
    const int idx = IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z);

#include "stencil_process.cuh"

/*
 * =============================================================================
 * Level 2 (Host calls)
 * =============================================================================
 */

////////////////REDUCE///////////////////////////
#include "src/core/math_utils.h" // is_power_of_two

/*
Reduction steps:
 1 of 3: Compute the initial value (a, a*a or exp(a)*exp(a)) and put the result in scratchpad
 2 of 3: Compute most of the reductions into a single block of data
 3 of 3: After all results have been stored into the final block, reduce the data in the final block
*/

// Function pointer definitions
typedef AcReal (*FilterFunc)(const AcReal&);
typedef AcReal (*FilterFuncVec)(const AcReal&, const AcReal&, const AcReal&);
typedef AcReal (*ReduceFunc)(const AcReal&, const AcReal&);

// clang-format off
/* Comparison funcs */
static __device__ inline AcReal
dmax(const AcReal& a, const AcReal& b) { return a > b ? a : b; }

static __device__ inline AcReal
dmin(const AcReal& a, const AcReal& b) { return a < b ? a : b; }

static __device__ inline AcReal
dsum(const AcReal& a, const AcReal& b) { return a + b; }

/* Function used to determine the values used during reduction */
static __device__ inline AcReal
dvalue(const AcReal& a) { return AcReal(a); }

static __device__ inline AcReal
dsquared(const AcReal& a) { return (AcReal)(a*a); }

static __device__ inline AcReal
dexp_squared(const AcReal& a) { return exp(a)*exp(a); }

static __device__ inline AcReal
dlength_vec(const AcReal& a, const AcReal& b, const AcReal& c) { return sqrt(a*a + b*b + c*c); }

static __device__ inline AcReal
dsquared_vec(const AcReal& a, const AcReal& b, const AcReal& c) { return dsquared(a) + dsquared(b) + dsquared(c); }

static __device__ inline AcReal
dexp_squared_vec(const AcReal& a, const AcReal& b, const AcReal& c) { return dexp_squared(a) + dexp_squared(b) + dexp_squared(c); }
// clang-format on

#include <assert.h>
template <FilterFunc filter>
__global__ void
kernel_filter(const __restrict__ AcReal* src, const int3 start, const int3 end, AcReal* dst)
{
    const int3 src_idx = (int3){start.x + threadIdx.x + blockIdx.x * blockDim.x,
                                start.y + threadIdx.y + blockIdx.y * blockDim.y,
                                start.z + threadIdx.z + blockIdx.z * blockDim.z};

    const int nx = end.x - start.x;
    const int ny = end.y - start.y;
    const int nz = end.z - start.z;
    (void)nz; // Suppressed unused variable warning when not compiling with debug flags

    const int3 dst_idx = (int3){threadIdx.x + blockIdx.x * blockDim.x,
                                threadIdx.y + blockIdx.y * blockDim.y,
                                threadIdx.z + blockIdx.z * blockDim.z};

    assert(src_idx.x < DCONST_INT(AC_nx_max) && src_idx.y < DCONST_INT(AC_ny_max) &&
           src_idx.z < DCONST_INT(AC_nz_max));
    assert(dst_idx.x < nx && dst_idx.y < ny && dst_idx.z < nz);
    assert(dst_idx.x + dst_idx.y * nx + dst_idx.z * nx * ny < nx * ny * nz);

    dst[dst_idx.x + dst_idx.y * nx + dst_idx.z * nx * ny] = filter(src[IDX(src_idx)]);
}

template <FilterFuncVec filter>
__global__ void
kernel_filter_vec(const __restrict__ AcReal* src0, const __restrict__ AcReal* src1,
                  const __restrict__ AcReal* src2, const int3 start, const int3 end, AcReal* dst)
{
    const int3 src_idx = (int3){start.x + threadIdx.x + blockIdx.x * blockDim.x,
                                start.y + threadIdx.y + blockIdx.y * blockDim.y,
                                start.z + threadIdx.z + blockIdx.z * blockDim.z};

    const int nx = end.x - start.x;
    const int ny = end.y - start.y;
    const int nz = end.z - start.z;
    (void)nz; // Suppressed unused variable warning when not compiling with debug flags

    const int3 dst_idx = (int3){threadIdx.x + blockIdx.x * blockDim.x,
                                threadIdx.y + blockIdx.y * blockDim.y,
                                threadIdx.z + blockIdx.z * blockDim.z};

    assert(src_idx.x < DCONST_INT(AC_nx_max) && src_idx.y < DCONST_INT(AC_ny_max) &&
           src_idx.z < DCONST_INT(AC_nz_max));
    assert(dst_idx.x < nx && dst_idx.y < ny && dst_idx.z < nz);
    assert(dst_idx.x + dst_idx.y * nx + dst_idx.z * nx * ny < nx * ny * nz);

    dst[dst_idx.x + dst_idx.y * nx + dst_idx.z * nx * ny] = filter(
        src0[IDX(src_idx)], src1[IDX(src_idx)], src2[IDX(src_idx)]);
}

template <ReduceFunc reduce>
__global__ void
kernel_reduce(AcReal* scratchpad, const int num_elems)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ AcReal smem[];
    if (idx < num_elems) {
        smem[threadIdx.x] = scratchpad[idx];
    }
    else {
        smem[threadIdx.x] = NAN;
    }
    __syncthreads();

    int offset = blockDim.x / 2;
    assert(offset % 2 == 0);
    while (offset > 0) {
        if (threadIdx.x < offset) {
            smem[threadIdx.x] = reduce(smem[threadIdx.x], smem[threadIdx.x + offset]);
        }
        offset /= 2;
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        scratchpad[idx] = smem[threadIdx.x];
    }
}

template <ReduceFunc reduce>
__global__ void
kernel_reduce_block(const __restrict__ AcReal* scratchpad, const int num_blocks,
                    const int block_size, AcReal* result)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx != 0) {
        return;
    }

    AcReal res = scratchpad[0];
    for (int i = 1; i < num_blocks; ++i) {
        res = reduce(res, scratchpad[i * block_size]);
    }
    *result = res;
}

AcReal
reduce_scal(const cudaStream_t stream, const ReductionType rtype, const int3& start,
            const int3& end, const AcReal* vtxbuf, AcReal* scratchpad, AcReal* reduce_result)
{
    const unsigned nx        = end.x - start.x;
    const unsigned ny        = end.y - start.y;
    const unsigned nz        = end.z - start.z;
    const unsigned num_elems = nx * ny * nz;

    const dim3 tpb_filter(32, 4, 1);
    const dim3 bpg_filter((unsigned int)ceil(nx / AcReal(tpb_filter.x)),
                          (unsigned int)ceil(ny / AcReal(tpb_filter.y)),
                          (unsigned int)ceil(nz / AcReal(tpb_filter.z)));

    const int tpb_reduce = 128;
    const int bpg_reduce = num_elems / tpb_reduce;

    ERRCHK(nx >= tpb_filter.x);
    ERRCHK(ny >= tpb_filter.y);
    ERRCHK(nz >= tpb_filter.z);
    ERRCHK(tpb_reduce <= num_elems);
    ERRCHK(nx * ny * nz % 2 == 0);

    // clang-format off
    if (rtype == RTYPE_MAX) {
        kernel_filter<dvalue><<<bpg_filter, tpb_filter, 0, stream>>>(vtxbuf, start, end, scratchpad);
        kernel_reduce<dmax><<<bpg_reduce, tpb_reduce, sizeof(AcReal) * tpb_reduce, stream>>>(scratchpad, num_elems);
        kernel_reduce_block<dmax><<<1, 1, 0, stream>>>(scratchpad, bpg_reduce, tpb_reduce, reduce_result);
    } else if (rtype == RTYPE_MIN) {
        kernel_filter<dvalue><<<bpg_filter, tpb_filter, 0, stream>>>(vtxbuf, start, end, scratchpad);
        kernel_reduce<dmin><<<bpg_reduce, tpb_reduce, sizeof(AcReal) * tpb_reduce, stream>>>(scratchpad, num_elems);
        kernel_reduce_block<dmin><<<1, 1, 0, stream>>>(scratchpad, bpg_reduce, tpb_reduce, reduce_result);
    } else if (rtype == RTYPE_RMS) {
        kernel_filter<dsquared><<<bpg_filter, tpb_filter, 0, stream>>>(vtxbuf, start, end, scratchpad);
        kernel_reduce<dsum><<<bpg_reduce, tpb_reduce, sizeof(AcReal) * tpb_reduce, stream>>>(scratchpad, num_elems);
        kernel_reduce_block<dsum><<<1, 1, 0, stream>>>(scratchpad, bpg_reduce, tpb_reduce, reduce_result);
    } else if (rtype == RTYPE_RMS_EXP) {
        kernel_filter<dexp_squared><<<bpg_filter, tpb_filter, 0, stream>>>(vtxbuf, start, end, scratchpad);
        kernel_reduce<dsum><<<bpg_reduce, tpb_reduce, sizeof(AcReal) * tpb_reduce, stream>>>(scratchpad, num_elems);
        kernel_reduce_block<dsum><<<1, 1, 0, stream>>>(scratchpad, bpg_reduce, tpb_reduce, reduce_result);
    } else if (rtype == RTYPE_SUM) {
        kernel_filter<dvalue><<<bpg_filter, tpb_filter, 0, stream>>>(vtxbuf, start, end, scratchpad);
        kernel_reduce<dsum><<<bpg_reduce, tpb_reduce, sizeof(AcReal) * tpb_reduce, stream>>>(scratchpad, num_elems);
        kernel_reduce_block<dsum><<<1, 1, 0, stream>>>(scratchpad, bpg_reduce, tpb_reduce, reduce_result);
    } else {
        ERROR("Unrecognized rtype");
    }
    // clang-format on
    cudaStreamSynchronize(stream);
    AcReal result;
    cudaMemcpy(&result, reduce_result, sizeof(AcReal), cudaMemcpyDeviceToHost);
    return result;
}

AcReal
reduce_vec(const cudaStream_t stream, const ReductionType rtype, const int3& start, const int3& end,
           const AcReal* vtxbuf0, const AcReal* vtxbuf1, const AcReal* vtxbuf2, AcReal* scratchpad,
           AcReal* reduce_result)
{
    const unsigned nx        = end.x - start.x;
    const unsigned ny        = end.y - start.y;
    const unsigned nz        = end.z - start.z;
    const unsigned num_elems = nx * ny * nz;

    const dim3 tpb_filter(32, 4, 1);
    const dim3 bpg_filter((unsigned int)ceil(nx / AcReal(tpb_filter.x)),
                          (unsigned int)ceil(ny / AcReal(tpb_filter.y)),
                          (unsigned int)ceil(nz / AcReal(tpb_filter.z)));

    const int tpb_reduce = 128;
    const int bpg_reduce = num_elems / tpb_reduce;

    ERRCHK(nx >= tpb_filter.x);
    ERRCHK(ny >= tpb_filter.y);
    ERRCHK(nz >= tpb_filter.z);
    ERRCHK(tpb_reduce <= num_elems);
    ERRCHK(nx * ny * nz % 2 == 0);

    // clang-format off
    if (rtype == RTYPE_MAX) {
        kernel_filter_vec<dlength_vec><<<bpg_filter, tpb_filter, 0, stream>>>(vtxbuf0, vtxbuf1, vtxbuf2, start, end, scratchpad);
        kernel_reduce<dmax><<<bpg_reduce, tpb_reduce, sizeof(AcReal) * tpb_reduce, stream>>>(scratchpad, num_elems);
        kernel_reduce_block<dmax><<<1, 1, 0, stream>>>(scratchpad, bpg_reduce, tpb_reduce, reduce_result);
    } else if (rtype == RTYPE_MIN) {
        kernel_filter_vec<dlength_vec><<<bpg_filter, tpb_filter, 0, stream>>>(vtxbuf0, vtxbuf1, vtxbuf2, start, end, scratchpad);
        kernel_reduce<dmin><<<bpg_reduce, tpb_reduce, sizeof(AcReal) * tpb_reduce, stream>>>(scratchpad, num_elems);
        kernel_reduce_block<dmin><<<1, 1, 0, stream>>>(scratchpad, bpg_reduce, tpb_reduce, reduce_result);
    } else if (rtype == RTYPE_RMS) {
        kernel_filter_vec<dsquared_vec><<<bpg_filter, tpb_filter, 0, stream>>>(vtxbuf0, vtxbuf1, vtxbuf2, start, end, scratchpad);
        kernel_reduce<dsum><<<bpg_reduce, tpb_reduce, sizeof(AcReal) * tpb_reduce, stream>>>(scratchpad, num_elems);
        kernel_reduce_block<dsum><<<1, 1, 0, stream>>>(scratchpad, bpg_reduce, tpb_reduce, reduce_result);
    } else if (rtype == RTYPE_RMS_EXP) {
        kernel_filter_vec<dexp_squared_vec><<<bpg_filter, tpb_filter, 0, stream>>>(vtxbuf0, vtxbuf1, vtxbuf2, start, end, scratchpad);
        kernel_reduce<dsum><<<bpg_reduce, tpb_reduce, sizeof(AcReal) * tpb_reduce, stream>>>(scratchpad, num_elems);
        kernel_reduce_block<dsum><<<1, 1, 0, stream>>>(scratchpad, bpg_reduce, tpb_reduce, reduce_result);
    } else if (rtype == RTYPE_SUM) {
        kernel_filter_vec<dlength_vec><<<bpg_filter, tpb_filter, 0, stream>>>(vtxbuf0, vtxbuf1, vtxbuf2, start, end, scratchpad);
        kernel_reduce<dsum><<<bpg_reduce, tpb_reduce, sizeof(AcReal) * tpb_reduce, stream>>>(scratchpad, num_elems);
        kernel_reduce_block<dsum><<<1, 1, 0, stream>>>(scratchpad, bpg_reduce, tpb_reduce, reduce_result);
    } else {
        ERROR("Unrecognized rtype");
    }
    // clang-format on

    cudaStreamSynchronize(stream);
    AcReal result;
    cudaMemcpy(&result, reduce_result, sizeof(AcReal), cudaMemcpyDeviceToHost);
    return result;
}
