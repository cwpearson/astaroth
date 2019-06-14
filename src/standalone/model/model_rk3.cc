/*
    Copyright (C) 2014-2018, Johannes Pekkilae, Miikka Vaeisalae.

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
#include "model_rk3.h"

#include <math.h>

#include "host_memory.h"
#include "model_boundconds.h"

typedef struct {
    ModelScalar x, y, z;
} ModelVector;

typedef struct {
    ModelVector row[3];
} ModelMatrix;

typedef struct {
    ModelScalar value;
    ModelVector gradient;
    ModelMatrix hessian;
} ModelScalarData;

typedef struct {
    ModelScalarData x;
    ModelScalarData y;
    ModelScalarData z;
} ModelVectorData;


static AcMeshInfo* mesh_info = NULL;

static inline int
get(const AcIntParam param)
{
    return mesh_info->int_params[param];
}

static inline ModelScalar
get(const AcRealParam param)
{
    return mesh_info->real_params[param];
}

static inline int
IDX(const int i, const int j, const int k)
{
    return AC_VTXBUF_IDX(i, j, k, (*mesh_info));
}

/*
 * =============================================================================
 * Stencil Assembly Stage
 * =============================================================================
 */
static inline ModelScalar
first_derivative(const ModelScalar* pencil, const ModelScalar inv_ds)
{
#if STENCIL_ORDER == 2
    const ModelScalar coefficients[] = {0, 1. / 2.};
#elif STENCIL_ORDER == 4
    const ModelScalar coefficients[] = {0, 2.0 / 3.0, -1.0 / 12.0};
#elif STENCIL_ORDER == 6
    const ModelScalar coefficients[] = {0, 3.0 / 4.0, -3.0 / 20.0, 1.0 / 60.0};
#elif STENCIL_ORDER == 8
    const ModelScalar coefficients[] = {0, 4.0 / 5.0, -1.0 / 5.0, 4.0 / 105.0,
                                   -1.0 / 280.0};
#endif

    #define MID (STENCIL_ORDER / 2)
    ModelScalar res    = 0;

//#pragma unroll
    for (int i = 1; i <= MID; ++i)
        res += coefficients[i] * (pencil[MID + i] - pencil[MID - i]);

    return res * inv_ds;
}

static inline ModelScalar
second_derivative(const ModelScalar* pencil, const ModelScalar inv_ds)
{
#if STENCIL_ORDER == 2
    const ModelScalar coefficients[] = {-2., 1.};
#elif STENCIL_ORDER == 4
    const ModelScalar coefficients[] = {-5.0/2.0, 4.0/3.0, -1.0/12.0};
#elif STENCIL_ORDER == 6
    const ModelScalar coefficients[] = {-49.0 / 18.0, 3.0 / 2.0, -3.0 / 20.0,
                                   1.0 / 90.0};
#elif STENCIL_ORDER == 8
    const ModelScalar coefficients[] = {-205.0 / 72.0, 8.0 / 5.0, -1.0 / 5.0,
                                   8.0 / 315.0, -1.0 / 560.0};
#endif

    #define MID (STENCIL_ORDER / 2)
    ModelScalar res    = coefficients[0] * pencil[MID];

//#pragma unroll
    for (int i = 1; i <= MID; ++i)
        res += coefficients[i] * (pencil[MID + i] + pencil[MID - i]);

    return res * inv_ds * inv_ds;
}

/** inv_ds: inverted mesh spacing f.ex. 1. / mesh.int_params[AC_dsx] */
static inline ModelScalar
cross_derivative(const ModelScalar* pencil_a,
                 const ModelScalar* pencil_b, const ModelScalar inv_ds_a,
                 const ModelScalar inv_ds_b)
{
#if STENCIL_ORDER == 2
    const ModelScalar coefficients[] = {0, 1.0 / 4.0};
#elif STENCIL_ORDER == 4
    const ModelScalar coefficients[] = {0, 1.0 / 32.0, 1.0 / 64.0}; // TODO correct coefficients, these are just placeholders
#elif STENCIL_ORDER == 6
    const ModelScalar fac            = (1. / 720.);
    const ModelScalar coefficients[] = {0.0 * fac, 270.0 * fac, -27.0 * fac,
                                   2.0 * fac};
#elif STENCIL_ORDER == 8
    const ModelScalar fac            = (1. / 20160.);
    const ModelScalar coefficients[] = {0.0 * fac, 8064. * fac, -1008. * fac,
                                   128. * fac, -9. * fac};
#endif

    #define MID (STENCIL_ORDER / 2)
    ModelScalar res    = ModelScalar(0.);

    //#pragma unroll
    for (int i = 1; i <= MID; ++i) {
        res += coefficients[i] * (pencil_a[MID + i] + pencil_a[MID - i] -
                                  pencil_b[MID + i] - pencil_b[MID - i]);
    }
    return res * inv_ds_a * inv_ds_b;
}

static inline ModelScalar
derx(const int i, const int j, const int k, const ModelScalar* arr)
{
    ModelScalar pencil[STENCIL_ORDER + 1];
//#pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil[offset] = arr[IDX(i + offset - STENCIL_ORDER / 2, j, k)];

    return first_derivative(pencil, get(AC_inv_dsx));
}

static inline ModelScalar
derxx(const int i, const int j, const int k, const ModelScalar* arr)
{
    ModelScalar pencil[STENCIL_ORDER + 1];
//#pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil[offset] = arr[IDX(i + offset - STENCIL_ORDER / 2, j, k)];

    return second_derivative(pencil, get(AC_inv_dsx));
}

static inline ModelScalar
derxy(const int i, const int j, const int k, const ModelScalar* arr)
{
    ModelScalar pencil_a[STENCIL_ORDER + 1];
//#pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil_a[offset] = arr[IDX(i + offset - STENCIL_ORDER / 2,
                                   j + offset - STENCIL_ORDER / 2, k)];

    ModelScalar pencil_b[STENCIL_ORDER + 1];
//#pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil_b[offset] = arr[IDX(i + offset - STENCIL_ORDER / 2,
                                   j + STENCIL_ORDER / 2 - offset, k)];

    return cross_derivative(pencil_a, pencil_b, get(AC_inv_dsx),
                            get(AC_inv_dsy));
}

static inline ModelScalar
derxz(const int i, const int j, const int k, const ModelScalar* arr)
{
    ModelScalar pencil_a[STENCIL_ORDER + 1];
//#pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil_a[offset] = arr[IDX(i + offset - STENCIL_ORDER / 2, j,
                                   k + offset - STENCIL_ORDER / 2)];

    ModelScalar pencil_b[STENCIL_ORDER + 1];
//#pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil_b[offset] = arr[IDX(i + offset - STENCIL_ORDER / 2, j,
                                   k + STENCIL_ORDER / 2 - offset)];

    return cross_derivative(pencil_a, pencil_b, get(AC_inv_dsx),
                            get(AC_inv_dsz));
}

static inline ModelScalar
dery(const int i, const int j, const int k, const ModelScalar* arr)
{
    ModelScalar pencil[STENCIL_ORDER + 1];
//#pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil[offset] = arr[IDX(i, j + offset - STENCIL_ORDER / 2, k)];

    return first_derivative(pencil, get(AC_inv_dsy));
}

static inline ModelScalar
deryy(const int i, const int j, const int k, const ModelScalar* arr)
{
    ModelScalar pencil[STENCIL_ORDER + 1];
//#pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil[offset] = arr[IDX(i, j + offset - STENCIL_ORDER / 2, k)];

    return second_derivative(pencil, get(AC_inv_dsy));
}

static inline ModelScalar
deryz(const int i, const int j, const int k, const ModelScalar* arr)
{
    ModelScalar pencil_a[STENCIL_ORDER + 1];
//#pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil_a[offset] = arr[IDX(i, j + offset - STENCIL_ORDER / 2,
                                   k + offset - STENCIL_ORDER / 2)];

    ModelScalar pencil_b[STENCIL_ORDER + 1];
//#pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil_b[offset] = arr[IDX(i, j + offset - STENCIL_ORDER / 2,
                                   k + STENCIL_ORDER / 2 - offset)];

    return cross_derivative(pencil_a, pencil_b, get(AC_inv_dsy),
                            get(AC_inv_dsz));
}

static inline ModelScalar
derz(const int i, const int j, const int k, const ModelScalar* arr)
{
    ModelScalar pencil[STENCIL_ORDER + 1];
//#pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil[offset] = arr[IDX(i, j, k + offset - STENCIL_ORDER / 2)];

    return first_derivative(pencil, get(AC_inv_dsz));
}

static inline ModelScalar
derzz(const int i, const int j, const int k, const ModelScalar* arr)
{
    ModelScalar pencil[STENCIL_ORDER + 1];
//#pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil[offset] = arr[IDX(i, j, k + offset - STENCIL_ORDER / 2)];

    return second_derivative(pencil, get(AC_inv_dsz));
}

static inline ModelScalar
compute_value(const int i, const int j, const int k,
              const ModelScalar* arr)
{
    return arr[IDX(i, j, k)];
}

static inline ModelVector
compute_gradient(const int i, const int j, const int k,
                 const ModelScalar* arr)
{
    return (ModelVector){derx(i, j, k, arr), dery(i, j, k, arr),
                     derz(i, j, k, arr)};
}

static inline ModelMatrix
compute_second_deriv(const int i, const int j, const int k,
                     const ModelScalar* arr)
{
    ModelMatrix hessian;

    hessian.row[0] = (ModelVector){derxx(i, j, k, arr), 0, 0};
    hessian.row[1] = (ModelVector){0, deryy(i, j, k, arr), 0};
    hessian.row[2] = (ModelVector){0, 0, derzz(i, j, k, arr)};

    return hessian;
}

static inline ModelMatrix
compute_hessian(const int i, const int j, const int k,
                const ModelScalar* arr)
{
    ModelMatrix hessian;

    hessian.row[0] = (ModelVector){derxx(i, j, k, arr), derxy(i, j, k, arr), derxz(i, j, k, arr)};
    hessian.row[1] = (ModelVector){hessian.row[0].y,    deryy(i, j, k, arr), deryz(i, j, k, arr)};
    hessian.row[2] = (ModelVector){hessian.row[0].z,    hessian.row[1].z,    derzz(i, j, k, arr)};

    return hessian;
}

static inline ModelScalarData
read_data(const int i, const int j, const int k,
          ModelScalar* buf[], const int handle)
{
    ModelScalarData data;

    data.value    = compute_value(i, j, k, buf[handle]);
    data.gradient = compute_gradient(i, j, k, buf[handle]);

    // No significant effect on performance even though we do not need the
    // diagonals with all arrays
    data.hessian = compute_hessian(i, j, k, buf[handle]);

    return data;
}

static inline ModelVectorData
read_data(const int i, const int j, const int k,
          ModelScalar* buf[], const int3& handle)
{
    ModelVectorData data;

    data.x = read_data(i, j, k, buf, handle.x);
    data.y = read_data(i, j, k, buf, handle.y);
    data.z = read_data(i, j, k, buf, handle.z);

    return data;
}

static inline ModelScalar
value(const ModelScalarData& data)
{
    return data.value;
}

static inline ModelVector
gradient(const ModelScalarData& data)
{
    return data.gradient;
}

static inline ModelMatrix
hessian(const ModelScalarData& data)
{
    return data.hessian;
}

static inline ModelVector
value(const ModelVectorData& data)
{
    return (ModelVector){value(data.x), value(data.y), value(data.z)};
}

static inline ModelMatrix
gradients(const ModelVectorData& data)
{
    return (ModelMatrix){gradient(data.x), gradient(data.y), gradient(data.z)};
}

static inline ModelScalar val2ue(const int i, const int j, const int k, ModelScalar* vertex) {
  return vertex[IDX(i, j, k)];
}
static inline ModelVector gradien2t(const int i, const int j, const int k, ModelScalar* vertex) {
  return (ModelVector){vertex[IDX(i - 1, j, k)] + vertex[IDX(i, j, k)] + vertex[IDX(i + 1, j, k)], vertex[IDX(i, j - 1, k)] + vertex[IDX(i, j, k)] + vertex[IDX(i, j + 1, k)], vertex[IDX(i, j, k - 1)] + vertex[IDX(i, j, k)] + vertex[IDX(i, j, k + 1)]};
}

/*
 * =============================================================================
 * Level 0.3 (Built-in functions available during the Stencil Processing Stage)
 * =============================================================================
 */

static inline ModelVector
operator-(const ModelVector& a, const ModelVector& b)
{
    return (ModelVector){a.x - b.x, a.y - b.y, a.z - b.z};
}

static inline ModelVector
operator+(const ModelVector& a, const ModelVector& b)
{
    return (ModelVector){a.x + b.x, a.y + b.y, a.z + b.z};
}

static inline ModelVector
operator-(const ModelVector& a)
{
    return (ModelVector){-a.x, -a.y, -a.z};
}

static  inline ModelVector
operator*(const ModelScalar a, const ModelVector& b)
{
    return (ModelVector){a * b.x, a * b.y, a * b.z};
}

static inline ModelScalar
dot(const ModelVector& a, const ModelVector& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline ModelVector
mul(const ModelMatrix& aa, const ModelVector& x)
{
    return (ModelVector){dot(aa.row[0], x), dot(aa.row[1], x), dot(aa.row[2], x)};
}

static inline ModelVector
cross(const ModelVector& a, const ModelVector& b)
{
    ModelVector c;

    c.x = a.y * b.z - a.z * b.y;
    c.y = a.z * b.x - a.x * b.z;
    c.z = a.x * b.y - a.y * b.x;

    return c;
}
/*
static inline bool
is_valid(const ModelScalar a)
{
    return !isnan(a) && !isinf(a);
}

static inline bool
is_valid(const ModelVector& a)
{
    return is_valid(a.x) && is_valid(a.y) && is_valid(a.z);
}
*/
/*
 * =============================================================================
 * Stencil Processing Stage (helper functions)
 * =============================================================================
 */
static inline ModelScalar
laplace(const ModelScalarData& data)
{
    return hessian(data).row[0].x + hessian(data).row[1].y + hessian(data).row[2].z;
}

static inline ModelScalar
divergence(const ModelVectorData& vec)
{
    return gradient(vec.x).x + gradient(vec.y).y + gradient(vec.z).z;
}

static inline ModelVector
laplace_vec(const ModelVectorData& vec)
{
    return (ModelVector){laplace(vec.x), laplace(vec.y), laplace(vec.z)};
}

static inline ModelVector
curl(const ModelVectorData& vec)
{
    return (ModelVector){gradient(vec.z).y - gradient(vec.y).z,
                     gradient(vec.x).z - gradient(vec.z).x,
                     gradient(vec.y).x - gradient(vec.x).y};
}

static inline ModelVector
gradient_of_divergence(const ModelVectorData& vec)
{
    return (ModelVector){hessian(vec.x).row[0].x + hessian(vec.y).row[0].y + hessian(vec.z).row[0].z,
                     hessian(vec.x).row[1].x + hessian(vec.y).row[1].y + hessian(vec.z).row[1].z,
                     hessian(vec.x).row[2].x + hessian(vec.y).row[2].y + hessian(vec.z).row[2].z};
}

// Takes uu gradients and returns S
static inline ModelMatrix
stress_tensor(const ModelVectorData& vec)
{
    ModelMatrix S;

    S.row[0].x = ModelScalar(2. / 3.) * gradient(vec.x).x -
                 ModelScalar(1. / 3.) * (gradient(vec.y).y + gradient(vec.z).z);
    S.row[0].y = ModelScalar(1. / 2.) * (gradient(vec.x).y + gradient(vec.y).x);
    S.row[0].z = ModelScalar(1. / 2.) * (gradient(vec.x).z + gradient(vec.z).x);

    S.row[1].y = ModelScalar(2. / 3.) * gradient(vec.y).y -
                 ModelScalar(1. / 3.) * (gradient(vec.x).x + gradient(vec.z).z);

    S.row[1].z = ModelScalar(1. / 2.) * (gradient(vec.y).z + gradient(vec.z).y);

    S.row[2].z = ModelScalar(2. / 3.) * gradient(vec.z).z -
                 ModelScalar(1. / 3.) * (gradient(vec.x).x + gradient(vec.y).y);

    S.row[1].x = S.row[0].y;
    S.row[2].x = S.row[0].z;
    S.row[2].y = S.row[1].z;

    return S;
}

static inline ModelScalar
contract(const ModelMatrix& mat)
{
    ModelScalar res = 0;

    //#pragma unroll
    for (int i = 0; i < 3; ++i)
        res += dot(mat.row[i], mat.row[i]);

    return res;
}

/*
 * =============================================================================
 * Stencil Processing Stage (equations)
 * =============================================================================
 */
static inline ModelScalar
continuity(const ModelVectorData& uu, const ModelScalarData& lnrho)
{
    return - dot(value(uu), gradient(lnrho)) - divergence(uu);
}

static inline ModelScalar
length(const ModelVector& vec)
{
    return sqrtl(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

static inline ModelScalar
reciprocal_len(const ModelVector& vec)
{
    return 1.l / sqrtl(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

static inline ModelVector
normalized(const ModelVector& vec)
{
    const ModelScalar inv_len = reciprocal_len(vec);
    return inv_len * vec;
}


// Note: LNT0 and LNRHO0 must be set very carefully: if the magnitude is different that other values in the mesh, then we will inherently lose precision
#define LNT0 (ModelScalar(0.0))
#define LNRHO0 (ModelScalar(0.0))

#define H_CONST (ModelScalar(0.0))
#define C_CONST (ModelScalar(0.0))

static inline ModelVector
momentum(const ModelVectorData& uu, const ModelScalarData& lnrho
#if LENTROPY
, const ModelScalarData& ss, const ModelVectorData& aa
#endif
)
{
    #if LENTROPY
    const ModelMatrix S = stress_tensor(uu);
    const ModelScalar cs2 = get(AC_cs2_sound) * expl(get(AC_gamma) * value(ss) / get(AC_cp_sound) + (get(AC_gamma) - 1) * (value(lnrho) - LNRHO0));
    const ModelVector  j = (ModelScalar(1.) / get(AC_mu0)) * (gradient_of_divergence(aa) - laplace_vec(aa)); // Current density
    const ModelVector B = curl(aa);
    const ModelScalar inv_rho = ModelScalar(1.) / expl(value(lnrho));

    const ModelVector mom = - mul(gradients(uu), value(uu)) 
                                                       - cs2 * ((ModelScalar(1.) / get(AC_cp_sound)) * gradient(ss) + gradient(lnrho))
                                                       + inv_rho * cross(j, B)
                                                       + get(AC_nu_visc) * (
                                                            laplace_vec(uu) 
                                                        + ModelScalar(1. / 3.) * gradient_of_divergence(uu) 
                                                        + ModelScalar(2.) * mul(S, gradient(lnrho))
                                                        )
                                                        + get(AC_zeta) * gradient_of_divergence(uu);
    return mom;
    #endif

    #if 0
    const ModelMatrix S = stress_tensor(uu);

    //#if LENTROPY
    //const ModelScalar lnrho0 = 1; // TODO correct lnrho0
    const ModelScalar cs02 = get(AC_cs2_sound); // TODO better naming
    const ModelScalar cs2 = cs02;// * expl(get(AC_gamma) * value(ss) / get(AC_cp_sound) + (get(AC_gamma)-ModelScalar(1.l)) * (value(lnrho) - lnrho0));

    mom = -mul(gradients(uu), value(uu)) -
    cs2 * ((ModelScalar(1.) / get(AC_cp_sound)) * gradient(ss) + gradient(lnrho)) +
    get(AC_nu_visc) *
    (laplace_vec(uu) + ModelScalar(1.l / 3.l) * gradient_of_divergence(uu) +
      ModelScalar(2.l) * mul(S, gradient(lnrho))) + get(AC_zeta) * gradient_of_divergence(uu);

    const ModelVector grad_div = gradient_of_divergence(aa);
    const ModelVector lap = laplace_vec(aa);
    const ModelVector j = (ModelScalar(1.l) / get(AC_mu0)) * (grad_div - lap);
    const ModelVector B = curl(aa);
    mom = mom + (ModelScalar(1.l) / expl(value(lnrho))) * cross(j, B);
    //#else // Basic hydro
        const ModelScalar cs02 = get(AC_cs2_sound);
        mom = -mul(gradients(uu), value(uu)) -
          cs02 * gradient(lnrho) +
          get(AC_nu_visc) *
              (laplace_vec(uu) + ModelScalar(1. / 3.) * gradient_of_divergence(uu) +
               ModelScalar(2.) * mul(S, gradient(lnrho))) + get(AC_zeta) * gradient_of_divergence(uu);
    //#endif
    #endif
    return mom;
}

static inline ModelVector
induction(const ModelVectorData& uu, const ModelVectorData& aa)
{
    ModelVector ind;
    // Note: We do (-nabla^2 A + nabla(nabla dot A)) instead of (nabla x (nabla
    // x A)) in order to avoid taking the first derivative twice (did the math,
    // yes this actually works. See pg.28 in arXiv:astro-ph/0109497)
    // u cross B - ETA * mu0 * (mu0^-1 * [- laplace A + grad div A ])
    const ModelVector B        = curl(aa);
    const ModelVector grad_div = gradient_of_divergence(aa);
    const ModelVector lap      = laplace_vec(aa);

    // Note, mu0 is cancelled out
    ind = cross(value(uu), B) - get(AC_eta) * (grad_div - lap);

    return ind;
}

static inline ModelScalar
lnT(const ModelScalarData& ss, const ModelScalarData& lnrho)
{
    const ModelScalar lnT = LNT0 + get(AC_gamma) * value(ss) / get(AC_cp_sound)
                     + (get(AC_gamma) - ModelScalar(1.)) * (value(lnrho) - LNRHO0);
    return lnT;
}


// Nabla dot (K nabla T) / (rho T)
static inline ModelScalar
heat_conduction(const ModelScalarData& ss, const ModelScalarData& lnrho)
{
    const ModelScalar inv_cp_sound = ModelScalar(1.) / get(AC_cp_sound);

    const ModelVector grad_ln_chi = - gradient(lnrho);

    const ModelScalar first_term = get(AC_gamma) * inv_cp_sound * laplace(ss)
                           + (get(AC_gamma) - ModelScalar(1.)) * laplace(lnrho);
    const ModelVector second_term = get(AC_gamma) * inv_cp_sound * gradient(ss)
                             + (get(AC_gamma) - ModelScalar(1.)) * gradient(lnrho);
    const ModelVector third_term = get(AC_gamma) * (inv_cp_sound * gradient(ss)
                                        + gradient(lnrho)) + grad_ln_chi;

    const ModelScalar chi = AC_THERMAL_CONDUCTIVITY / (expl(value(lnrho)) * get(AC_cp_sound));
    return get(AC_cp_sound) * chi * (first_term + dot(second_term, third_term));
}

static inline ModelScalar
entropy(const ModelScalarData& ss, const ModelVectorData& uu, const ModelScalarData& lnrho, const ModelVectorData& aa)
{
    const ModelMatrix S = stress_tensor(uu);
    const ModelScalar inv_pT = ModelScalar(1.) / (expl(value(lnrho)) * expl(lnT(ss, lnrho)));
    const ModelVector  j = (ModelScalar(1.) / get(AC_mu0)) * (gradient_of_divergence(aa) - laplace_vec(aa)); // Current density
    const ModelScalar RHS = H_CONST - C_CONST
                                                + get(AC_eta) * get(AC_mu0) * dot(j, j) 
                                                + ModelScalar(2.) * expl(value(lnrho)) * get(AC_nu_visc) * contract(S)
                                                + get(AC_zeta) * expl(value(lnrho)) * divergence(uu) * divergence(uu);

    return - dot(value(uu), gradient(ss))
                  + inv_pT * RHS
                  + heat_conduction(ss, lnrho);
    /*
    const ModelMatrix S = stress_tensor(uu);

    // nabla x nabla x A / mu0 = nabla(nabla dot A) - nabla^2(A)
    const ModelVector j = gradient_of_divergence(aa) - laplace_vec(aa);

    const ModelScalar inv_pT = ModelScalar(1.) / (expl(value(lnrho)) + expl(lnT(ss, lnrho)));

    return - dot(value(uu), gradient(ss))
           + inv_pT * ( H_CONST - C_CONST
                + get(AC_eta) * get(AC_mu0) * dot(j, j)
                + ModelScalar(2.) * expl(value(lnrho)) * get(AC_nu_visc) * contract(S)
                + get(AC_zeta) * expl(value(lnrho)) * divergence(uu) * divergence(uu)
            )
            + heat_conduction(ss, lnrho);
    */
}

static void
solve_alpha_step(const int step_number, const ModelScalar dt,
                 const int i, const int j, const int k,
                 ModelMesh& in, ModelMesh* out)
{
    const int idx = AC_VTXBUF_IDX(i, j, k, in.info);

    const ModelScalarData lnrho = read_data(i, j, k, in.vertex_buffer, VTXBUF_LNRHO);
    const ModelVectorData uu = read_data(i, j, k, in.vertex_buffer, (int3){VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ});

    ModelScalar rate_of_change[NUM_VTXBUF_HANDLES] = {0};
    rate_of_change[VTXBUF_LNRHO] = continuity(uu, lnrho);

    #if LINDUCTION
        const ModelVectorData aa = read_data(i, j, k, in.vertex_buffer, (int3){VTXBUF_AX, VTXBUF_AY, VTXBUF_AZ});
        const ModelVector aa_res = induction(uu, aa);
        rate_of_change[VTXBUF_AX] = aa_res.x;
        rate_of_change[VTXBUF_AY] = aa_res.y;
        rate_of_change[VTXBUF_AZ] = aa_res.z;
    #endif
    #if LENTROPY
        const ModelScalarData ss = read_data(i, j, k, in.vertex_buffer, VTXBUF_ENTROPY);
        const ModelVector uu_res = momentum(uu, lnrho, ss, aa);
        rate_of_change[VTXBUF_UUX] = uu_res.x;
        rate_of_change[VTXBUF_UUY] = uu_res.y;
        rate_of_change[VTXBUF_UUZ] = uu_res.z;
        rate_of_change[VTXBUF_ENTROPY] = entropy(ss, uu, lnrho, aa);
    #else
        const ModelVector uu_res = momentum(uu, lnrho);
        rate_of_change[VTXBUF_UUX] = uu_res.x;
        rate_of_change[VTXBUF_UUY] = uu_res.y;
        rate_of_change[VTXBUF_UUZ] = uu_res.z;
    #endif



    // Williamson (1980) NOTE: older version of astaroth used inhomogenous
    const ModelScalar alpha[] = {ModelScalar(.0), ModelScalar(-5. / 9.), ModelScalar(-153. / 128.)};
    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        if (step_number == 0) {
            out->vertex_buffer[w][idx] = rate_of_change[w] * dt;
        } else {
            out->vertex_buffer[w][idx] = alpha[step_number] * out->vertex_buffer[w][idx]
                                       + rate_of_change[w] * dt;
        }
    }
}

static void
solve_beta_step(const int step_number, const int i, const int j, const int k,
                   const ModelMesh& in, ModelMesh* out)
{
    const int idx = AC_VTXBUF_IDX(i, j, k, in.info);

    // Williamson (1980) NOTE: older version of astaroth used inhomogenous
    const ModelScalar beta[]  = {ModelScalar(1. / 3.), ModelScalar(15. / 16.), ModelScalar(8. / 15.)};

    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
        out->vertex_buffer[w][idx] += beta[step_number] * in.vertex_buffer[w][idx];
}

void
model_rk3_step(const int step_number, const ModelScalar dt, ModelMesh* mesh)
{
    mesh_info = &(mesh->info);

    ModelMesh* tmp = modelmesh_create(mesh->info);

	boundconds(mesh->info, mesh);
	#pragma omp parallel for
	for (int k = get(AC_nz_min); k < get(AC_nz_max); ++k) {
	    for (int j = get(AC_ny_min); j < get(AC_ny_max); ++j) {
		for (int i = get(AC_nx_min); i < get(AC_nx_max); ++i) {
		    solve_alpha_step(step_number, dt, i, j, k, *mesh, tmp);
		}
	    }
	}
	#pragma omp parallel for
	for (int k = get(AC_nz_min); k < get(AC_nz_max); ++k) {
	    for (int j = get(AC_ny_min); j < get(AC_ny_max); ++j) {
		for (int i = get(AC_nx_min); i < get(AC_nx_max); ++i) {
		    solve_beta_step(step_number, i, j, k, *tmp, mesh);
		}
	    }
	}

    modelmesh_destroy(tmp);
    mesh_info = NULL;
}

void
model_rk3(const ModelScalar dt, ModelMesh* mesh)
{
    mesh_info = &(mesh->info);

    ModelMesh* tmp = modelmesh_create(mesh->info);

    for (int step_number = 0; step_number < 3; ++step_number) {
        boundconds(mesh->info, mesh);
        #pragma omp parallel for
        for (int k = get(AC_nz_min); k < get(AC_nz_max); ++k) {
            for (int j = get(AC_ny_min); j < get(AC_ny_max); ++j) {
                for (int i = get(AC_nx_min); i < get(AC_nx_max); ++i) {
                    solve_alpha_step(step_number, dt, i, j, k, *mesh, tmp);
                }
            }
        }
        #pragma omp parallel for
        for (int k = get(AC_nz_min); k < get(AC_nz_max); ++k) {
            for (int j = get(AC_ny_min); j < get(AC_ny_max); ++j) {
                for (int i = get(AC_nx_min); i < get(AC_nx_max); ++i) {
                    solve_beta_step(step_number, i, j, k, *tmp, mesh);
                }
            }
        }
    }

    modelmesh_destroy(tmp);
    mesh_info = NULL;
}
#if 0
static MODEL_REAL
continuity(const int& i, const int& j, const int& k, const ModelMesh& mesh)
{
    return -vec_dot_nabla_scal(
               i, j, k, mesh.info, mesh.vertex_buffer[VTXBUF_UUX],
               mesh.vertex_buffer[VTXBUF_UUY], mesh.vertex_buffer[VTXBUF_UUZ],
               mesh.vertex_buffer[VTXBUF_LNRHO]) -
           div_vec(i, j, k, mesh.info, mesh.vertex_buffer[VTXBUF_UUX],
                   mesh.vertex_buffer[VTXBUF_UUY],
                   mesh.vertex_buffer[VTXBUF_UUZ]);

    // return laplace_scal(i, j, k, mesh.info,
    // mesh.vertex_buffer[VTXBUF_LNRHO])*mesh.info.real_params[AC_nu_visc];
}

static void
momentum(const int& i, const int& j, const int& k, const ModelMesh& mesh,
         MODEL_REAL* mom_x, MODEL_REAL* mom_y, MODEL_REAL* mom_z)
{
    // Vec dot nabla uu
    const MODEL_REAL vec_dot_nabla_uux = vec_dot_nabla_scal(
        i, j, k, mesh.info, mesh.vertex_buffer[VTXBUF_UUX],
        mesh.vertex_buffer[VTXBUF_UUY], mesh.vertex_buffer[VTXBUF_UUZ],
        mesh.vertex_buffer[VTXBUF_UUX]);
    const MODEL_REAL vec_dot_nabla_uuy = vec_dot_nabla_scal(
        i, j, k, mesh.info, mesh.vertex_buffer[VTXBUF_UUX],
        mesh.vertex_buffer[VTXBUF_UUY], mesh.vertex_buffer[VTXBUF_UUZ],
        mesh.vertex_buffer[VTXBUF_UUY]);
    const MODEL_REAL vec_dot_nabla_uuz = vec_dot_nabla_scal(
        i, j, k, mesh.info, mesh.vertex_buffer[VTXBUF_UUX],
        mesh.vertex_buffer[VTXBUF_UUY], mesh.vertex_buffer[VTXBUF_UUZ],
        mesh.vertex_buffer[VTXBUF_UUZ]);
    // Gradient
    MODEL_REAL ddx_lnrho, ddy_lnrho, ddz_lnrho;
    grad(i, j, k, mesh.info, mesh.vertex_buffer[VTXBUF_LNRHO], &ddx_lnrho,
         &ddy_lnrho, &ddz_lnrho);

    // Viscosity
    MODEL_REAL visc_x, visc_y, visc_z;
    nu_const(i, j, k, mesh.info, mesh.vertex_buffer[VTXBUF_UUX],
             mesh.vertex_buffer[VTXBUF_UUY], mesh.vertex_buffer[VTXBUF_UUZ],
             mesh.vertex_buffer[VTXBUF_LNRHO], &visc_x, &visc_y, &visc_z);

    *mom_x = -vec_dot_nabla_uux -
             mesh.info.real_params[AC_cs2_sound] * ddx_lnrho + visc_x;
    *mom_y = -vec_dot_nabla_uuy -
             mesh.info.real_params[AC_cs2_sound] * ddy_lnrho + visc_y;
    *mom_z = -vec_dot_nabla_uuz -
             mesh.info.real_params[AC_cs2_sound] * ddz_lnrho + visc_z;


    #if  LENTROPY

    #endif
}

#if LINDUCTION
static void
induction(const int& i, const int& j, const int& k, const ModelMesh& mesh,
          MODEL_REAL* ind_x, MODEL_REAL* ind_y, MODEL_REAL* ind_z)
{
    /*
     *ind_x = mesh.vertex_buffer[VTXBUF_AX][AC_VTXBUF_IDX(i, j, k, mesh.info)];
     *ind_y = mesh.vertex_buffer[VTXBUF_AY][AC_VTXBUF_IDX(i, j, k, mesh.info)];
     *ind_z = mesh.vertex_buffer[VTXBUF_AZ][AC_VTXBUF_IDX(i, j, k, mesh.info)];
     */
    const MODEL_REAL ddx_Az = der_scal<AXIS_X>(i, j, k, mesh.info,
                                           mesh.vertex_buffer[VTXBUF_AZ]);
    const MODEL_REAL ddx_Ay = der_scal<AXIS_X>(i, j, k, mesh.info,
                                           mesh.vertex_buffer[VTXBUF_AY]);
    const MODEL_REAL ddy_Ax = der_scal<AXIS_Y>(i, j, k, mesh.info,
                                           mesh.vertex_buffer[VTXBUF_AX]);
    const MODEL_REAL ddy_Az = der_scal<AXIS_Y>(i, j, k, mesh.info,
                                           mesh.vertex_buffer[VTXBUF_AZ]);
    const MODEL_REAL ddz_Ay = der_scal<AXIS_Z>(i, j, k, mesh.info,
                                           mesh.vertex_buffer[VTXBUF_AY]);
    const MODEL_REAL ddz_Ax = der_scal<AXIS_Z>(i, j, k, mesh.info,
                                           mesh.vertex_buffer[VTXBUF_AX]);

    const MODEL_REAL Bx = ddy_Az - ddz_Ay;
    const MODEL_REAL By = ddz_Ax - ddx_Az;
    const MODEL_REAL Bz = ddx_Ay - ddy_Ax;

    MODEL_REAL lx, ly, lz;
    laplace_vec(i, j, k, mesh.info, mesh.vertex_buffer[VTXBUF_AX],
                mesh.vertex_buffer[VTXBUF_AY], mesh.vertex_buffer[VTXBUF_AZ],
                &lx, &ly, &lz);

    MODEL_REAL gx, gy, gz;
    grad_div_vec(i, j, k, mesh.info, mesh.vertex_buffer[VTXBUF_AX],
                 mesh.vertex_buffer[VTXBUF_AY], mesh.vertex_buffer[VTXBUF_AZ],
                 &gx, &gy, &gz);

    const int idx = AC_VTXBUF_IDX(i, j, k, mesh.info);
    *ind_x        = mesh.vertex_buffer[VTXBUF_UUY][idx] * Bz -
             mesh.vertex_buffer[VTXBUF_UUZ][idx] * By -
             mesh.info.real_params[AC_eta] * (-lx + gx);
    *ind_y = mesh.vertex_buffer[VTXBUF_UUZ][idx] * Bx -
             mesh.vertex_buffer[VTXBUF_UUX][idx] * Bz -
             mesh.info.real_params[AC_eta] * (-ly + gy);
    *ind_z = mesh.vertex_buffer[VTXBUF_UUX][idx] * By -
             mesh.vertex_buffer[VTXBUF_UUY][idx] * Bx -
             mesh.info.real_params[AC_eta] * (-lz + gz);
}
#endif

#if LINDUCTION
static inline void
entropy(const int& i, const int& j, const int& k, const ModelMesh& mesh,
        MODEL_REAL* entropy)
{
    // Unused
    (void)i;
    (void)j;
    (void)k;
    (void)mesh;
    (void)entropy;
}
#endif

void
model_rk3(const MODEL_REAL& dt, ModelMesh* mesh)
{
#define INT_PARAM(X) (mesh->info.int_params[X])

    ModelMesh* tmp = modelmesh_create(mesh->info);

    // Williamson (1980) NOTE: older version of astaroth used inhomogenous
    const ModelScalar alphas[] = {.0l, -5.l / 9.l, -153.l / 128.l};
    const ModelScalar betas[]  = {1.l / 3.l, 15.l / 16.l, 8.l / 15.l};

    for (int step_number = 0; step_number < 3; ++step_number) {
        const MODEL_REAL alpha = MODEL_REAL(alphas[step_number]);
        const MODEL_REAL beta  = MODEL_REAL(betas[step_number]);

        boundconds(mesh->info, mesh);
//#pragma omp parallel for
        for (int k = INT_PARAM(AC_nz_min); k < INT_PARAM(AC_nz_max); ++k) {
            for (int j = INT_PARAM(AC_ny_min); j < INT_PARAM(AC_ny_max); ++j) {
                for (int i = INT_PARAM(AC_nx_min); i < INT_PARAM(AC_nx_max);
                     ++i) {
                    const int idx = AC_VTXBUF_IDX(i, j, k, mesh->info);

                    if (step_number == 0) {
                        for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
                            tmp->vertex_buffer[w][idx] = 0;
                    }

                    tmp->vertex_buffer
                        [VTXBUF_LNRHO]
                        [idx] = alpha * tmp->vertex_buffer[VTXBUF_LNRHO][idx] +
                                continuity(i, j, k, *mesh) * dt;

                    MODEL_REAL mom_x, mom_y, mom_z;
                    momentum(i, j, k, *mesh, &mom_x, &mom_y, &mom_z);
                    tmp->vertex_buffer[VTXBUF_UUX]
                                      [idx] = alpha *
                                                  tmp->vertex_buffer[VTXBUF_UUX]
                                                                    [idx] +
                                              mom_x * dt;
                    tmp->vertex_buffer[VTXBUF_UUY]
                                      [idx] = alpha *
                                                  tmp->vertex_buffer[VTXBUF_UUY]
                                                                    [idx] +
                                              mom_y * dt;
                    tmp->vertex_buffer[VTXBUF_UUZ]
                                      [idx] = alpha *
                                                  tmp->vertex_buffer[VTXBUF_UUZ]
                                                                    [idx] +
                                              mom_z * dt;

#if LINDUCTION
                    MODEL_REAL indx, indy, indz;
                    induction(i, j, k, *mesh, &indx, &indy, &indz);
                    tmp->vertex_buffer[VTXBUF_AX]
                                      [idx] = alpha *
                                                  tmp->vertex_buffer[VTXBUF_AX]
                                                                    [idx] +
                                              indx * dt;
                    tmp->vertex_buffer[VTXBUF_AY]
                                      [idx] = alpha *
                                                  tmp->vertex_buffer[VTXBUF_AY]
                                                                    [idx] +
                                              indy * dt;
                    tmp->vertex_buffer[VTXBUF_AZ]
                                      [idx] = alpha *
                                                  tmp->vertex_buffer[VTXBUF_AZ]
                                                                    [idx] +
                                              indz * dt;
#endif

#if LENTROPY
    //MODEL_REAL s
#endif
                }
            }
        }
//#pragma omp parallel for
        for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
            for (int k = INT_PARAM(AC_nz_min); k < INT_PARAM(AC_nz_max); ++k) {
                for (int j = INT_PARAM(AC_ny_min); j < INT_PARAM(AC_ny_max);
                     ++j) {
                    for (int i = INT_PARAM(AC_nx_min); i < INT_PARAM(AC_nx_max);
                         ++i) {
                        const int idx = AC_VTXBUF_IDX(i, j, k, mesh->info);
                        mesh->vertex_buffer[VertexBufferHandle(
                            w)][idx] += beta *
                                        tmp->vertex_buffer[VertexBufferHandle(
                                            w)][idx];
                    }
                }
            }
        }
    }

#undef INT_PARAM
}
#endif
