/*
    Copyright (C) 2014-2020, Johannes Pekkila, Miikka Vaisala.

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
 #include "astaroth.h"

#include <math.h>

#include "errchk.h"

#if AC_DOUBLE_PRECISION == 0 // HACK TODO fix, make cleaner (purkkaratkaisu)
#define fabs fabsf
#define exp expf
#define sqrt sqrtf
#endif

// Function pointer definitions
typedef AcReal (*ReduceFunc)(const AcReal, const AcReal);
typedef AcReal (*ReduceInitialScalFunc)(const AcReal);
typedef AcReal (*ReduceInitialVecFunc)(const AcReal, const AcReal,
                                            const AcReal);

// clang-format off
/* Comparison funcs */
static inline AcReal
max(const AcReal a, const AcReal b) { return a > b ? a : b; }

static inline AcReal
min(const AcReal a, const AcReal b) { return a < b ? a : b; }

static inline AcReal
sum(const AcReal a, const AcReal b) { return a + b; }

/* Function used to determine the values used during reduction */
static inline AcReal
length_scal(const AcReal a) { return (AcReal)(a); }

static inline AcReal
length_vec(const AcReal a, const AcReal b, const AcReal c) { return sqrt(a*a + b*b + c*c); }

static inline AcReal
squared_scal(const AcReal a) { return (AcReal)(a*a); }

static inline AcReal
squared_vec(const AcReal a, const AcReal b, const AcReal c) { return squared_scal(a) + squared_scal(b) + squared_scal(c); }

static inline AcReal
exp_squared_scal(const AcReal a) { return exp(a)*exp(a); }

static inline AcReal
exp_squared_vec(const AcReal a, const AcReal b, const AcReal c) { return exp_squared_scal(a) + exp_squared_scal(b) + exp_squared_scal(c); }
// clang-format on

AcReal
acHostReduceScal(const AcMesh mesh, const ReductionType rtype, const VertexBufferHandle a)
{
    ReduceInitialScalFunc reduce_initial;
    ReduceFunc reduce;

    bool solve_mean = false;

    switch (rtype) {
    case RTYPE_MAX:
        reduce_initial = length_scal;
        reduce         = max;
        break;
    case RTYPE_MIN:
        reduce_initial = length_scal;
        reduce         = min;
        break;
    case RTYPE_RMS:
        reduce_initial = squared_scal;
        reduce         = sum;
        solve_mean     = true;
        break;
    case RTYPE_RMS_EXP:
        reduce_initial = exp_squared_scal;
        reduce         = sum;
        solve_mean     = true;
        break;
    case RTYPE_SUM:
        reduce_initial = length_scal;
        reduce         = sum;
        break;
    default:
        ERROR("Unrecognized RTYPE");
    }

    const int initial_idx = acVertexBufferIdx(mesh.info.int_params[AC_nx_min],
                                              mesh.info.int_params[AC_ny_min],
                                              mesh.info.int_params[AC_nz_min], mesh.info);

    AcReal res;
    if (rtype == RTYPE_MAX || rtype == RTYPE_MIN)
        res = reduce_initial(mesh.vertex_buffer[a][initial_idx]);
    else
        res = 0;

    for (int k = mesh.info.int_params[AC_nz_min]; k < mesh.info.int_params[AC_nz_max]; ++k) {
        for (int j = mesh.info.int_params[AC_ny_min]; j < mesh.info.int_params[AC_ny_max]; ++j) {
            for (int i = mesh.info.int_params[AC_nx_min]; i < mesh.info.int_params[AC_nx_max];
                 ++i) {
                const int idx              = acVertexBufferIdx(i, j, k, mesh.info);
                const AcReal curr_val = reduce_initial(mesh.vertex_buffer[a][idx]);
                res                        = reduce(res, curr_val);
            }
        }
    }

    if (solve_mean) {
        const AcReal inv_n = (AcReal)1.0 / mesh.info.int_params[AC_nxyz];
        return sqrt(inv_n * res);
    }
    else {
        return res;
    }
}

AcReal
acHostReduceVec(const AcMesh mesh, const ReductionType rtype, const VertexBufferHandle a,
                 const VertexBufferHandle b, const VertexBufferHandle c)
{
    // AcReal (*reduce_initial)(AcReal, AcReal, AcReal);
    ReduceInitialVecFunc reduce_initial;
    ReduceFunc reduce;

    bool solve_mean = false;

    switch (rtype) {
    case RTYPE_MAX:
        reduce_initial = length_vec;
        reduce         = max;
        break;
    case RTYPE_MIN:
        reduce_initial = length_vec;
        reduce         = min;
        break;
    case RTYPE_RMS:
        reduce_initial = squared_vec;
        reduce         = sum;
        solve_mean     = true;
        break;
    case RTYPE_RMS_EXP:
        reduce_initial = exp_squared_vec;
        reduce         = sum;
        solve_mean     = true;
        break;
    case RTYPE_SUM:
        reduce_initial = length_vec;
        reduce         = sum;
        break;
    default:
        ERROR("Unrecognized RTYPE");
    }

    const int initial_idx = acVertexBufferIdx(mesh.info.int_params[AC_nx_min],
                                              mesh.info.int_params[AC_ny_min],
                                              mesh.info.int_params[AC_nz_min], mesh.info);

    AcReal res;
    if (rtype == RTYPE_MAX || rtype == RTYPE_MIN)
        res = reduce_initial(mesh.vertex_buffer[a][initial_idx], mesh.vertex_buffer[b][initial_idx],
                             mesh.vertex_buffer[c][initial_idx]);
    else
        res = 0;

    for (int k = mesh.info.int_params[AC_nz_min]; k < mesh.info.int_params[AC_nz_max]; k++) {
        for (int j = mesh.info.int_params[AC_ny_min]; j < mesh.info.int_params[AC_ny_max]; j++) {
            for (int i = mesh.info.int_params[AC_nx_min]; i < mesh.info.int_params[AC_nx_max];
                 i++) {
                const int idx              = acVertexBufferIdx(i, j, k, mesh.info);
                const AcReal curr_val = reduce_initial(mesh.vertex_buffer[a][idx],
                                                            mesh.vertex_buffer[b][idx],
                                                            mesh.vertex_buffer[c][idx]);
                res                        = reduce(res, curr_val);
            }
        }
    }

    if (solve_mean) {
        const AcReal inv_n = (AcReal)1.0 / mesh.info.int_params[AC_nxyz];
        return sqrt(inv_n * res);
    }
    else {
        return res;
    }
}
