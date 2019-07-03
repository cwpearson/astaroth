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
 * Provides an interface to Astaroth. Contains all the necessary configuration
 * structs and functions for running the code on multiple GPUs.
 *
 * All interface functions declared here (such as acInit()) operate all GPUs
 * available in the node under the hood, and the user does not need any
 * information about the decomposition, synchronization or such to use these
 * functions.
 *
 */
#pragma once

/* Prevent name mangling */
#ifdef __cplusplus
extern "C" {
#endif

#include <float.h>        // FLT_EPSILON, etc
#include <stdlib.h>       // size_t
#include <vector_types.h> // CUDA vector types (float4, etc)

/*
 * =============================================================================
 * Flags for auto-optimization
 * =============================================================================
 */
#define AUTO_OPTIMIZE (0) // DEPRECATED TODO remove
#define BOUNDCONDS_OPTIMIZE (0)
#define GENERATE_BENCHMARK_DATA (0)
#define VERBOSE_PRINTING (1)

// Device info
#define REGISTERS_PER_THREAD (255)
#define MAX_REGISTERS_PER_BLOCK (65536)
#define MAX_THREADS_PER_BLOCK (1024)
#define MAX_TB_DIM (MAX_THREADS_PER_BLOCK)
#define NUM_ITERATIONS (10)
#define WARP_SIZE (32)
/*
 * =============================================================================
 * Compile-time constants used during simulation (user definable)
 * =============================================================================
 */
// USER_PROVIDED_DEFINES must be defined in user.h if the user wants to override the following
// logical switches
#include "user.h"

// clang-format off
#ifndef USER_PROVIDED_DEFINES
    #include "acc/mhd_solver/stencil_defines.h" // TODO: this should be in a standard location
#endif
// clang-format on

#define AC_THERMAL_CONDUCTIVITY (AcReal(0.001)) // TODO: make an actual config parameter

/*
 * =============================================================================
 * Identifiers used to construct the parameter lists for AcMeshInfo
 * (IntParamType and RealParamType)
 * (user definable)
 * =============================================================================
 */
// clang-format off
#define AC_FOR_INT_PARAM_TYPES(FUNC)\
        /* cparams */\
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
        /* Other */\
        FUNC(AC_max_steps), \
        FUNC(AC_save_steps), \
        FUNC(AC_bin_steps), \
        FUNC(AC_bc_type), \
        /* Additional */\
        FUNC(AC_mxy),\
        FUNC(AC_nxy),\
        FUNC(AC_nxyz)
#define AC_FOR_REAL_PARAM_TYPES(FUNC)\
        /* cparams */\
        FUNC(AC_dsx), \
        FUNC(AC_dsy), \
        FUNC(AC_dsz), \
        FUNC(AC_dsmin), \
        /* physical grid*/\
        FUNC(AC_xlen), \
        FUNC(AC_ylen), \
        FUNC(AC_zlen), \
        FUNC(AC_xorig), \
        FUNC(AC_yorig), \
        FUNC(AC_zorig), \
        /*Physical units*/\
        FUNC(AC_unit_density),\
        FUNC(AC_unit_velocity),\
        FUNC(AC_unit_length),\
        /* properties of gravitating star*/\
        FUNC(AC_star_pos_x),\
        FUNC(AC_star_pos_y),\
        FUNC(AC_star_pos_z),\
        FUNC(AC_M_star),\
        /* Run params */\
        FUNC(AC_cdt), \
        FUNC(AC_cdtv), \
        FUNC(AC_cdts), \
        FUNC(AC_nu_visc), \
        FUNC(AC_cs_sound), \
        FUNC(AC_eta), \
        FUNC(AC_mu0), \
        FUNC(AC_cp_sound), \
        FUNC(AC_gamma), \
        FUNC(AC_cv_sound), \
        FUNC(AC_lnT0), \
        FUNC(AC_lnrho0), \
        FUNC(AC_zeta), \
        FUNC(AC_trans),\
        /* Other */\
        FUNC(AC_bin_save_t), \
        /* Initial condition params */\
        FUNC(AC_ampl_lnrho), \
        FUNC(AC_ampl_uu), \
        FUNC(AC_angl_uu), \
        FUNC(AC_lnrho_edge),\
        FUNC(AC_lnrho_out),\
	/* Forcing parameters. User configured. */\
        FUNC(AC_forcing_magnitude),\
        FUNC(AC_relhel), \
        FUNC(AC_kmin), \
        FUNC(AC_kmax), \
	/* Forcing parameters. Set by the generator. */\
        FUNC(AC_forcing_phase),\
        FUNC(AC_k_forcex),\
        FUNC(AC_k_forcey),\
        FUNC(AC_k_forcez),\
        FUNC(AC_kaver),\
        FUNC(AC_ff_hel_rex),\
        FUNC(AC_ff_hel_rey),\
        FUNC(AC_ff_hel_rez),\
        FUNC(AC_ff_hel_imx),\
        FUNC(AC_ff_hel_imy),\
        FUNC(AC_ff_hel_imz),\
        /* Additional helper params */\
        /* (deduced from other params do not set these directly!) */\
        FUNC(AC_G_CONST),\
        FUNC(AC_GM_star),\
        FUNC(AC_sq2GM_star),\
        FUNC(AC_cs2_sound), \
        FUNC(AC_inv_dsx), \
        FUNC(AC_inv_dsy), \
        FUNC(AC_inv_dsz)
// clang-format on

/*
 * =============================================================================
 * Identifiers for VertexBufferHandle
 * (i.e. the arrays used to construct AcMesh)
 * (user definable)
 * =============================================================================
 */
// clang-format off
#if LDENSITY
#define AC_FOR_DENSITY_VTXBUF_HANDLES(FUNC) \
        FUNC(VTXBUF_LNRHO),
#else
#define AC_FOR_DENSITY_VTXBUF_HANDLES(FUNC)
#endif

#if LHYDRO
#define AC_FOR_HYDRO_VTXBUF_HANDLES(FUNC) \
        FUNC(VTXBUF_UUX), \
        FUNC(VTXBUF_UUY), \
        FUNC(VTXBUF_UUZ),
#else
#define AC_FOR_HYDRO_VTXBUF_HANDLES(FUNC)
#endif

#if LMAGNETIC
#define AC_FOR_MAGNETIC_VTXBUF_HANDLES(FUNC) \
        FUNC(VTXBUF_AX), \
        FUNC(VTXBUF_AY), \
        FUNC(VTXBUF_AZ),
#else
#define AC_FOR_MAGNETIC_VTXBUF_HANDLES(FUNC)
#endif

#if LENTROPY
#define AC_FOR_ENTROPY_VTXBUF_HANDLES(FUNC) \
        FUNC(VTXBUF_ENTROPY),
#else
#define AC_FOR_ENTROPY_VTXBUF_HANDLES(FUNC)
#endif

//MR: Temperature must not have an additional variable slot, but should sit on the
//    same as entropy.
#if LTEMPERATURE
    #define AC_FOR_TEMPERATURE_VTXBUF_HANDLES(FUNC)\
          FUNC(VTXBUF_TEMPERATURE),
#else
    #define AC_FOR_TEMPERATURE_VTXBUF_HANDLES(FUNC)
#endif

#define AC_FOR_VTXBUF_HANDLES(FUNC) AC_FOR_HYDRO_VTXBUF_HANDLES(FUNC) \
                                    AC_FOR_DENSITY_VTXBUF_HANDLES(FUNC) \
                                    AC_FOR_ENTROPY_VTXBUF_HANDLES(FUNC) \
                                    AC_FOR_MAGNETIC_VTXBUF_HANDLES(FUNC) \
// clang-format on

/*
 * =============================================================================
 * Single/double precision switch
 * =============================================================================
 */
// clang-format off
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
// clang-format on

typedef struct {
    AcReal3 row[3];
} AcMatrix;

/*
 * =============================================================================
 * Helper macros
 * =============================================================================
 */
#define AC_GEN_ID(X) X
#define AC_GEN_STR(X) #X

/*
 * =============================================================================
 * Error codes
 * =============================================================================
 */
typedef enum { AC_SUCCESS = 0, AC_FAILURE = 1 } AcResult;

/*
 * =============================================================================
 * Reduction types
 * =============================================================================
 */
typedef enum { RTYPE_MAX, RTYPE_MIN, RTYPE_RMS, RTYPE_RMS_EXP, NUM_REDUCTION_TYPES } ReductionType;

/*
 * =============================================================================
 * Definitions for the enums and structs for AcMeshInfo (DO NOT TOUCH)
 * =============================================================================
 */
typedef enum { AC_FOR_INT_PARAM_TYPES(AC_GEN_ID), NUM_INT_PARAM_TYPES } AcIntParam;

typedef enum { AC_FOR_REAL_PARAM_TYPES(AC_GEN_ID), NUM_REAL_PARAM_TYPES } AcRealParam;
// typedef enum { AC_FOR_VEC_PARAM_TYPES(AC_GEN_ID), NUM_VEC_PARAM_TYPES } AcVecParam;

extern const char* intparam_names[];  // Defined in astaroth.cu
extern const char* realparam_names[]; // Defined in astaroth.cu

typedef struct {
    int int_params[NUM_INT_PARAM_TYPES];
    AcReal real_params[NUM_REAL_PARAM_TYPES];
    // AcReal* vec_params[NUM_VEC_PARAM_TYPES];
} AcMeshInfo;

/*
 * =============================================================================
 * Definitions for the enums and structs for AcMesh (DO NOT TOUCH)
 * =============================================================================
 */
typedef enum { AC_FOR_VTXBUF_HANDLES(AC_GEN_ID) NUM_VTXBUF_HANDLES } VertexBufferHandle;

extern const char* vtxbuf_names[]; // Defined in astaroth.cu

/*
typedef struct {
    AcReal* data;
} VertexBuffer;
*/

// NOTE: there's no particular benefit declaring AcMesh a class, since
// a library user may already have allocated memory for the vertex_buffers.
// But then we would allocate memory again when the user wants to start
// filling the class with data. => Its better to consider AcMesh as a
// payload-only struct
typedef struct {
    AcReal* vertex_buffer[NUM_VTXBUF_HANDLES];
    AcMeshInfo info;
} AcMesh;

#define AC_VTXBUF_SIZE(mesh_info)                                                                  \
    ((size_t)(mesh_info.int_params[AC_mx] * mesh_info.int_params[AC_my] *                          \
              mesh_info.int_params[AC_mz]))

#define AC_VTXBUF_SIZE_BYTES(mesh_info) (sizeof(AcReal) * AC_VTXBUF_SIZE(mesh_info))

#define AC_VTXBUF_COMPDOMAIN_SIZE(mesh_info)                                                       \
    (mesh_info.int_params[AC_nx] * mesh_info.int_params[AC_ny] * mesh_info.int_params[AC_nz])

#define AC_VTXBUF_COMPDOMAIN_SIZE_BYTES(mesh_info)                                                 \
    (sizeof(AcReal) * AC_VTXBUF_COMPDOMAIN_SIZE(mesh_info))

#define AC_VTXBUF_IDX(i, j, k, mesh_info)                                                          \
    ((i) + (j)*mesh_info.int_params[AC_mx] +                                                       \
     (k)*mesh_info.int_params[AC_mx] * mesh_info.int_params[AC_my])

/*
 * =============================================================================
 * Astaroth interface
 * =============================================================================
 */

/** Checks whether there are any CUDA devices available. Returns AC_SUCCESS if there is 1 or more,
    AC_FAILURE otherwise. */
AcResult acCheckDeviceAvailability(void);

/** Starting point of all GPU computation. Handles the allocation and
initialization of *all memory needed on all GPUs in the node*. In other words,
setups everything GPU-side so that calling any other GPU interface function
afterwards does not result in illegal memory accesses. */
AcResult acInit(const AcMeshInfo& mesh_info);

/** Splits the host_mesh and distributes it among the GPUs in the node */
AcResult acLoad(const AcMesh& host_mesh);
AcResult acLoadWithOffset(const AcMesh& host_mesh, const int3& start, const int num_vertices);

/** Does all three steps of the RK3 integration and computes the boundary
conditions when necessary. Note that the boundary conditions are not applied
after the final integration step.
The result can be fetched to CPU memory with acStore(). */
AcResult acIntegrate(const AcReal& dt);

/** Performs a single RK3 step without computing boundary conditions. */
AcResult acIntegrateStep(const int& isubstep, const AcReal& dt);

/** Performs a single RK3 step without computing boundary conditions.
    Operates on a three-dimensional cuboid, where start and end are the
    opposite corners. */
AcResult acIntegrateStepWithOffset(const int& isubstep, const AcReal& dt, const int3& start,
                                   const int3& end);

/** Applies boundary conditions on the GPU meshs and communicates the
 ghost zones among GPUs if necessary */
AcResult acBoundcondStep(void);

/** Performs a scalar reduction on all GPUs in the node and returns the result.
 */
AcReal acReduceScal(const ReductionType& rtype, const VertexBufferHandle& a);

/** Performs a vector reduction on all GPUs in the node and returns the result.
 */
AcReal acReduceVec(const ReductionType& rtype, const VertexBufferHandle& a,
                   const VertexBufferHandle& b, const VertexBufferHandle& c);

/** Stores the mesh distributed among GPUs of the node back to a single host
 * mesh */
AcResult acStore(AcMesh* host_mesh);
AcResult acStoreWithOffset(const int3& start, const int num_vertices, AcMesh* host_mesh);

/** Frees all GPU allocations and resets all devices in the node. Should be
 * called at exit. */
AcResult acQuit(void);

/** Synchronizes all devices. All calls to Astaroth are asynchronous by default
    unless otherwise stated. */
AcResult acSynchronize(void);

/** */
AcResult acLoadDeviceConstant(const AcRealParam param, const AcReal value);

/** Tool for loading forcing vector information into the device memory
 */
AcResult acForcingVec(const AcReal forcing_magnitude, const AcReal3 k_force,
                      const AcReal3 ff_hel_re, const AcReal3 ff_hel_im, const AcReal forcing_phase,
                      const AcReal kaver);

/* End extern "C" */
#ifdef __cplusplus
}
#endif
