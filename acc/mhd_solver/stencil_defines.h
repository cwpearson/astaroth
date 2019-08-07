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

/*
 * =============================================================================
 * Logical switches
 * =============================================================================
 */
#define STENCIL_ORDER (6)
#define NGHOST (STENCIL_ORDER / 2)
#define LDENSITY (1)
#define LHYDRO (1)
#define LMAGNETIC (1)
#define LENTROPY (1)
#define LTEMPERATURE (0)
#define LFORCING (1)
#define LUPWD (1)

#define AC_THERMAL_CONDUCTIVITY (AcReal(0.001)) // TODO: make an actual config parameter

/*
 * =============================================================================
 * User-defined parameters
 * =============================================================================
 */
// clang-format off
#define AC_FOR_USER_INT_PARAM_TYPES(FUNC)\
        /* Other */\
        FUNC(AC_max_steps), \
        FUNC(AC_save_steps), \
        FUNC(AC_bin_steps), \
        FUNC(AC_bc_type),

#define AC_FOR_USER_INT3_PARAM_TYPES(FUNC)

#define AC_FOR_USER_REAL_PARAM_TYPES(FUNC)\
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
        FUNC(AC_inv_dsz),

#define AC_FOR_USER_REAL3_PARAM_TYPES(FUNC)
// clang-format on

/*
 * =============================================================================
 * User-defined vertex buffers
 * =============================================================================
 */
// clang-format off
#if LENTROPY
#define AC_FOR_VTXBUF_HANDLES(FUNC) \
    FUNC(VTXBUF_LNRHO), \
    FUNC(VTXBUF_UUX), \
    FUNC(VTXBUF_UUY), \
    FUNC(VTXBUF_UUZ), \
    FUNC(VTXBUF_AX), \
    FUNC(VTXBUF_AY), \
    FUNC(VTXBUF_AZ), \
    FUNC(VTXBUF_ENTROPY),
#elif LMAGNETIC
#define AC_FOR_VTXBUF_HANDLES(FUNC) \
    FUNC(VTXBUF_LNRHO), \
    FUNC(VTXBUF_UUX), \
    FUNC(VTXBUF_UUY), \
    FUNC(VTXBUF_UUZ), \
    FUNC(VTXBUF_AX), \
    FUNC(VTXBUF_AY), \
    FUNC(VTXBUF_AZ),
#elif LHYDRO
#define AC_FOR_VTXBUF_HANDLES(FUNC) \
    FUNC(VTXBUF_LNRHO), \
    FUNC(VTXBUF_UUX), \
    FUNC(VTXBUF_UUY), \
    FUNC(VTXBUF_UUZ),
#else
#define AC_FOR_VTXBUF_HANDLES(FUNC) \
    FUNC(VTXBUF_LNRHO),
#endif
// clang-format on
