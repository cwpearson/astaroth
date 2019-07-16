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
#include "device_globals.cuh"


//TODO MV: MAKE A BETTER SWITCH
#define B_VELTYPE 666
//#define B_VELTYPE 1

////////////////////////////////////
// Define the destination indices //
////////////////////////////////////


// Get the standard coordinate indices. 
__device__ void
get_dst_index(const int3 start, int* i_dst, int* j_dst, int* k_dst, int* dst_idx)
{
     *i_dst = start.x + threadIdx.x + blockIdx.x * blockDim.x;
     *j_dst = start.y + threadIdx.y + blockIdx.y * blockDim.y;
     *k_dst = start.z + threadIdx.z + blockIdx.z * blockDim.z;
     *dst_idx      = DEVICE_VTXBUF_IDX(*i_dst, *j_dst, *k_dst);

     //printf("*i_dst = %i, *j_dst = %i, *k_dst = %i, *dst_idx = %i \n", *i_dst, *j_dst, *k_dst, *dst_idx);
}


__device__ void
_sym_indexing_xz(int* edge_idx, int* src_idx, 
                 const int i_dst, const int j_dst, const int k_dst)
{

    int i_edge, j_edge, k_edge;
    int i_diff, k_diff;
    int i_src,  k_src ;
    int is_ztop = 0, is_zbot = 0;
    int is_xtop = 0, is_xbot = 0;

    if (i_dst  < DCONST_INT(AC_nx_min)){
        i_edge = DCONST_INT(AC_nx_min);
        is_xbot = 1;
    } else if (i_dst >=  DCONST_INT(AC_nx_max)){
        i_edge = DCONST_INT(AC_nx_max)-1;
        is_xtop = 1;
    } else {
        i_edge = i_dst;
    }

    j_edge = j_dst;

    if (k_dst < DCONST_INT(AC_nz_min)) {
        k_edge = DCONST_INT(AC_nz_min);
        is_zbot = 1;
    } else if (k_dst >=  DCONST_INT(AC_nz_max)) {
        k_edge = DCONST_INT(AC_nz_max)-1;
        is_ztop = 1;
    } else {
        k_edge = k_dst;
    }

    *edge_idx = DEVICE_VTXBUF_IDX(i_edge, j_edge, k_edge);

    //TODO: problematic on the corners!!! 
    if (is_xtop == 1 || is_xbot == 1) {
        i_diff = i_edge - i_dst;
        i_src  = i_edge + i_diff;
        //OK 
        //printf("i_edge %i, j_edge %i, k_edge %i, i_dst %i, j_dst %i, k_dst %i \n", i_edge, j_edge, k_edge, i_dst, j_dst, k_dst);
        *src_idx = DEVICE_VTXBUF_IDX(i_src, j_dst, k_dst);
    } else if (is_ztop == 1 || is_zbot == 1) {
        k_diff = k_edge - k_dst;
        k_src  = k_edge + k_diff;
        *src_idx = DEVICE_VTXBUF_IDX(i_dst, j_dst, k_src);
    } else {
        *src_idx = NULL;
    }
}


//////////////////////////////////////
// Choose surface value points only //
//////////////////////////////////////

// Choose negative x-boundary SURFACE values.  
__device__ int
choose_negxbound_point(const int i_dst, const int j_dst, const int k_dst)
{
    if (i_dst == DCONST_INT(AC_nx_min))
        return 0;

    return 1;
}

// Choose positive x-boundary SURFACE values.  
__device__ int
choose_posxbound_point(const int i_dst, const int j_dst, const int k_dst)
{
    if (i_dst == (DCONST_INT(AC_nx_max)-1))
        return 0;

    return 1;
}

// Choose negative z-boundary SURFACE values.  
__device__ int
choose_negzbound_point(const int i_dst, const int j_dst, const int k_dst)
{
    if (k_dst == DCONST_INT(AC_nz_min))
        return 0;

    return 1;
}

// Choose positive z-boundary SURFACE values.  
__device__ int
choose_poszbound_point(const int i_dst, const int j_dst, const int k_dst)
{
    if (k_dst == (DCONST_INT(AC_nz_max)-1))
        return 0;

    return 1;
}

////////////////////////////////////
// Filtering out of bounds values //
////////////////////////////////////

// If within the start-end range (this allows threadblock dims that are not
// divisible by end - start)
__device__ int
filter_outbound(const int3 end, const int i_dst, const int j_dst, const int k_dst)
{
    if (i_dst >= end.x || j_dst >= end.y || k_dst >= end.z)
        return 1;

    return 0; 
}

// If destination index is inside the computational domain, return since
// the boundary conditions are only applied to the ghost zones
__device__ int
filter_inbound(const int i_dst, const int j_dst, const int k_dst)
{
    if (i_dst >= DCONST_INT(AC_nx_min) && i_dst < DCONST_INT(AC_nx_max) &&
        j_dst >= DCONST_INT(AC_ny_min) && j_dst < DCONST_INT(AC_ny_max) &&
        k_dst >= DCONST_INT(AC_nz_min) && k_dst < DCONST_INT(AC_nz_max))
        return 1;

    return 0;
}

// If destination index is within x boundary, we do not need it
__device__ int
filter_xbound(const int i_dst, const int j_dst, const int k_dst)
{
    if (i_dst < DCONST_INT(AC_nx_min) || i_dst >= DCONST_INT(AC_nx_max))
        return 1;

    return 0;
}


// Discard negative x-boundary values.  
__device__ int
filter_negxbound(const int i_dst, const int j_dst, const int k_dst)
{
    if (i_dst < DCONST_INT(AC_nx_min))
        return 1;

    return 0;
}

// Discard positive x-boundary values.  
__device__ int
filter_posxbound(const int i_dst, const int j_dst, const int k_dst)
{
    if (i_dst >= DCONST_INT(AC_nx_max))
        return 1;

    return 0;
}

// Discard y-boundary values.  
__device__ int
filter_ybound(const int i_dst, const int j_dst, const int k_dst)
{
    if ((j_dst <  DCONST_INT(AC_ny_min) || j_dst >= DCONST_INT(AC_ny_max)))
        return 1;

    return 0;
}

// Discard negative y-boundary values.  
__device__ int
filter_negybound(const int i_dst, const int j_dst, const int k_dst)
{
    if (j_dst < DCONST_INT(AC_ny_min))
        return 1;

    return 0;
}

// Discard positive y-boundary values.  
__device__ int
filter_posybound(const int i_dst, const int j_dst, const int k_dst)
{
    if (j_dst >= DCONST_INT(AC_ny_max))
        return 1;

    return 0;
}

// If destination index is within z boundary, we do not need it
__device__ int
filter_zbound(const int i_dst, const int j_dst, const int k_dst)
{
    if (k_dst < DCONST_INT(AC_nz_min) || k_dst >= DCONST_INT(AC_nz_max))
        return 1;

    return 0;
}


// Discard negative z-boundary values.  
__device__ int
filter_negzbound(const int i_dst, const int j_dst, const int k_dst)
{
    if (k_dst < DCONST_INT(AC_nz_min))
        return 1;

    return 0;
}

// Discard positive x-boundary values.  
__device__ int
filter_poszbound(const int i_dst, const int j_dst, const int k_dst)
{
    if (k_dst >= DCONST_INT(AC_nz_max))
        return 1;

    return 0;
}

// Discard all exept negative x
__device__ int
filter_allbut_negxbound(const int i_dst, const int j_dst, const int k_dst)
{
    if (i_dst >= DCONST_INT(AC_nx_min))
        return 1;

    return 0;
}

// Discard all exept positive x
__device__ int
filter_allbut_posxbound(const int i_dst, const int j_dst, const int k_dst)
{
    if (i_dst < DCONST_INT(AC_nx_max))
        return 1;

    return 0;
}






/////////////////////////////////////
// Constant values at the boundary //
/////////////////////////////////////


__global__ void
_set_density_xzin(const int3 start, const int3 end, AcReal* density_buffer)    
{
    int i_dst, j_dst, k_dst, dst_idx;
    get_dst_index(start, &i_dst, &j_dst, &k_dst, &dst_idx);

    //Skip threads which are not at valid boundaries.
    if (filter_outbound(end, i_dst, j_dst, k_dst)) return;
    if ( (choose_posxbound_point(i_dst, j_dst, k_dst)) &&
         (choose_negzbound_point(i_dst, j_dst, k_dst)) && 
         (choose_poszbound_point(i_dst, j_dst, k_dst)) ) return;

    density_buffer[dst_idx] =  DCONST_REAL(AC_lnrho_edge);

}

__global__ void
_set_density_xin(const int3 start, const int3 end, AcReal* density_buffer)    
{
    int i_dst, j_dst, k_dst, dst_idx;
    get_dst_index(start, &i_dst, &j_dst, &k_dst, &dst_idx);
    //const int i_dst = start.x + threadIdx.x + blockIdx.x * blockDim.x;
    //const int j_dst = start.y + threadIdx.y + blockIdx.y * blockDim.y;
    //const int k_dst = start.z + threadIdx.z + blockIdx.z * blockDim.z;
    //int dst_idx      = DEVICE_VTXBUF_IDX(i_dst, j_dst, k_dst);

    //Skip threads which are not at valid boundaries.
    if (filter_outbound(end, i_dst, j_dst, k_dst)) return;
    //if ( choose_posxbound_point(i_dst, j_dst, k_dst) ) return;
    if ( filter_allbut_posxbound(i_dst, j_dst, k_dst) ) return;

    //if (dst_idx >= DCONST_INT(AC_mx)*DCONST_INT(AC_my)*DCONST_INT(AC_mz)) 
    //printf(" %i %i %i %i XX %i %i  \n", i_dst, j_dst, k_dst, dst_idx,  
    //                                    i_dst + j_dst*DCONST_INT(AC_mx) + k_dst*DCONST_INT(AC_mx)*DCONST_INT(AC_my), 
    //                                    DCONST_INT(AC_mx)*DCONST_INT(AC_my)*DCONST_INT(AC_mz) );

    //TODO: add a powerlaw gradient. 
    density_buffer[dst_idx] = DCONST_REAL(AC_lnrho_edge);

    //printf(" %i %i %i %i %e %i %i  \n", i_dst, j_dst, k_dst, dst_idx, density_buffer[dst_idx], 
    //                                    i_dst + j_dst*DCONST_INT(AC_mx) + k_dst*DCONST_INT(AC_mx)*DCONST_INT(AC_my), 
    //                                    DCONST_INT(AC_mx)*DCONST_INT(AC_my)*DCONST_INT(AC_mz) );

}


__global__ void
_set_density_xout(const int3 start, const int3 end, AcReal* density_buffer)    
{
    int i_dst, j_dst, k_dst, dst_idx;
    get_dst_index(start, &i_dst, &j_dst, &k_dst, &dst_idx);

    if (filter_outbound(end, i_dst, j_dst, k_dst)) return;
    if ( choose_negxbound_point(i_dst, j_dst, k_dst) ) return;

    //printf(" PIIP ");
    density_buffer[dst_idx] = DCONST_REAL(AC_lnrho_out); 
    //printf(" %i %i %i %i %e %i %i  \n", i_dst, j_dst, k_dst, dst_idx, density_buffer[dst_idx], 
    //                                    i_dst + j_dst*DCONST_INT(AC_mx) + k_dst*DCONST_INT(AC_mx)*DCONST_INT(AC_my), 
    //                                    DCONST_INT(AC_mx)*DCONST_INT(AC_my)*DCONST_INT(AC_mz) );


}

__device__ void 
_calc_csconst(const int dst_idx, AcReal* entropy_buffer, AcReal* density_buffer)
{
    //DUMMY!!! TODO: Test isothermal first. 

    //AcReal TT_bound = 0.0; //AcReal(2.0) * DCONST_REAL(AC_cv_sound) * log(cs2bound/ DCONST_REAL(AC_cp_sound));

    //entropy_buffer[dst_idx] = 0.0; //AcReal(0.5)*TT_bound
                              //- (DCONST_REAL(AC_cp_sound) - DCONST_REAL(AC_cv_sound))
                              //*(density_buffer[dst_idx] - DCONST_REAL(AC_lnrho0));
}


// This boundary condion sets the edge point in the system to a specific value.
// In this case, the sound speed. At the outflow boundary
// This way a constant value at the boundary is defined. The ghost zones are
// instead for the purpose of defining the behaviour of the derivatives. 
__global__ void
_set_csconst_xbot(const int3 start, const int3 end, AcReal* entropy_buffer, AcReal* density_buffer) 
{
    //const int i_dst = start.x + threadIdx.x + blockIdx.x * blockDim.x;
    //const int j_dst = start.y + threadIdx.y + blockIdx.y * blockDim.y;
    //const int k_dst = start.z + threadIdx.z + blockIdx.z * blockDim.z;
    //const int dst_idx      = DEVICE_VTXBUF_IDX(i_dst, j_dst, k_dst);
    int i_dst, j_dst, k_dst, dst_idx;
    get_dst_index(start, &i_dst, &j_dst, &k_dst, &dst_idx);

    //Skip threads which are not at valid boundaries.
    if (filter_outbound(end, i_dst, j_dst, k_dst)) return;
    if (choose_negxbound_point(i_dst, j_dst, k_dst)) return;

    _calc_csconst(dst_idx, entropy_buffer, density_buffer);

}

// This boundary condion sets the edge point in the system to a specific value.
// In this case, the sound speed. At the inflow boundaries. 
__global__ void
_set_csconst_xzin(const int3 start, const int3 end, AcReal* entropy_buffer, AcReal* density_buffer) 
{
    
    int i_dst, j_dst, k_dst, dst_idx;
    get_dst_index(start, &i_dst, &j_dst, &k_dst, &dst_idx);

    //Skip threads which are not at valid boundaries.
    if (filter_outbound(end, i_dst, j_dst, k_dst)) return;
    if ( (choose_posxbound_point(i_dst, j_dst, k_dst)) &&
         (choose_negzbound_point(i_dst, j_dst, k_dst)) && 
         (choose_poszbound_point(i_dst, j_dst, k_dst)) ) return;

    _calc_csconst(dst_idx, entropy_buffer, density_buffer);

}

// This boundary condion sets the edge point in the system to a specific value.
// In this case, the sound speed. At the inflow boundaries. 
__global__ void
_set_csconst_xin(const int3 start, const int3 end, AcReal* entropy_buffer, AcReal* density_buffer) 
{
    
    int i_dst, j_dst, k_dst, dst_idx;
    get_dst_index(start, &i_dst, &j_dst, &k_dst, &dst_idx);

    //Skip threads which are not at valid boundaries.
    if (filter_outbound(end, i_dst, j_dst, k_dst)) return;
    if ( (choose_posxbound_point(i_dst, j_dst, k_dst)) ) return;

    _calc_csconst(dst_idx, entropy_buffer, density_buffer);

}


// Calculate inflow velocity at a coordinate point
__device__ void
_calc_inflow_velocity(const int dst_idx, const AcReal xx, const AcReal zz, AcReal* uux_buffer, AcReal* uuy_buffer, AcReal* uuz_buffer)
{

    const AcReal delx = xx - DCONST_REAL(AC_star_pos_x);
    const AcReal delz = zz - DCONST_REAL(AC_star_pos_z);

    //TODO: Figure out isthis needed. Now a placeholder.
    //tanhz = fabs(tanh(zz/DCONST_REAL(AC_trans)));
    const AcReal tanhz = 1.0;
                    
    const AcReal RR     = sqrt(delx*delx + delz*delz);
    const AcReal veltot = DCONST_REAL(AC_sq2GM_star)/sqrt(RR); //Free fall velocity
    
    //Normal velocity components
    const AcReal uu_x = - veltot*(delx/RR);  
    const AcReal uu_z = - veltot*(delz/RR);

    //Take into account either the top or bottom direction if the inflow angle is transformed in some way.  
    if (delz >= 0.0) {
        uux_buffer[dst_idx] = ( uu_x*cos(DCONST_REAL(AC_angl_uu)) - uu_z*sin(DCONST_REAL(AC_angl_uu)) )*tanhz;
        uuz_buffer[dst_idx] = ( uu_x*sin(DCONST_REAL(AC_angl_uu)) + uu_z*cos(DCONST_REAL(AC_angl_uu)) )*tanhz;
    } else {        
        uux_buffer[dst_idx] = ( uu_x*cos(DCONST_REAL(AC_angl_uu)) + uu_z*sin(DCONST_REAL(AC_angl_uu)) )*tanhz;
        uuz_buffer[dst_idx] = (-uu_x*sin(DCONST_REAL(AC_angl_uu)) + uu_z*cos(DCONST_REAL(AC_angl_uu)) )*tanhz;
    }
    uuy_buffer[dst_idx] = AcReal(0.0) ;
}


//TODO: Make inflow xtrapolation based on the free fall profile 
__device__ void
_extrapolate_inflow_velocity_xonly(const int dst_idx, const int edge_idx, const AcReal xx, 
                                   const AcReal xx_edge, const AcReal zz, AcReal* uux_buffer, 
                                   AcReal* uuy_buffer, AcReal* uuz_buffer)
{

    const AcReal delx      = xx - DCONST_REAL(AC_star_pos_x);
    const AcReal delx_edge = xx_edge - DCONST_REAL(AC_star_pos_x);
                    

    const AcReal RR              = sqrt(delx*delx);
    const AcReal RR_edge         = sqrt(delx_edge*delx_edge);
    const AcReal RR_rel          = sqrt(RR_edge/RR); // 1/sqrt(R) scaling

    const AcReal veltot          = uux_buffer[edge_idx]*RR_rel;
    const AcReal veltot_freefall = -DCONST_REAL(AC_sq2GM_star)/sqrt(RR); //Free fall velocity
    
    //Normal velocity components
    //Set the roof to free fall velocity
    AcReal uu_x;
    if (veltot >= veltot_freefall) {
        uu_x = veltot; 
        //if (uux_buffer[edge_idx] < 0.0) printf("uux_buffer[%i] %e  veltot %e veltot_freefall %e RR_rel %e \n", edge_idx, uux_buffer[edge_idx], veltot, veltot_freefall, RR_rel);
        //if (veltot < uux_buffer[edge_idx]) printf("uux_buffer[%i] %e  veltot %e veltot_freefall %e RR_rel %e \n", edge_idx, uux_buffer[edge_idx], veltot, veltot_freefall, RR_rel);
        //if (veltot > uux_buffer[edge_idx]) printf("uux_buffer[%i] RR %e RR_edge %e RR_rel %e \n", edge_idx, RR, RR_edge, RR_rel);
    } else { 
        uu_x = veltot_freefall; 
        //printf("%i %e %e \n", dst_idx, veltot, veltot_freefall);
    }

    //Take into account either the top or bottom direction if the inflow angle is transformed in some way.  
    uux_buffer[dst_idx]    = uu_x;
    uuz_buffer[dst_idx]    = AcReal(0.0);
    uuy_buffer[dst_idx]    = AcReal(0.0);

}


__device__ void
_calc_inflow_velocity_xonly(const int dst_idx, const AcReal xx, const AcReal zz, AcReal* uux_buffer, AcReal* uuy_buffer, AcReal* uuz_buffer)
{

    const AcReal delx = xx - DCONST_REAL(AC_star_pos_x);
                    
    const AcReal RR     = sqrt(delx*delx);
    const AcReal veltot = DCONST_REAL(AC_sq2GM_star)/sqrt(RR); //Free fall velocity
    
    //Normal velocity components
    const AcReal uu_x = - veltot;  

    //Take into account either the top or bottom direction if the inflow angle is transformed in some way.  
    uux_buffer[dst_idx] = uu_x;
    uuz_buffer[dst_idx] = AcReal(0.0);
    uuy_buffer[dst_idx] = AcReal(0.0) ;
}


// Set inflow velocity based on free-fall at both vertical and horizontal boundaries. 
__global__ void
_set_uinflow_xzin(const int3 start, const int3 end, AcReal* uux_buffer, AcReal* uuy_buffer, AcReal* uuz_buffer)
{
    
    int i_dst, j_dst, k_dst, dst_idx;
    get_dst_index(start, &i_dst, &j_dst, &k_dst, &dst_idx);

    const AcReal xx = DCONST_REAL(AC_dsx) * AcReal(i_dst) - DCONST_REAL(AC_xorig);
    const AcReal zz = DCONST_REAL(AC_dsz) * AcReal(k_dst) - DCONST_REAL(AC_zorig);

    //Skip threads which are not at valid boundaries.
    if (filter_outbound(end, i_dst, j_dst, k_dst)) return;
    if ( (choose_posxbound_point(i_dst, j_dst, k_dst)) &&
         (choose_negzbound_point(i_dst, j_dst, k_dst)) && 
         (choose_poszbound_point(i_dst, j_dst, k_dst)) ) return;

    _calc_inflow_velocity(dst_idx, xx, zz, uux_buffer, uuy_buffer, uuz_buffer);

} 

// Set inflow velocity based on free-fall only at horizontal boundary. 
__global__ void
_set_uinflow_xin(const int3 start, const int3 end, AcReal* uux_buffer, AcReal* uuy_buffer, AcReal* uuz_buffer, AcReal* lnrho_buffer)
{
    
    int i_dst, j_dst, k_dst, dst_idx;
    get_dst_index(start, &i_dst, &j_dst, &k_dst, &dst_idx);

    //Skip threads which are not at valid boundaries.
    if (filter_outbound(end, i_dst, j_dst, k_dst)) return;
    //Fix the whole boundary
    //if ( (choose_posxbound_point(i_dst, j_dst, k_dst)) ) return;
    
    if (filter_allbut_posxbound(i_dst, j_dst, k_dst)) return;

    int i_edge, edge_idx, src_idx;
    _sym_indexing_xz(&edge_idx, &src_idx, i_dst, j_dst, k_dst);
    i_edge = DCONST_INT(AC_nx_max)-1;

    const AcReal xx      = DCONST_REAL(AC_dsx) * AcReal(i_dst)  - DCONST_REAL(AC_xorig);
    const AcReal zz      = DCONST_REAL(AC_dsz) * AcReal(k_dst)  - DCONST_REAL(AC_zorig);
    const AcReal xx_edge = DCONST_REAL(AC_dsx) * AcReal(i_edge) - DCONST_REAL(AC_xorig);

    //Free fall
    ////_calc_inflow_velocity_xonly(dst_idx, xx, zz, uux_buffer, uuy_buffer, uuz_buffer);

    //Basic momentum from boundcond
    const AcReal delx = xx - DCONST_REAL(AC_star_pos_x);
    const AcReal RR     = sqrt(delx*delx);
    const AcReal vel_freefall = -DCONST_REAL(AC_sq2GM_star)/sqrt(RR); //Free fall velocity

    //Extrapolation scheme but only if inflow
    if (uux_buffer[edge_idx] <= AcReal(0.0)) { 
        _extrapolate_inflow_velocity_xonly(dst_idx, edge_idx, xx, xx_edge, zz, uux_buffer, uuy_buffer, uuz_buffer);

        //uux_buffer[dst_idx] = AcReal(-0.001);
        //uuz_buffer[dst_idx] = AcReal(0.0);
        //uuy_buffer[dst_idx] = AcReal(0.0);
    } else {
        uux_buffer[dst_idx] = AcReal(0.0);
        uuz_buffer[dst_idx] = AcReal(0.0);
        uuy_buffer[dst_idx] = AcReal(0.0);

        uux_buffer[edge_idx] = AcReal(0.0);
    }
          
    //Simple linear interpolation for density scaling.
    lnrho_buffer[dst_idx] = (DCONST_REAL(AC_lnrho_edge) - DCONST_REAL(AC_ampl_lnrho))*(uux_buffer[dst_idx]/vel_freefall) + DCONST_REAL(AC_ampl_lnrho);    
   
}


///////////////////////////////////////
// Antisymmetric boundary conditions //
///////////////////////////////////////



//Set constant pressure at the inflow boundary it manage the "ramming effect" causing wiggles.
// WARNING: NOT TESTED
__global__ void
_inflow_set_der_xtop(const int3 start, const int3 end, const AcReal der_value, AcReal* vertex_buffer)
{
    int i_dst, j_dst, k_dst, dst_idx;
    get_dst_index(start, &i_dst, &j_dst, &k_dst, &dst_idx);

    //Skip threads which are not at valid boundaries.
    if (filter_outbound(end, i_dst, j_dst, k_dst)) return;
    if (filter_inbound(i_dst, j_dst, k_dst)) return;
    if (filter_ybound(i_dst, j_dst, k_dst)) return;
    if (filter_zbound(i_dst, j_dst, k_dst)) return;
    if (filter_negxbound(i_dst, j_dst, k_dst)) return; 

    //Get the correct, symmetric indices for the boundary compurtation
    int edge_idx, src_idx;
    _sym_indexing_xz(&edge_idx, &src_idx, i_dst, j_dst, k_dst);

    int i_edge = DCONST_INT(AC_nx_max)-1;
    int i_diff = i_edge - i_dst;

    const AcReal xx = DCONST_REAL(AC_dsx) * AcReal(i_edge + i_diff) - DCONST_REAL(AC_xorig);
    const AcReal xx_edge = DCONST_REAL(AC_dsx) * AcReal(i_edge) - DCONST_REAL(AC_xorig);

    vertex_buffer[dst_idx] =  vertex_buffer[src_idx] + (AcReal(2.0)*(xx_edge-xx))*der_value;

}


__device__ void
_rel_antisym_general(int i_dst, int j_dst, int k_dst, int dst_idx, AcReal* vertex_buffer)
{
    //Get the correct, symmetric indices for the boundary compurtation
    int edge_idx, src_idx;
    _sym_indexing_xz(&edge_idx, &src_idx, i_dst, j_dst, k_dst);

    AcReal sgn = -1.0; //Relative anti-symmetric
    vertex_buffer[dst_idx] = sgn*vertex_buffer[src_idx] + AcReal(2.0)*vertex_buffer[edge_idx];       
}

__device__ void
_sym_antisym_general(int i_dst, int j_dst, int k_dst, int dst_idx, AcReal* vertex_buffer, AcReal sgn)
{
    //Get the correct, symmetric indices for the boundary compurtation
    int edge_idx, src_idx;
    _sym_indexing_xz(&edge_idx, &src_idx, i_dst, j_dst, k_dst);

    vertex_buffer[dst_idx] = sgn*vertex_buffer[src_idx];       
}

__global__ void
_rel_antisym_xbot(const int3 start, const int3 end, AcReal* vertex_buffer)
{
    int i_dst, j_dst, k_dst, dst_idx;
    get_dst_index(start, &i_dst, &j_dst, &k_dst, &dst_idx);

    //Skip threads which are not at valid boundaries.
    if (filter_outbound(end, i_dst, j_dst, k_dst)) return;
    if (filter_inbound(i_dst, j_dst, k_dst)) return;
    if (filter_ybound(i_dst, j_dst, k_dst)) return;
    if (filter_zbound(i_dst, j_dst, k_dst)) return;
    if (filter_posxbound(i_dst, j_dst, k_dst)) return; 

    _rel_antisym_general(i_dst, j_dst, k_dst, dst_idx, vertex_buffer);

    //printf(" %i %i %i %i %e %i %i  \n", i_dst, j_dst, k_dst, dst_idx, vertex_buffer[dst_idx], 
    //                                    i_dst + j_dst*DCONST_INT(AC_mx) + k_dst*DCONST_INT(AC_mx)*DCONST_INT(AC_my), 
    //                                    DCONST_INT(AC_mx)*DCONST_INT(AC_my)*DCONST_INT(AC_mz) );

}

__global__ void
_rel_antisym_xtop(const int3 start, const int3 end, AcReal* vertex_buffer)
{
    int i_dst, j_dst, k_dst, dst_idx;
    get_dst_index(start, &i_dst, &j_dst, &k_dst, &dst_idx);

    //Skip threads which are not at valid boundaries.
    if (filter_outbound(end, i_dst, j_dst, k_dst)) return;
    if (filter_inbound(i_dst, j_dst, k_dst)) return;
    if (filter_ybound(i_dst, j_dst, k_dst)) return;
    if (filter_zbound(i_dst, j_dst, k_dst)) return;
    if (filter_negxbound(i_dst, j_dst, k_dst)) return; 

    _rel_antisym_general(i_dst, j_dst, k_dst, dst_idx, vertex_buffer);

}

__global__ void
_rel_antisym_zbot(const int3 start, const int3 end, AcReal* vertex_buffer)
{
    int i_dst, j_dst, k_dst, dst_idx;
    get_dst_index(start, &i_dst, &j_dst, &k_dst, &dst_idx);

    //Skip threads which are not at valid boundaries.
    if (filter_outbound(end, i_dst, j_dst, k_dst)) return;
    if (filter_inbound(i_dst, j_dst, k_dst)) return;
    if (filter_ybound(i_dst, j_dst, k_dst)) return;
    if (filter_xbound(i_dst, j_dst, k_dst)) return;
    if (filter_poszbound(i_dst, j_dst, k_dst)) return; 
    
    _rel_antisym_general(i_dst, j_dst, k_dst, dst_idx, vertex_buffer);

}

__global__ void
_rel_antisym_ztop(const int3 start, const int3 end, AcReal* vertex_buffer)
{
    int i_dst, j_dst, k_dst, dst_idx;
    get_dst_index(start, &i_dst, &j_dst, &k_dst, &dst_idx);

    //Skip threads which are not at valid boundaries.
    if (filter_outbound(end, i_dst, j_dst, k_dst)) return;
    if (filter_inbound(i_dst, j_dst, k_dst)) return;
    if (filter_ybound(i_dst, j_dst, k_dst)) return;
    if (filter_xbound(i_dst, j_dst, k_dst)) return;
    if (filter_negzbound(i_dst, j_dst, k_dst)) return; 
    
    _rel_antisym_general(i_dst, j_dst, k_dst, dst_idx, vertex_buffer);

}  

__global__ void
_antisym_xbot(const int3 start, const int3 end, AcReal* vertex_buffer)
{
    int i_dst, j_dst, k_dst, dst_idx;
    get_dst_index(start, &i_dst, &j_dst, &k_dst, &dst_idx);

    //Skip threads which are not at valid boundaries.
    if (filter_outbound(end, i_dst, j_dst, k_dst)) return;
    if (filter_inbound(i_dst, j_dst, k_dst)) return;
    if (filter_ybound(i_dst, j_dst, k_dst)) return;
    if (filter_zbound(i_dst, j_dst, k_dst)) return;
    if (filter_posxbound(i_dst, j_dst, k_dst)) return; 

    _sym_antisym_general(i_dst, j_dst, k_dst, dst_idx, vertex_buffer, AcReal(-1.0));

}

__global__ void
_antisym_xtop(const int3 start, const int3 end, AcReal* vertex_buffer)
{
    int i_dst, j_dst, k_dst, dst_idx;
    get_dst_index(start, &i_dst, &j_dst, &k_dst, &dst_idx);

    //Skip threads which are not at valid boundaries.
    if (filter_outbound(end, i_dst, j_dst, k_dst)) return;
    if (filter_inbound(i_dst, j_dst, k_dst)) return;
    if (filter_ybound(i_dst, j_dst, k_dst)) return;
    if (filter_zbound(i_dst, j_dst, k_dst)) return;
    if (filter_negxbound(i_dst, j_dst, k_dst)) return; 

    _sym_antisym_general(i_dst, j_dst, k_dst, dst_idx, vertex_buffer, AcReal(-1.0));

}

__global__ void
_antisym_zbot(const int3 start, const int3 end, AcReal* vertex_buffer)
{
    int i_dst, j_dst, k_dst, dst_idx;
    get_dst_index(start, &i_dst, &j_dst, &k_dst, &dst_idx);

    //Skip threads which are not at valid boundaries.
    if (filter_outbound(end, i_dst, j_dst, k_dst)) return;
    if (filter_inbound(i_dst, j_dst, k_dst)) return;
    if (filter_ybound(i_dst, j_dst, k_dst)) return;
    if (filter_xbound(i_dst, j_dst, k_dst)) return;
    if (filter_poszbound(i_dst, j_dst, k_dst)) return; 
    
    _sym_antisym_general(i_dst, j_dst, k_dst, dst_idx, vertex_buffer, AcReal(-1.0));

}

__global__ void
_antisym_ztop(const int3 start, const int3 end, AcReal* vertex_buffer)
{
    int i_dst, j_dst, k_dst, dst_idx;
    get_dst_index(start, &i_dst, &j_dst, &k_dst, &dst_idx);

    //Skip threads which are not at valid boundaries.
    if (filter_outbound(end, i_dst, j_dst, k_dst)) return;
    if (filter_inbound(i_dst, j_dst, k_dst)) return;
    if (filter_ybound(i_dst, j_dst, k_dst)) return;
    if (filter_xbound(i_dst, j_dst, k_dst)) return;
    if (filter_negzbound(i_dst, j_dst, k_dst)) return; 
    
    _sym_antisym_general(i_dst, j_dst, k_dst, dst_idx, vertex_buffer, AcReal(-1.0));

}  

 
__global__ void
_sym_xbot(const int3 start, const int3 end, AcReal* vertex_buffer)
{
    int i_dst, j_dst, k_dst, dst_idx;
    get_dst_index(start, &i_dst, &j_dst, &k_dst, &dst_idx);

    //Skip threads which are not at valid boundaries.
    if (filter_outbound(end, i_dst, j_dst, k_dst)) return;
    if (filter_inbound(i_dst, j_dst, k_dst)) return;
    if (filter_ybound(i_dst, j_dst, k_dst)) return;
    if (filter_zbound(i_dst, j_dst, k_dst)) return;
    if (filter_posxbound(i_dst, j_dst, k_dst)) return; 

    _sym_antisym_general(i_dst, j_dst, k_dst, dst_idx, vertex_buffer, AcReal(1.0));

}

__global__ void
_sym_xtop(const int3 start, const int3 end, AcReal* vertex_buffer)
{
    int i_dst, j_dst, k_dst, dst_idx;
    get_dst_index(start, &i_dst, &j_dst, &k_dst, &dst_idx);

    //Skip threads which are not at valid boundaries.
    if (filter_outbound(end, i_dst, j_dst, k_dst)) return;
    if (filter_inbound(i_dst, j_dst, k_dst)) return;
    if (filter_ybound(i_dst, j_dst, k_dst)) return;
    if (filter_zbound(i_dst, j_dst, k_dst)) return;
    if (filter_negxbound(i_dst, j_dst, k_dst)) return; 

    _sym_antisym_general(i_dst, j_dst, k_dst, dst_idx, vertex_buffer, AcReal(1.0));

}

__global__ void
_sym_zbot(const int3 start, const int3 end, AcReal* vertex_buffer)
{
    int i_dst, j_dst, k_dst, dst_idx;
    get_dst_index(start, &i_dst, &j_dst, &k_dst, &dst_idx);

    //Skip threads which are not at valid boundaries.
    if (filter_outbound(end, i_dst, j_dst, k_dst)) return;
    if (filter_inbound(i_dst, j_dst, k_dst)) return;
    if (filter_ybound(i_dst, j_dst, k_dst)) return;
    if (filter_xbound(i_dst, j_dst, k_dst)) return;
    if (filter_poszbound(i_dst, j_dst, k_dst)) return; 
    
    _sym_antisym_general(i_dst, j_dst, k_dst, dst_idx, vertex_buffer, AcReal(1.0));

}

__global__ void
_sym_ztop(const int3 start, const int3 end, AcReal* vertex_buffer)
{
    int i_dst, j_dst, k_dst, dst_idx;
    get_dst_index(start, &i_dst, &j_dst, &k_dst, &dst_idx);

    //Skip threads which are not at valid boundaries.
    if (filter_outbound(end, i_dst, j_dst, k_dst)) return;
    if (filter_inbound(i_dst, j_dst, k_dst)) return;
    if (filter_ybound(i_dst, j_dst, k_dst)) return;
    if (filter_xbound(i_dst, j_dst, k_dst)) return;
    if (filter_negzbound(i_dst, j_dst, k_dst)) return; 
    
    _sym_antisym_general(i_dst, j_dst, k_dst, dst_idx, vertex_buffer, AcReal(1.0));

}  

//Outflow boundary conditions: 
//Symmetric conditon when velocity vector points outside the box.
//Antisymmetric condition when velocity points inside.
__global__ void
_uu_outflow_xbot(const int3 start, const int3 end, AcReal* uux_buffer, AcReal* uuy_buffer, AcReal* uuz_buffer)     
{
    int i_dst, j_dst, k_dst, dst_idx;
    get_dst_index(start, &i_dst, &j_dst, &k_dst, &dst_idx);

    //Skip threads which are not at valid boundaries.
    if (filter_outbound(end, i_dst, j_dst, k_dst)) return;
    if (filter_allbut_negxbound(i_dst, j_dst, k_dst)) return;
    
    //Get the correct, symmetric indices for the boundary compurtation
    int edge_idx, src_idx;
    _sym_indexing_xz(&edge_idx, &src_idx, i_dst, j_dst, k_dst);

    AcReal sgnx;
    if (uux_buffer[edge_idx] <= 0.0) {  
        sgnx = 1.0; //Symmetric
    } else {
        sgnx = -1.0; //Antisymmetric
        //TODO: This might have synchronization problem.
        uux_buffer[edge_idx] = 0.0;  //Antisymmetric condition requires value to vanish at the boundary. TODO: This might  
    }
    AcReal sgny = 1.0, sgnz = 1.0; //Symmetric

    uux_buffer[dst_idx] = sgnx*uux_buffer[src_idx];       
    uuy_buffer[dst_idx] = sgny*uuy_buffer[src_idx];       
    uuz_buffer[dst_idx] = sgnz*uuz_buffer[src_idx];       


} 

//Inflow boundary conditions Pencil Code style: 
//Symmetric conditon when velocity vector points inside the box.
//Antisymmetric condition when velocity points outside.
__global__ void
_uu_inflow_simple_xbot(const int3 start, const int3 end, AcReal* uux_buffer, AcReal* uuy_buffer, AcReal* uuz_buffer)     
{
    int i_dst, j_dst, k_dst, dst_idx;
    get_dst_index(start, &i_dst, &j_dst, &k_dst, &dst_idx);

    //Skip threads which are not at valid boundaries.
    if (filter_outbound(end, i_dst, j_dst, k_dst)) return;
    if (filter_allbut_posxbound(i_dst, j_dst, k_dst)) return;

    //Get the correct, symmetric indices for the boundary compurtation
    int edge_idx, src_idx;
    _sym_indexing_xz(&edge_idx, &src_idx, i_dst, j_dst, k_dst);

    AcReal sgnx;
    if (uux_buffer[edge_idx] <= 0.0) {  
        sgnx = 1.0; //Symmetric
    } else {
        sgnx = -1.0; //Antisymmetric
        //TODO: This might have synchronization problem.
        uux_buffer[edge_idx] = 0.0;  //Antisymmetric condition requires value to vanish at the boundary. 
    }
    AcReal sgny = 1.0, sgnz = 1.0; //Symmetric

    uux_buffer[dst_idx] = sgnx*uux_buffer[src_idx];       
    uuy_buffer[dst_idx] = sgny*uuy_buffer[src_idx];       
    uuz_buffer[dst_idx] = sgnz*uuz_buffer[src_idx];       


} 

 

//////////////////////////////////
// Periodic boundary conditions //
//////////////////////////////////

// This boundary condition is only peridic in y-direction and will require
// other boundaries to determine x and z direction. Build for the vedge model
// in mind. Use with discretion. 
__global__ void
_y_periodic_boundconds(const int3 start, const int3 end, AcReal* vertex_buffer)
{
    const int i_dst = start.x + threadIdx.x + blockIdx.x * blockDim.x;
    const int j_dst = start.y + threadIdx.y + blockIdx.y * blockDim.y;
    const int k_dst = start.z + threadIdx.z + blockIdx.z * blockDim.z;

    //Skip threads which are not at valid boundaries.
    if (filter_outbound(end, i_dst, j_dst, k_dst)) return;
    if (filter_inbound(i_dst, j_dst, k_dst)) return;
    if (filter_xbound(i_dst, j_dst, k_dst)) return;
    if (filter_zbound(i_dst, j_dst, k_dst)) return;

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

    const int src_idx      = DEVICE_VTXBUF_IDX(i_src, j_src, k_src);
    const int dst_idx      = DEVICE_VTXBUF_IDX(i_dst, j_dst, k_dst);
    vertex_buffer[dst_idx] = vertex_buffer[src_idx];     
}




// This boundary condition is only peridic in y-direction and will require
// other boundaries to determine x direction. Build for the vedge model
// teasting in mind. Use with discretion. 
__global__ void
_yz_periodic_boundconds(const int3 start, const int3 end, AcReal* vertex_buffer)
{
    const int i_dst = start.x + threadIdx.x + blockIdx.x * blockDim.x;
    const int j_dst = start.y + threadIdx.y + blockIdx.y * blockDim.y;
    const int k_dst = start.z + threadIdx.z + blockIdx.z * blockDim.z;

    //Skip threads which are not at valid boundaries.
    if (filter_outbound(end, i_dst, j_dst, k_dst)) return;
    if (filter_inbound(i_dst, j_dst, k_dst)) return;
    if (filter_xbound(i_dst, j_dst, k_dst)) return;

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

    const int src_idx      = DEVICE_VTXBUF_IDX(i_src, j_src, k_src);
    const int dst_idx      = DEVICE_VTXBUF_IDX(i_dst, j_dst, k_dst);
    vertex_buffer[dst_idx] = vertex_buffer[src_idx];     
}

//Bundle of boundary conditions for xy inflow
void
xz_inflow_boundconds(const cudaStream_t stream, const dim3& tpb, const dim3 bpg,
                     const int3& start, const int3& end, VertexBufferArray d_buffer)
{
    _set_density_xzin<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_LNRHO]);                                               ERRCHK_CUDA_KERNEL();    
    //_set_csconst_xzin<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_ENTROPY], d_buffer.in[VTXBUF_LNRHO]);                  ERRCHK_CUDA_KERNEL();    
    _set_uinflow_xzin<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_UUX], d_buffer.in[VTXBUF_UUY], d_buffer.in[VTXBUF_UUZ]); ERRCHK_CUDA_KERNEL();    

    _rel_antisym_xtop<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_LNRHO]); ERRCHK_CUDA_KERNEL();   
    _rel_antisym_xtop<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_UUX]);     ERRCHK_CUDA_KERNEL();    
    _rel_antisym_xtop<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_UUY]);     ERRCHK_CUDA_KERNEL();    
    _rel_antisym_xtop<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_UUZ]);     ERRCHK_CUDA_KERNEL();    

    _rel_antisym_zbot<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_LNRHO]); ERRCHK_CUDA_KERNEL();   
    _rel_antisym_zbot<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_UUX]);     ERRCHK_CUDA_KERNEL();    
    _rel_antisym_zbot<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_UUY]);     ERRCHK_CUDA_KERNEL();    
    _rel_antisym_zbot<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_UUZ]);     ERRCHK_CUDA_KERNEL();    

    _rel_antisym_ztop<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_LNRHO]); ERRCHK_CUDA_KERNEL();   
    _rel_antisym_ztop<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_UUX]);     ERRCHK_CUDA_KERNEL();    
    _rel_antisym_ztop<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_UUY]);     ERRCHK_CUDA_KERNEL();    
    _rel_antisym_ztop<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_UUZ]);     ERRCHK_CUDA_KERNEL();    
}

//Bundle of boundary conditions for x inflow
void
x_inflow_boundconds(const cudaStream_t stream, const dim3& tpb, const dim3 bpg, 
                    const int3& start, const int3& end, VertexBufferArray d_buffer)
{
    //_set_density_xin<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_LNRHO]);                                               ERRCHK_CUDA_KERNEL(); //OK!!!   
    _set_uinflow_xin<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_UUX], d_buffer.in[VTXBUF_UUY], d_buffer.in[VTXBUF_UUZ], d_buffer.in[VTXBUF_LNRHO]); ERRCHK_CUDA_KERNEL();    
    //_uu_inflow_simple_xbot<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_UUX], d_buffer.in[VTXBUF_UUY], d_buffer.in[VTXBUF_UUZ]); ERRCHK_CUDA_KERNEL();

    //acSynchronize();

    //_rel_antisym_xtop<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_LNRHO]); ERRCHK_CUDA_KERNEL(); //OK!!! 
    ////TODO: Corners have to be determined by some condintion. Otherwise weird values appear. 
    //_rel_antisym_xtop<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_UUX]); ERRCHK_CUDA_KERNEL(); //OK!!! 
    //_rel_antisym_xtop<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_UUY]); ERRCHK_CUDA_KERNEL(); //OK!!! 
    //_rel_antisym_xtop<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_UUZ]); ERRCHK_CUDA_KERNEL(); //OK!!! 

    //_rel_antisym_xtop<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_LNRHO]); ERRCHK_CUDA_KERNEL(); //OK!!! 
    ////TODO: Corners have to be determined by some condintion. Otherwise weird values appear. 
    //_sym_xtop<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_UUX]); ERRCHK_CUDA_KERNEL(); //OK!!! 
    //_sym_xtop<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_UUY]); ERRCHK_CUDA_KERNEL(); //OK!!! 
    //_sym_xtop<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_UUZ]); ERRCHK_CUDA_KERNEL(); //OK!!! 
    //_rel_antisym_xtop<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_UUX]); ERRCHK_CUDA_KERNEL(); //OK!!! 
    //_rel_antisym_xtop<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_UUY]); ERRCHK_CUDA_KERNEL(); //OK!!! 
    //_rel_antisym_xtop<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_UUZ]); ERRCHK_CUDA_KERNEL(); //OK!!! 



}

//Bundle of boundary conditions for outflow
void
x_outflow_boundconds(const cudaStream_t stream, const dim3& tpb, const dim3 bpg,
                    const int3& start, const int3& end, VertexBufferArray d_buffer)
{
   // _set_density_xout<<<bpg, tpb, 0, stream>>>(start,  end, d_buffer.in[VTXBUF_LNRHO]); ERRCHK_CUDA_KERNEL(); //OK!!! 
   //acSynchronize();

   //DO we need to take the gravity effects explicitly in account here? 

    //_rel_antisym_xbot<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_LNRHO]); ERRCHK_CUDA_KERNEL();  //OK!!!   
    _rel_antisym_xbot<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_LNRHO]); ERRCHK_CUDA_KERNEL();  //OK!!!   
    //TODO: Chenk that these actually work.  
    _uu_outflow_xbot<<< bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_UUX], d_buffer.in[VTXBUF_UUY], d_buffer.in[VTXBUF_UUZ]); ERRCHK_CUDA_KERNEL();     
    
}




/*

IMPORTANT NOTE! 

The above boundary conditions have ben written for the pseudodisk model by
mvaisala. At the moment therefore there two alternative approaches for the
antisymmetric boundary condition. These need to be adapted in a smart way.

Fortunately the kernels as they are do not interfere each other due to
differing naming conventions.

jpekkila's antisymmetric kernel

_antisymmetric_boundconds(...)

mvaisala's antisymmetric kernels: 

_rel_antisym_general(...)
_rel_antisym_xtop(...);    
_rel_antisym_zbot(...);    
_rel_antisym_ztop(...);    

*/





__global__ void
_periodic_boundconds(const int3 start, const int3 end, AcReal* vertex_buffer)
{
    const int i_dst = start.x + threadIdx.x + blockIdx.x * blockDim.x;
    const int j_dst = start.y + threadIdx.y + blockIdx.y * blockDim.y;
    const int k_dst = start.z + threadIdx.z + blockIdx.z * blockDim.z;

    // If within the start-end range (this allows threadblock dims that are not
    // divisible by end - start)
    if (i_dst >= end.x || j_dst >= end.y || k_dst >= end.z)
        return;

    //if (i_dst >= DCONST_INT(AC_mx) || j_dst >= DCONST_INT(AC_my) || k_dst >= DCONST_INT(AC_mz))
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

    const int src_idx      = DEVICE_VTXBUF_IDX(i_src, j_src, k_src);
    const int dst_idx      = DEVICE_VTXBUF_IDX(i_dst, j_dst, k_dst);
    vertex_buffer[dst_idx] = vertex_buffer[src_idx];    
}

void
periodic_boundconds(const cudaStream_t stream, const dim3& tpb, 
                    const int3& start, const int3& end, AcReal* vertex_buffer)
{
    const dim3 bpg((unsigned int)ceil((end.x - start.x) / (float)tpb.x),
                   (unsigned int)ceil((end.y - start.y) / (float)tpb.y),
                   (unsigned int)ceil((end.z - start.z) / (float)tpb.z));

    _periodic_boundconds<<<bpg, tpb, 0, stream>>>(start, end, vertex_buffer);
    ERRCHK_CUDA_KERNEL();
}


typedef enum {
    X_AXIS,
    Y_AXIS,
    Z_AXIS,
    NUM_AXES
} Axis;

__global__ void
_antisymmetric_boundconds(const int3 start, const int3 end, const Axis symmetry_axis, const AcReal base_value, AcReal* vertex_buffer)
{
    const int i_dst = start.x + threadIdx.x + blockIdx.x * blockDim.x;
    const int j_dst = start.y + threadIdx.y + blockIdx.y * blockDim.y;
    const int k_dst = start.z + threadIdx.z + blockIdx.z * blockDim.z;

    // If within the start-end range (this allows threadblock dims that are not
    // divisible by end - start)
    if (i_dst >= end.x || j_dst >= end.y || k_dst >= end.z)
        return;

    //if (i_dst >= DCONST_INT(AC_mx) || j_dst >= DCONST_INT(AC_my) || k_dst >= DCONST_INT(AC_mz))
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

    const int src_idx      = DEVICE_VTXBUF_IDX(i_src, j_src, k_src);
    const int dst_idx      = DEVICE_VTXBUF_IDX(i_dst, j_dst, k_dst);

    if (base_value >= 0) {
        vertex_buffer[dst_idx] = -vertex_buffer[src_idx];
    } else {
        int bound_idx = -1;
        if (symmetry_axis == X_AXIS) {
            int boundary_idx = 0;
            if (i_dst < STENCIL_ORDER/2)
                boundary_idx = STENCIL_ORDER/2;
            else
                boundary_idx = DCONST_INT(AC_nx_max) - 1;

            bound_idx = DEVICE_VTXBUF_IDX(boundary_idx, j_src, k_src);
        } else if (symmetry_axis == Y_AXIS) {
            int boundary_idx = 0;
            if (j_dst < STENCIL_ORDER/2)
                boundary_idx = STENCIL_ORDER/2;
            else
                boundary_idx = DCONST_INT(AC_ny_max) - 1;

            bound_idx = DEVICE_VTXBUF_IDX(i_src, boundary_idx, k_src);
        } else { // symmetry_axis == Z_AXIS
            int boundary_idx = 0;
            if (k_dst < STENCIL_ORDER/2)
                boundary_idx = STENCIL_ORDER/2;
            else
                boundary_idx = DCONST_INT(AC_nz_max) - 1;

            bound_idx = DEVICE_VTXBUF_IDX(i_src, j_src, boundary_idx);
        }
        vertex_buffer[dst_idx] = -(vertex_buffer[src_idx] - vertex_buffer[bound_idx]) + vertex_buffer[bound_idx];
    }
}

void
antisymmetric_boundconds(const cudaStream_t stream, const dim3& tpb, 
                    const int3& start, const int3& end, const Axis symmetry_axis, const AcReal base_value, AcReal* vertex_buffer)
{
    const dim3 bpg((unsigned int)ceil((end.x - start.x) / (float)tpb.x),
                   (unsigned int)ceil((end.y - start.y) / (float)tpb.y),
                   (unsigned int)ceil((end.z - start.z) / (float)tpb.z));

    _antisymmetric_boundconds<<<bpg, tpb, 0, stream>>>(start, end, symmetry_axis, base_value, vertex_buffer);
    ERRCHK_CUDA_KERNEL();
}


//Here we attempt to construct the boundary condition for the vedge setup.
void 
wedge_boundconds(const cudaStream_t stream, const dim3& tpb,
                 const int3& start, const int3& end, VertexBufferArray d_buffer)
{
    const dim3 bpg((unsigned int)ceil((end.x - start.x) / (float)tpb.x),
                   (unsigned int)ceil((end.y - start.y) / (float)tpb.y),
                   (unsigned int)ceil((end.z - start.z) / (float)tpb.z));

    //Y direction is always periodic
    // Repeat fo all buffers
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
#if B_VELTYPE == 1
        _y_periodic_boundconds<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[i]); 
#else
        _yz_periodic_boundconds<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[i]);
#endif
        ERRCHK_CUDA_KERNEL();
    }


    //NOTE: Testing first now with fixed value antisymmetric boundary conditions. 
    //Conditions for inflow boundary 
#if B_VELTYPE == 1
    xz_inflow_boundconds(stream, tpb, bpg, start, end, d_buffer);
#else
    x_inflow_boundconds(stream, tpb, bpg, start, end, d_buffer);
#endif

    //Conditions for outflow boundary
    x_outflow_boundconds(stream, tpb, bpg, start, end, d_buffer);
    
    //Conditions for magnetic field 
    //USING PERIODIC WHILE DOING BASIC HYDRO TESTING.
    _periodic_boundconds<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_AX]); ERRCHK_CUDA_KERNEL();
    _periodic_boundconds<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_AY]); ERRCHK_CUDA_KERNEL();
    _periodic_boundconds<<<bpg, tpb, 0, stream>>>(start, end, d_buffer.in[VTXBUF_AZ]); ERRCHK_CUDA_KERNEL();


}






































