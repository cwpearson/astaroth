#pragma once

static __global__ void
kernel_symmetric_boundconds(const int3 start, const int3 end, AcReal* vtxbuf, const int3 bindex, const int sign)
{
    const int i_dst = start.x + threadIdx.x + blockIdx.x * blockDim.x;
    const int j_dst = start.y + threadIdx.y + blockIdx.y * blockDim.y;
    const int k_dst = start.z + threadIdx.z + blockIdx.z * blockDim.z;

    // If within the start-end range (this allows threadblock dims that are not
    // divisible by end - start)
    if (i_dst >= end.x || j_dst >= end.y || k_dst >= end.z)
        return;

    // If destination index is inside the computational domain, return since
    // the boundary conditions are only applied to the ghost zones
    if (i_dst >= DCONST(AC_nx_min) && i_dst < DCONST(AC_nx_max) && j_dst >= DCONST(AC_ny_min) &&
        j_dst < DCONST(AC_ny_max) && k_dst >= DCONST(AC_nz_min) && k_dst < DCONST(AC_nz_max))
        return;

    // Find the source index
    // Map to nx, ny, nz coordinates
    int i_src, j_src, k_src, boundloc;
    int bsize = STENCIL_ORDER/(int) 2;

    //if (bindex.x != 0)
    //{
    //    // Pick up the mirroring value.
    //    if ((i_dst < bsize) && ((bindex.x == 3) || (bindex.x ==1)))
    //    {
    //        boundloc = bsize;         //Location of central border point. 
    //        i_src = 2*boundloc - i_dst;
    //    } else if ((i_dst >= DCONST(AC_nx_min) - bsize) && ((bindex.x == 2) || (bindex.x ==1)))
    //    {
    //        boundloc = DCONST(AC_nx_min) - bsize - 1; //Location of central border point. 
    //        i_src = 2*boundloc - i_dst;
    //    }
    //} 
    //if (bindex.y != 0)
    //{
    //    // Pick up the mirroring value.
    //    if ((j_dst < bsize) && ((bindex.y == 3) || (bindex.y ==1)))
    //    {
    //        boundloc = bsize;         //Location of central border point. 
    //        j_src = 2*boundloc - j_dst;
    //    } else if ((j_dst >= DCONST(AC_nx_min) - bsize) && ((bindex.y == 2) || (bindex.y ==1)))
    //    {
    //        boundloc = DCONST(AC_ny_min) - bsize - 1; //Location of central border point. 
    //        i_src = 2*boundloc - j_dst;
    //    }
    //} 
    //if (bindex.z != 0)
    //{
    //    // Pick up the mirroring value.
    //    if ((k_dst < bsize) && ((bindex.z == 3) || (bindex.z ==1)))
    //    {
    //        boundloc = bsize;         //Location of central border point. 
    //        k_src = 2*boundloc - k_dst;
    //    } else if ((i_dst >= DCONST(AC_nz_min) - bsize) && ((bindex.z == 2) || (bindex.z ==1)))
    //    {
    //        boundloc = DCONST(AC_nz_min) - bsize - 1; //Location of central border point. 
    //        k_src = 2*boundloc - k_dst;
    //    }
    //}

    if (bindex.x < 0)
    { 

        // Pick up the mirroring value.
        if ((i_dst < bsize))
        {
            boundloc = bsize;         //Location of central border point. 
            i_src = 2*boundloc - i_dst;
        } else if ((i_dst >= DCONST(AC_nx_min) - bsize))
        {
            boundloc = DCONST(AC_nx_min) - bsize - 1; //Location of central border point. 
            i_src = 2*boundloc - i_dst;
        }
        
        // Pick up the mirroring value.
        if ((j_dst < bsize))
        {
            boundloc = bsize;         //Location of central border point. 
            j_src = 2*boundloc - j_dst;
        } else if ((j_dst >= DCONST(AC_nx_min) - bsize))
        {
            boundloc = DCONST(AC_ny_min) - bsize - 1; //Location of central border point. 
            i_src = 2*boundloc - j_dst;
        }
        
        // Pick up the mirroring value.
        if ((k_dst < bsize))
        {
            boundloc = bsize;         //Location of central border point. 
            k_src = 2*boundloc - k_dst;
        } else if ((i_dst >= DCONST(AC_nz_min) - bsize))
        {
            boundloc = DCONST(AC_nz_min) - bsize - 1; //Location of central border point. 
            k_src = 2*boundloc - k_dst;
        }

    }

    const int src_idx = DEVICE_VTXBUF_IDX(i_src, j_src, k_src);
    const int dst_idx = DEVICE_VTXBUF_IDX(i_dst, j_dst, k_dst);
    vtxbuf[dst_idx]   = sign*vtxbuf[src_idx]; // sign = 1 symmetric, sign = -1 antisymmetric
}


static __global__ void
kernel_periodic_boundconds(const int3 start, const int3 end, AcReal* vtxbuf)
{
    const int i_dst = start.x + threadIdx.x + blockIdx.x * blockDim.x;
    const int j_dst = start.y + threadIdx.y + blockIdx.y * blockDim.y;
    const int k_dst = start.z + threadIdx.z + blockIdx.z * blockDim.z;

    // If within the start-end range (this allows threadblock dims that are not
    // divisible by end - start)
    if (i_dst >= end.x || j_dst >= end.y || k_dst >= end.z)
        return;

    // if (i_dst >= DCONST(AC_mx) || j_dst >= DCONST(AC_my) || k_dst >= DCONST(AC_mz))
    //    return;

    // If destination index is inside the computational domain, return since
    // the boundary conditions are only applied to the ghost zones
    if (i_dst >= DCONST(AC_nx_min) && i_dst < DCONST(AC_nx_max) && j_dst >= DCONST(AC_ny_min) &&
        j_dst < DCONST(AC_ny_max) && k_dst >= DCONST(AC_nz_min) && k_dst < DCONST(AC_nz_max))
        return;

    // Find the source index
    // Map to nx, ny, nz coordinates
    int i_src = i_dst - DCONST(AC_nx_min);
    int j_src = j_dst - DCONST(AC_ny_min);
    int k_src = k_dst - DCONST(AC_nz_min);

    // Translate (s.t. the index is always positive)
    i_src += DCONST(AC_nx);
    j_src += DCONST(AC_ny);
    k_src += DCONST(AC_nz);

    // Wrap
    i_src %= DCONST(AC_nx);
    j_src %= DCONST(AC_ny);
    k_src %= DCONST(AC_nz);

    // Map to mx, my, mz coordinates
    i_src += DCONST(AC_nx_min);
    j_src += DCONST(AC_ny_min);
    k_src += DCONST(AC_nz_min);

    const int src_idx = DEVICE_VTXBUF_IDX(i_src, j_src, k_src);
    const int dst_idx = DEVICE_VTXBUF_IDX(i_dst, j_dst, k_dst);
    vtxbuf[dst_idx]   = vtxbuf[src_idx];
}

AcResult
acKernelPeriodicBoundconds(const cudaStream_t stream, const int3 start, const int3 end,
                           AcReal* vtxbuf)
{
    const dim3 tpb(8, 2, 8);
    const dim3 bpg((unsigned int)ceil((end.x - start.x) / (float)tpb.x),
                   (unsigned int)ceil((end.y - start.y) / (float)tpb.y),
                   (unsigned int)ceil((end.z - start.z) / (float)tpb.z));

    kernel_periodic_boundconds<<<bpg, tpb, 0, stream>>>(start, end, vtxbuf);
    ERRCHK_CUDA_KERNEL();
    return AC_SUCCESS;
}

AcResult 
acKernelGeneralBoundconds(const cudaStream_t stream, const int3 start, const int3 end,
                          AcReal* vtxbuf, const AcMeshInfo config, const int3 bindex)
{
    const dim3 tpb(8, 2, 8);
    const dim3 bpg((unsigned int)ceil((end.x - start.x) / (float)tpb.x),
                   (unsigned int)ceil((end.y - start.y) / (float)tpb.y),
                   (unsigned int)ceil((end.z - start.z) / (float)tpb.z));

    int3 bc_top = {config.int_params[AC_bc_type_top_x], config.int_params[AC_bc_type_top_y], 
                   config.int_params[AC_bc_type_top_z]};
    int3 bc_bot = {config.int_params[AC_bc_type_bot_x], config.int_params[AC_bc_type_bot_y], 
                   config.int_params[AC_bc_type_bot_z]};

    if (bc_top.x == AC_BOUNDCOND_SYMMETRIC) 
    {
        kernel_symmetric_boundconds<<<bpg, tpb, 0, stream>>>(start, end, vtxbuf, bindex,  1);
        ERRCHK_CUDA_KERNEL();
    } 
    else if (bc_bot.x == AC_BOUNDCOND_ANTISYMMETRIC) 
    {
        kernel_symmetric_boundconds<<<bpg, tpb, 0, stream>>>(start, end, vtxbuf, bindex, -1);
        ERRCHK_CUDA_KERNEL();
    } 
    else if (bc_bot.x == AC_BOUNDCOND_PERIODIC) 
    {
        kernel_periodic_boundconds<<<bpg, tpb, 0, stream>>>(start, end, vtxbuf);
        ERRCHK_CUDA_KERNEL();
    } 
    else 
    {
        printf("ERROR: Boundary condition not recognized!\n");
        printf("ERROR: bc_top = %i, %i, %i \n", bc_top.x, bc_top.y, bc_top.z);
        printf("ERROR: bc_bot = %i, %i, %i \n", bc_bot.x, bc_bot.y, bc_bot.z);

        return AC_FAILURE;
    }

    return AC_SUCCESS;
}
