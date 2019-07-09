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
#include "run.h"

#include "config_loader.h"
#include "core/errchk.h"
#include "core/math_utils.h"
#include "model/host_forcing.h"
#include "model/host_memory.h"
#include "model/host_timestep.h"
#include "model/model_reduce.h"
#include "model/model_rk3.h"
#include "timer_hires.h"

#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

// Write all setting info into a separate ascii file. This is done to guarantee
// that we have the data specifi information in the thing, even though in
// principle these things are in the astaroth.conf.
static inline void
write_mesh_info(const AcMeshInfo* config)
{

    FILE* infotxt;

    infotxt = fopen("purge.sh", "w");
    fprintf(infotxt, "#!/bin/bash\n");
    fprintf(infotxt, "rm *.list *.mesh *.ts purge.sh\n");
    fclose(infotxt);

    infotxt = fopen("mesh_info.list", "w");

    // Total grid dimensions
    fprintf(infotxt, "int  AC_mx        %i \n", config->int_params[AC_mx]);
    fprintf(infotxt, "int  AC_my        %i \n", config->int_params[AC_my]);
    fprintf(infotxt, "int  AC_mz        %i \n", config->int_params[AC_mz]);

    // Bounds for the computational domain, i.e. nx_min <= i < nx_max
    fprintf(infotxt, "int  AC_nx_min    %i \n", config->int_params[AC_nx_min]);
    fprintf(infotxt, "int  AC_nx_max    %i \n", config->int_params[AC_nx_max]);
    fprintf(infotxt, "int  AC_ny_min    %i \n", config->int_params[AC_ny_min]);
    fprintf(infotxt, "int  AC_ny_max    %i \n", config->int_params[AC_ny_max]);
    fprintf(infotxt, "int  AC_nz_min    %i \n", config->int_params[AC_nz_min]);
    fprintf(infotxt, "int  AC_nz_max    %i \n", config->int_params[AC_nz_max]);

    // Spacing
    fprintf(infotxt, "real AC_dsx       %e \n", (double)config->real_params[AC_dsx]);
    fprintf(infotxt, "real AC_dsy       %e \n", (double)config->real_params[AC_dsy]);
    fprintf(infotxt, "real AC_dsz       %e \n", (double)config->real_params[AC_dsz]);
    fprintf(infotxt, "real AC_inv_dsx   %e \n", (double)config->real_params[AC_inv_dsx]);
    fprintf(infotxt, "real AC_inv_dsy   %e \n", (double)config->real_params[AC_inv_dsy]);
    fprintf(infotxt, "real AC_inv_dsz   %e \n", (double)config->real_params[AC_inv_dsz]);
    fprintf(infotxt, "real AC_dsmin     %e \n", (double)config->real_params[AC_dsmin]);

    /* Additional helper params */
    // Int helpers
    fprintf(infotxt, "int  AC_mxy       %i \n", config->int_params[AC_mxy]);
    fprintf(infotxt, "int  AC_nxy       %i \n", config->int_params[AC_nxy]);
    fprintf(infotxt, "int  AC_nxyz      %i \n", config->int_params[AC_nxyz]);

    // Real helpers
    fprintf(infotxt, "real AC_cs2_sound %e \n", (double)config->real_params[AC_cs2_sound]);
    fprintf(infotxt, "real AC_cv_sound  %e \n", (double)config->real_params[AC_cv_sound]);

    fclose(infotxt);
}

// This funtion writes a run state into a set of C binaries. For the sake of
// accuracy, all floating point numbers are to be saved in long double precision
// regardless of the choise of accuracy during runtime.
static inline void
save_mesh(const AcMesh& save_mesh, const int step, const AcReal t_step)
{
    FILE* save_ptr;

    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        const size_t n = AC_VTXBUF_SIZE(save_mesh.info);

        const char* buffername = vtxbuf_names[w];
        char cstep[11];
        char bin_filename[80] = "\0";

        // sprintf(bin_filename, "");

        sprintf(cstep, "%d", step);

        strcat(bin_filename, buffername);
        strcat(bin_filename, "_");
        strcat(bin_filename, cstep);
        strcat(bin_filename, ".mesh");

        printf("Savefile %s \n", bin_filename);

        save_ptr = fopen(bin_filename, "wb");

        // Start file with time stamp
        long double write_long_buf = (long double)t_step;
        fwrite(&write_long_buf, sizeof(long double), 1, save_ptr);
        // Grid data
        for (size_t i = 0; i < n; ++i) {
            const AcReal point_val     = save_mesh.vertex_buffer[VertexBufferHandle(w)][i];
            long double write_long_buf = (long double)point_val;
            fwrite(&write_long_buf, sizeof(long double), 1, save_ptr);
        }
        fclose(save_ptr);
    }
}

// This function prints out the diagnostic values to std.out and also saves and
// appends an ascii file to contain all the result.
static inline void
print_diagnostics(const int step, const AcReal dt, const AcReal t_step, FILE* diag_file)
{

    AcReal buf_rms, buf_max, buf_min;
    const int max_name_width = 16;

    // Calculate rms, min and max from the velocity vector field
    buf_max = acReduceVec(RTYPE_MAX, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ);
    buf_min = acReduceVec(RTYPE_MIN, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ);
    buf_rms = acReduceVec(RTYPE_RMS, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ);

    // MV: The ordering in the earlier version was wrong in terms of variable
    // MV: name and its diagnostics.
    printf("Step %d, t_step %.3e, dt %e s\n", step, double(t_step), double(dt));
    printf("  %*s: min %.3e,\trms %.3e,\tmax %.3e\n", max_name_width, "uu total", double(buf_min),
           double(buf_rms), double(buf_max));
    fprintf(diag_file, "%d %e %e %e %e %e ", step, double(t_step), double(dt), double(buf_min),
            double(buf_rms), double(buf_max));

    // Calculate rms, min and max from the variables as scalars
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        buf_max = acReduceScal(RTYPE_MAX, VertexBufferHandle(i));
        buf_min = acReduceScal(RTYPE_MIN, VertexBufferHandle(i));
        buf_rms = acReduceScal(RTYPE_RMS, VertexBufferHandle(i));

        printf("  %*s: min %.3e,\trms %.3e,\tmax %.3e\n", max_name_width, vtxbuf_names[i],
               double(buf_min), double(buf_rms), double(buf_max));
        fprintf(diag_file, "%e %e %e ", double(buf_min), double(buf_rms), double(buf_max));
    }

    fprintf(diag_file, "\n");
}

/*
    MV NOTE: At the moment I have no clear idea how to calculate magnetic
    diagnostic variables from grid. Vector potential measures have a limited
    value. TODO: Smart way to get brms, bmin and bmax.
*/

int
run_simulation(void)
{
    /* Parse configs */
    AcMeshInfo mesh_info;
    load_config(&mesh_info);

    AcMesh* mesh = acmesh_create(mesh_info);
    // TODO: This need to be possible to define in astaroth.conf
    acmesh_init_to(INIT_TYPE_GAUSSIAN_RADIAL_EXPL, mesh);

    acInit(mesh_info);
    acLoad(*mesh);

    FILE* diag_file;
    diag_file = fopen("timeseries.ts", "a");
    // TODO Get time from earlier state.
    AcReal t_step = 0.0;

    // Generate the title row.
    fprintf(diag_file, "step  t_step  dt  uu_total_min  uu_total_rms  uu_total_max  ");
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        fprintf(diag_file, "%s_min  %s_rms  %s_max  ", vtxbuf_names[i], vtxbuf_names[i],
                vtxbuf_names[i]);
    }

    fprintf(diag_file, "\n");

    write_mesh_info(&mesh_info);
    print_diagnostics(0, AcReal(.0), t_step, diag_file);

    acStore(mesh);
    save_mesh(*mesh, 0, t_step);

    const int max_steps      = mesh_info.int_params[AC_max_steps];
    const int save_steps     = mesh_info.int_params[AC_save_steps];
    const int bin_save_steps = mesh_info.int_params[AC_bin_steps]; // TODO Get from mesh_info

    AcReal bin_save_t = mesh_info.real_params[AC_bin_save_t];
    AcReal bin_crit_t = bin_save_t;

    /* initialize random seed: */
    srand(312256655);

    /* Step the simulation */
    for (int i = 1; i < max_steps; ++i) {
        const AcReal umax = acReduceVec(RTYPE_MAX, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ);
        const AcReal dt   = host_timestep(umax, mesh_info);

#if LFORCING
        const ForcingParams forcing_params = generateForcingParams(mesh_info);
        loadForcingParamsToDevice(forcing_params);
#endif

        acIntegrate(dt);

        t_step += dt;

        /* Save the simulation state and print diagnostics */
        if ((i % save_steps) == 0) {

            /*
                print_diagnostics() writes out both std.out printout from the
                results and saves the diagnostics into a table for ascii file
                timeseries.ts.
            */

            print_diagnostics(i, dt, t_step, diag_file);

            /*
                We would also might want an XY-average calculating funtion,
                which can be very useful when observing behaviour of turbulent
                simulations. (TODO)
            */
        }

        /* Save the simulation state and print diagnostics */
        if ((i % bin_save_steps) == 0 || t_step >= bin_crit_t) {

            /*
                This loop saves the data into simple C binaries which can be
                used for analysing the data snapshots closely.

                Saving simulation state should happen in a separate stage. We do
                not want to save it as often as diagnostics. The file format
                should IDEALLY be HDF5 which has become a well supported, portable and
                reliable data format when it comes to HPC applications.
                However, implementing it will have to for more simpler approach
                to function. (TODO?)
            */

            /*
                The updated mesh will be located on the GPU. Also all calls
                to the astaroth interface (functions beginning with ac*) are
                assumed to be asynchronous, so the meshes must be also synchronized
                before transferring the data to the CPU. Like so:

                acSynchronize();
                acStore(mesh);
            */
            acStore(mesh);

            save_mesh(*mesh, i, t_step);

            bin_crit_t += bin_save_t;
        }
    }

    //////Save the final snapshot
    ////acSynchronize();
    ////acStore(mesh);

    ////save_mesh(*mesh, , t_step);

    acQuit();
    acmesh_destroy(mesh);

    fclose(diag_file);

    return 0;
}
