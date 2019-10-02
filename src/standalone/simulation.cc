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
#include "run.h"

#include "config_loader.h"
#include "model/host_forcing.h"
#include "model/host_memory.h"
#include "model/host_timestep.h"
#include "model/model_reduce.h"
#include "model/model_rk3.h"
#include "src/core/errchk.h"
#include "src/core/math_utils.h"
#include "timer_hires.h"

#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

// NEED TO BE DEFINED HERE. IS NOT NOTICED BY compile_acc call.
#define LFORCING (1)
#define LSINK (0)

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

    // Determine endianness
    unsigned int EE = 1;
    char *CC = (char*) &EE;
    const int endianness = (int) *CC; 
    // endianness = 0 -> big endian 
    // endianness = 1 -> little endian

    fprintf(infotxt, "size_t %s %lu \n", "AcRealSize", sizeof(AcReal));

    fprintf(infotxt, "int %s %i \n", "endian", endianness);

    // JP: this could be done shorter and with smaller chance for errors with the following
    // (modified from acPrintMeshInfo() in astaroth.cu)
    // MV: Now adapted into working condition. E.g. removed useless / harmful formatting. 

    for (int i = 0; i < NUM_INT_PARAMS; ++i)
        fprintf(infotxt, "int %s %d\n", intparam_names[i], config->int_params[i]);

    for (int i = 0; i < NUM_INT3_PARAMS; ++i)
        fprintf(infotxt, "int3 %s  %d %d %d\n", int3param_names[i], config->int3_params[i].x,
                                                                        config->int3_params[i].y,
                                                                        config->int3_params[i].z);

    for (int i = 0; i < NUM_REAL_PARAMS; ++i)
        fprintf(infotxt, "real %s %g\n", realparam_names[i], double(config->real_params[i]));

    for (int i = 0; i < NUM_REAL3_PARAMS; ++i)
        fprintf(infotxt, "real3 %s  %g %g %g\n", real3param_names[i],
                                                    double(config->real3_params[i].x),
                                                    double(config->real3_params[i].y),
                                                    double(config->real3_params[i].z));

    fclose(infotxt);
}

// This funtion writes a run state into a set of C binaries. 
static inline void
save_mesh(const AcMesh& save_mesh, const int step, const AcReal t_step)
{
    FILE* save_ptr;

    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        const size_t n = acVertexBufferSize(save_mesh.info);

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
        AcReal write_long_buf = (AcReal)t_step;
        fwrite(&write_long_buf, sizeof(AcReal), 1, save_ptr);
        // Grid data
        for (size_t i = 0; i < n; ++i) {
            const AcReal point_val     = save_mesh.vertex_buffer[VertexBufferHandle(w)][i];
            AcReal write_long_buf = (AcReal)point_val;
            fwrite(&write_long_buf, sizeof(AcReal), 1, save_ptr);
        }
        fclose(save_ptr);
    }
}

// This funtion reads a run state from a set of C binaries.
static inline void 
read_mesh(AcMesh& read_mesh, const int step, AcReal* t_step)
{
    FILE* read_ptr;

    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        const size_t n = acVertexBufferSize(read_mesh.info);

        const char* buffername = vtxbuf_names[w];
        char cstep[11];
        char bin_filename[80] = "\0";

        // sprintf(bin_filename, "");

        sprintf(cstep, "%d", step);

        strcat(bin_filename, buffername);
        strcat(bin_filename, "_");
        strcat(bin_filename, cstep);
        strcat(bin_filename, ".mesh");

        printf("Reading savefile %s \n", bin_filename);

        read_ptr = fopen(bin_filename, "rb");

        // Start file with time stamp
        size_t result;
        result = fread(t_step, sizeof(AcReal), 1, read_ptr);
        // Read grid data
        AcReal read_buf;
        for (size_t i = 0; i < n; ++i) {
            result = fread(&read_buf, sizeof(AcReal), 1, read_ptr);
            read_mesh.vertex_buffer[VertexBufferHandle(w)][i] = read_buf;
            if (int(result) != 1) {
                fprintf(stderr, "Reading error in %s, element %i\n", vtxbuf_names[w], int(i));
                fprintf(stderr, "Result = %i,  \n", int(result));
            }
        }
        fclose(read_ptr);
    }
}

// This function prints out the diagnostic values to std.out and also saves and
// appends an ascii file to contain all the result.
static inline void
print_diagnostics(const int step, const AcReal dt, const AcReal t_step, FILE* diag_file,
                  const AcReal sink_mass, const AcReal accreted_mass)
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

    if ((sink_mass >= AcReal(0.0)) || (accreted_mass >= AcReal(0.0))) {
        fprintf(diag_file, "%e %e ", double(sink_mass), double(accreted_mass));
    }

    fprintf(diag_file, "\n");
}

/*
    MV NOTE: At the moment I have no clear idea how to calculate magnetic
    diagnostic variables from grid. Vector potential measures have a limited
    value. TODO: Smart way to get brms, bmin and bmax.
*/

int
run_simulation(const char* config_path)
{
    /* Parse configs */
    AcMeshInfo mesh_info;
    load_config(config_path, &mesh_info);

    AcMesh* mesh = acmesh_create(mesh_info);
    // TODO: This need to be possible to define in astaroth.conf
    acmesh_init_to(INIT_TYPE_GAUSSIAN_RADIAL_EXPL, mesh);
    // acmesh_init_to(INIT_TYPE_SIMPLE_CORE, mesh); //Initial condition for a collapse test

#if LSINK
    vertex_buffer_set(VTXBUF_ACCRETION, 0.0, mesh);
#endif

    // Read old binary if we want to continue from an existing snapshot 
    // WARNING: Explicit specification of step needed!
    const int start_step = mesh_info.int_params[AC_start_step];
    AcReal t_step = 0.0;
    if (start_step > 0) { 
        read_mesh(*mesh, start_step, &t_step);
    }
 
    acInit(mesh_info);
    acLoad(*mesh);

    FILE* diag_file;
    diag_file = fopen("timeseries.ts", "a");

    // Generate the title row.
    if (start_step == 0) {
        fprintf(diag_file, "step  t_step  dt  uu_total_min  uu_total_rms  uu_total_max  ");
        for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
            fprintf(diag_file, "%s_min  %s_rms  %s_max  ", vtxbuf_names[i], vtxbuf_names[i],
                    vtxbuf_names[i]);
        }
    }
#if LSINK
    fprintf(diag_file, "sink_mass  accreted_mass  ");
#endif
    fprintf(diag_file, "\n");

    write_mesh_info(&mesh_info);

    if (start_step == 0) {
#if LSINK
        print_diagnostics(0, AcReal(.0), t_step, diag_file, mesh_info.real_params[AC_M_sink_init], 0.0);
#else
        print_diagnostics(0, AcReal(.0), t_step, diag_file, -1.0, -1.0);
#endif
    }

    acBoundcondStep();
    acStore(mesh);
    if (start_step == 0) {
        save_mesh(*mesh, 0, t_step);
    }

    const int max_steps      = mesh_info.int_params[AC_max_steps];
    const int save_steps     = mesh_info.int_params[AC_save_steps];
    const int bin_save_steps = mesh_info.int_params[AC_bin_steps]; 

    const AcReal max_time   = mesh_info.real_params[AC_max_time]; 
    const AcReal bin_save_t = mesh_info.real_params[AC_bin_save_t];
    AcReal bin_crit_t = bin_save_t;

    /* initialize random seed: */
    srand(312256655);

    /* Step the simulation */
    AcReal accreted_mass = 0.0;
    AcReal sink_mass     = 0.0;
    for (int i = start_step + 1; i < max_steps; ++i) {
        const AcReal umax = acReduceVec(RTYPE_MAX, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ);
        const AcReal dt   = host_timestep(umax, mesh_info);

#if LSINK

        const AcReal sum_mass = acReduceScal(RTYPE_SUM, VTXBUF_ACCRETION);
        accreted_mass         = accreted_mass + sum_mass;
        sink_mass             = 0.0;
        sink_mass             = mesh_info.real_params[AC_M_sink_init] + accreted_mass;
        acLoadDeviceConstant(AC_M_sink, sink_mass);
        vertex_buffer_set(VTXBUF_ACCRETION, 0.0, mesh);

        int on_off_switch;
        if (i < 1) {
            on_off_switch = 0; // accretion is off till certain amount of steps.
        }
        else {
            on_off_switch = 1;
        }
        acLoadDeviceConstant(AC_switch_accretion, on_off_switch);
#else
        accreted_mass = -1.0;
        sink_mass     = -1.0;
#endif

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

            print_diagnostics(i, dt, t_step, diag_file, sink_mass, accreted_mass);
#if LSINK
            printf("sink mass is: %.15e \n", double(sink_mass));
            printf("accreted mass is: %.15e \n", double(accreted_mass));
#endif
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

                The updated mesh will be located on the GPU. Also all calls
                to the astaroth interface (functions beginning with ac*) are
                assumed to be asynchronous, so the meshes must be also synchronized
                before transferring the data to the CPU. Like so:

                acBoundcondStep();
                acStore(mesh);
            */
            acBoundcondStep();
            acStore(mesh);

            save_mesh(*mesh, i, t_step);

            bin_crit_t += bin_save_t;
        }

        // End loop if max time reached. 
        if (max_time > AcReal(0.0)) {
            if (t_step >= max_time) {
                printf("Time limit reached! at t = %e \n", double(t_step));
                break;
                
            }
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
