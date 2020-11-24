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
    Running: mpirun -np <num processes> <executable>
*/
#if AC_MPI_ENABLED
#include "astaroth.h"
#include "astaroth_utils.h"

#include <mpi.h>
#include <string.h>

#include "config_loader.h"
#include "errchk.h"
#include "host_forcing.h"
#include "host_memory.h"

#define fprintf(...)                                                                               \
    {                                                                                              \
        int tmppid;                                                                                \
        MPI_Comm_rank(MPI_COMM_WORLD, &tmppid);                                                    \
        if (!tmppid) {                                                                             \
            fprintf(__VA_ARGS__);                                                                  \
        }                                                                                          \
    }

#define printf(...)                                                                                \
    {                                                                                              \
        int tmppid;                                                                                \
        MPI_Comm_rank(MPI_COMM_WORLD, &tmppid);                                                    \
        if (!tmppid) {                                                                             \
            printf(__VA_ARGS__);                                                                   \
        }                                                                                          \
    }

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(*arr))

// NEED TO BE DEFINED HERE. IS NOT NOTICED BY compile_acc call.
#define LFORCING (0)
#define LSINK (0)

// Write all setting info into a separate ascii file. This is done to guarantee
// that we have the data specifi information in the thing, even though in
// principle these things are in the astaroth.conf.
static inline void
write_info(const AcMeshInfo* config)
{

    FILE* infotxt;

    infotxt = fopen("purge.sh", "w");
    fprintf(infotxt, "#!/bin/bash\n");
    fprintf(infotxt, "rm *.list *.mesh *.ts purge.sh\n");
    fclose(infotxt);

    infotxt = fopen("info.list", "w");

    // Determine endianness
    unsigned int EE      = 1;
    char* CC             = (char*)&EE;
    const int endianness = (int)*CC;
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
                config->int3_params[i].y, config->int3_params[i].z);

    for (int i = 0; i < NUM_REAL_PARAMS; ++i)
        fprintf(infotxt, "real %s %g\n", realparam_names[i], double(config->real_params[i]));

    for (int i = 0; i < NUM_REAL3_PARAMS; ++i)
        fprintf(infotxt, "real3 %s  %g %g %g\n", real3param_names[i],
                double(config->real3_params[i].x), double(config->real3_params[i].y),
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
            const AcReal point_val = save_mesh.vertex_buffer[VertexBufferHandle(w)][i];
            AcReal write_long_buf2 = (AcReal)point_val;
            fwrite(&write_long_buf2, sizeof(AcReal), 1, save_ptr);
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
// JP: MUST BE CALLED FROM PROC 0. Must be rewritten for multiple processes (this implementation
// has write race condition)
static inline void
print_diagnostics_host(const AcMesh mesh, const int step, const AcReal dt, const AcReal t_step,
                       FILE* diag_file, const AcReal sink_mass, const AcReal accreted_mass)
{

    AcReal buf_rms, buf_max, buf_min;
    const int max_name_width = 16;

    // Calculate rms, min and max from the velocity vector field
    buf_max = acHostReduceVec(mesh, RTYPE_MAX, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ);
    buf_min = acHostReduceVec(mesh, RTYPE_MIN, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ);
    buf_rms = acHostReduceVec(mesh, RTYPE_RMS, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ);

    // MV: The ordering in the earlier version was wrong in terms of variable
    // MV: name and its diagnostics.
    printf("Step %d, t_step %.3e, dt %e s\n", step, double(t_step), double(dt));
    printf("  %*s: min %.3e,\trms %.3e,\tmax %.3e\n", max_name_width, "uu total", double(buf_min),
           double(buf_rms), double(buf_max));
    fprintf(diag_file, "%d %e %e %e %e %e ", step, double(t_step), double(dt), double(buf_min),
            double(buf_rms), double(buf_max));

    // Calculate rms, min and max from the variables as scalars
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        buf_max = acHostReduceScal(mesh, RTYPE_MAX, VertexBufferHandle(i));
        buf_min = acHostReduceScal(mesh, RTYPE_MIN, VertexBufferHandle(i));
        buf_rms = acHostReduceScal(mesh, RTYPE_RMS, VertexBufferHandle(i));

        printf("  %*s: min %.3e,\trms %.3e,\tmax %.3e\n", max_name_width, vtxbuf_names[i],
               double(buf_min), double(buf_rms), double(buf_max));
        fprintf(diag_file, "%e %e %e ", double(buf_min), double(buf_rms), double(buf_max));
    }

    if ((sink_mass >= AcReal(0.0)) || (accreted_mass >= AcReal(0.0))) {
        fprintf(diag_file, "%e %e ", double(sink_mass), double(accreted_mass));
    }

    fprintf(diag_file, "\n");
}

// This function prints out the diagnostic values to std.out and also saves and
// appends an ascii file to contain all the result.
// JP: EXECUTES ON MULTIPLE GPUS, MUST BE CALLED FROM ALL PROCS
static inline void
print_diagnostics(const int step, const AcReal dt, const AcReal t_step, FILE* diag_file,
                  const AcReal sink_mass, const AcReal accreted_mass)
{

    AcReal buf_rms, buf_max, buf_min;
    const int max_name_width = 16;

    // Calculate rms, min and max from the velocity vector field
    acGridReduceVec(STREAM_DEFAULT, RTYPE_MAX, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, &buf_max);
    acGridReduceVec(STREAM_DEFAULT, RTYPE_MIN, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, &buf_min);
    acGridReduceVec(STREAM_DEFAULT, RTYPE_RMS, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, &buf_rms);

    // MV: The ordering in the earlier version was wrong in terms of variable
    // MV: name and its diagnostics.
    printf("Step %d, t_step %.3e, dt %e s\n", step, double(t_step), double(dt));
    printf("  %*s: min %.3e,\trms %.3e,\tmax %.3e\n", max_name_width, "uu total", double(buf_min),
           double(buf_rms), double(buf_max));
    fprintf(diag_file, "%d %e %e %e %e %e ", step, double(t_step), double(dt), double(buf_min),
            double(buf_rms), double(buf_max));

    // Calculate rms, min and max from the variables as scalars
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        acGridReduceScal(STREAM_DEFAULT, RTYPE_MAX, VertexBufferHandle(i), &buf_max);
        acGridReduceScal(STREAM_DEFAULT, RTYPE_MIN, VertexBufferHandle(i), &buf_min);
        acGridReduceScal(STREAM_DEFAULT, RTYPE_RMS, VertexBufferHandle(i), &buf_rms);

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

#include "math_utils.h"
AcReal
calc_timestep(const AcMeshInfo info)
{
    AcReal uumax;
    acGridReduceVec(STREAM_DEFAULT, RTYPE_MAX, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, &uumax);

    const long double cdt  = info.real_params[AC_cdt];
    const long double cdtv = info.real_params[AC_cdtv];
    // const long double cdts     = info.real_params[AC_cdts];
    const long double cs2_sound = info.real_params[AC_cs2_sound];
    const long double nu_visc   = info.real_params[AC_nu_visc];
    const long double eta       = info.real_params[AC_eta];
    const long double chi       = 0; // info.real_params[AC_chi]; // TODO not calculated
    const long double gamma     = info.real_params[AC_gamma];
    const long double dsmin     = info.real_params[AC_dsmin];

    // Old ones from legacy Astaroth
    // const long double uu_dt   = cdt * (dsmin / (uumax + cs_sound));
    // const long double visc_dt = cdtv * dsmin * dsmin / nu_visc;

    // New, closer to the actual Courant timestep
    // See Pencil Code user manual p. 38 (timestep section)
    const long double uu_dt   = cdt * dsmin / (fabsl(uumax) + sqrtl(cs2_sound + 0.0l));
    const long double visc_dt = cdtv * dsmin * dsmin / max(max(nu_visc, eta), max(gamma, chi));

    const long double dt = min(uu_dt, visc_dt);
    ERRCHK_ALWAYS(is_valid((AcReal)dt));
    return AcReal(dt);
}

int
main(int argc, char** argv)
{
    MPI_Init(NULL, NULL);
    int nprocs, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    /////////// Simple example START
    (void)argc; // Unused
    (void)argv; // Unused

    // Set random seed for reproducibility
    srand(321654987);

    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);
    load_config(AC_DEFAULT_CONFIG, &info);
    update_config(&info);

    AcMesh mesh;
    if (pid == 0) {
        acHostMeshCreate(info, &mesh);
        acmesh_init_to(INIT_TYPE_GAUSSIAN_RADIAL_EXPL, &mesh);
    }
    acGridInit(info);
    acGridLoadMesh(STREAM_DEFAULT, mesh);

    for (int t_step = 0; t_step < 100; ++t_step) {
        const AcReal dt = calc_timestep(info);
        acGridIntegrate(STREAM_DEFAULT, dt);

        if (1) { // Diag step
            AcReal uumin, uumax, uurms;
            acGridReduceVec(STREAM_DEFAULT, RTYPE_MIN, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, &uumin);
            acGridReduceVec(STREAM_DEFAULT, RTYPE_MAX, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, &uumax);
            acGridReduceVec(STREAM_DEFAULT, RTYPE_RMS, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, &uurms);

            printf("Step %d, dt: %g\n", t_step, (double)dt);
            printf("%*s: %.3e, %.3e, %.3e\n", 16, "UU", (double)uumin, (double)uumax,
                   (double)uurms);

            for (size_t vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
                AcReal scalmin, scalmax, scalrms;
                acGridReduceScal(STREAM_DEFAULT, RTYPE_MIN, (VertexBufferHandle)vtxbuf, &scalmin);
                acGridReduceScal(STREAM_DEFAULT, RTYPE_MAX, (VertexBufferHandle)vtxbuf, &scalmax);
                acGridReduceScal(STREAM_DEFAULT, RTYPE_RMS, (VertexBufferHandle)vtxbuf, &scalrms);

                printf("%*s: %.3e, %.3e, %.3e\n", 16, vtxbuf_names[vtxbuf], (double)scalmin,
                       (double)scalmax, (double)scalrms);
            }
        }
    }
    if (pid == 0)
        acHostMeshDestroy(&mesh);

    acGridQuit();
    /////////////// Simple example END

// JP: The following is directly from standalone/simulation.cc and modified to work with MPI
// However, not extensively tested
#if 0
    // Load config to AcMeshInfo
    AcMeshInfo info;
    if (argc > 1) {
        if (argc == 3 && (!strcmp(argv[1], "-c") || !strcmp(argv[1], "--config"))) {
            acLoadConfig(argv[2], &info);
            load_config(argv[2], &info);
            acHostUpdateBuiltinParams(&info);
        }
        else {
            printf("Usage: ./ac_run\n");
            printf("Usage: ./ac_run -c <config_path>\n");
            printf("Usage: ./ac_run --config <config_path>\n");
            return EXIT_FAILURE;
        }
    }
    else {
        acLoadConfig(AC_DEFAULT_CONFIG, &info);
        load_config(AC_DEFAULT_CONFIG, &info);
        acHostUpdateBuiltinParams(&info);
    }

    const int start_step     = info.int_params[AC_start_step];
    const int max_steps      = info.int_params[AC_max_steps];
    const int save_steps     = info.int_params[AC_save_steps];
    const int bin_save_steps = info.int_params[AC_bin_steps];

    const AcReal max_time   = info.real_params[AC_max_time];
    const AcReal bin_save_t = info.real_params[AC_bin_save_t];
    AcReal bin_crit_t       = bin_save_t;
    AcReal t_step           = 0.0;
    FILE* diag_file         = fopen("timeseries.ts", "a");
    ERRCHK_ALWAYS(diag_file);

    AcMesh mesh;
    ///////////////////////////////// PROC 0 BLOCK START ///////////////////////////////////////////
    if (pid == 0) {
        acHostMeshCreate(info, &mesh);
        // TODO: This need to be possible to define in astaroth.conf
        acmesh_init_to(INIT_TYPE_GAUSSIAN_RADIAL_EXPL, &mesh);
        // acmesh_init_to(INIT_TYPE_SIMPLE_CORE, mesh); //Initial condition for a collapse test

#if LSINK
        acVertexBufferSet(VTXBUF_ACCRETION, 0.0, &mesh);
#endif
        // Read old binary if we want to continue from an existing snapshot
        // WARNING: Explicit specification of step needed!
        if (start_step > 0) {
            read_mesh(mesh, start_step, &t_step);
        }

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

        write_info(&info);

        if (start_step == 0) {
#if LSINK
            print_diagnostics_host(mesh, 0, AcReal(.0), t_step, diag_file,
                                   info.real_params[AC_M_sink_init], 0.0);
#else
            print_diagnostics_host(mesh, 0, AcReal(.0), t_step, diag_file, -1.0, -1.0);
#endif
        }

        acHostMeshApplyPeriodicBounds(&mesh);
        if (start_step == 0) {
            save_mesh(mesh, 0, t_step);
        }
    }
    ////////////////////////////////// PROC 0 BLOCK END ////////////////////////////////////////////

    // Init GPU
    acGridInit(info);
    acGridLoadMesh(STREAM_DEFAULT, mesh);

    /* initialize random seed: */
    srand(312256655);

    /* Step the simulation */
    AcReal accreted_mass = 0.0;
    AcReal sink_mass     = 0.0;
    for (int i = start_step + 1; i < max_steps; ++i) {
        AcReal umax;
        acGridReduceVec(STREAM_DEFAULT, RTYPE_MAX, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, &umax);
        const AcReal dt = host_timestep(umax, info);

#if LSINK
        AcReal sum_mass;
        acGridReduceScal(STREAM_DEFAULT, RTYPE_SUM, VTXBUF_ACCRETION, &sum_mass);
        accreted_mass = accreted_mass + sum_mass;
        sink_mass     = 0.0;
        sink_mass     = info.real_params[AC_M_sink_init] + accreted_mass;
        acGridLoadScalarUniform(STREAM_DEFAULT, AC_M_sink, sink_mass);

        // JP: !!! WARNING !!! acVertexBufferSet operates in host memory. The mesh is
        // never loaded to device memory. Is this intended?
        if (pid == 0)
            acVertexBufferSet(VTXBUF_ACCRETION, 0.0, mesh);

        int on_off_switch;
        if (i < 1) {
            on_off_switch = 0; // accretion is off till certain amount of steps.
        }
        else {
            on_off_switch = 1;
        }
        acGridLoadScalarUniform(STREAM_DEFAULT, AC_switch_accretion, on_off_switch);
#else
        accreted_mass = -1.0;
        sink_mass = -1.0;
#endif

#if LFORCING
        const ForcingParams forcing_params = generateForcingParams(info);
        loadForcingParamsToGrid(forcing_params);
#endif

        acGridIntegrate(STREAM_DEFAULT, dt);

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
            acGridPeriodicBoundconds(STREAM_DEFAULT);
            acGridStoreMesh(STREAM_DEFAULT, &mesh);

            if (pid == 0)
                save_mesh(mesh, i, t_step);

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

    acGridQuit();
    if (pid == 0)
        acHostMeshDestroy(&mesh);
    fclose(diag_file);
#endif

    MPI_Finalize();
    return EXIT_SUCCESS;
}

#else
#include <stdio.h>
#include <stdlib.h>

int
main(void)
{
    printf("The library was built without MPI support, cannot run mpitest. Rebuild Astaroth with "
           "cmake -DMPI_ENABLED=ON .. to enable.\n");
    return EXIT_FAILURE;
}
#endif // AC_MPI_ENABLES
