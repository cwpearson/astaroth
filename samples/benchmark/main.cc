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
    Running: benchmark -np <num processes> <executable>
*/
#include "astaroth.h"
#include "astaroth_utils.h"

#include "errchk.h"
#include "timer_hires.h"

#if AC_MPI_ENABLED

#include <mpi.h>

#include <algorithm>
#include <string.h>
#include <vector>

typedef enum {
    TEST_STRONG_SCALING,
    TEST_WEAK_SCALING,
    NUM_TESTS,
} TestType;

int
main(void)
{
    MPI_Init(NULL, NULL);
    int nprocs, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    // CPU alloc
    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);

    const TestType test = TEST_STRONG_SCALING;
    if (test == TEST_WEAK_SCALING)
        info.int_params[AC_nz] *= nprocs;

    /*
    AcMesh model, candidate;
    if (pid == 0) {
        acMeshCreate(info, &model);
        acMeshCreate(info, &candidate);
        acMeshRandomize(&model);
        acMeshRandomize(&candidate);
    }*/

    // GPU alloc & compute
    acGridInit(info);
    /*
    acGridLoadMesh(model, STREAM_DEFAULT);

    acGridIntegrate(STREAM_DEFAULT, FLT_EPSILON);
    acGridPeriodicBoundconds(STREAM_DEFAULT);

    acGridStoreMesh(STREAM_DEFAULT, &candidate);

    // Verify
    if (pid == 0) {
        acModelIntegrateStep(model, FLT_EPSILON);
        acMeshApplyPeriodicBounds(&model);

        AcResult retval = acVerifyMesh(model, candidate);
        acMeshDestroy(&model);
        acMeshDestroy(&candidate);

        if (retval != AC_SUCCESS) {
            fprintf(stderr, "Failures found, benchmark invalid. Skipping\n");
            return EXIT_FAILURE;
        }
    }*/

    // Warmup
    for (size_t i = 0; i < 10; ++i)
        acGridIntegrate(STREAM_DEFAULT, FLT_EPSILON);

    // Benchmark
    Timer t;
    const AcReal dt             = FLT_EPSILON;
    const size_t num_iters      = 100;
    const double nth_percentile = 0.90;

    std::vector<double> results; // ms
    results.reserve(num_iters);

    for (size_t i = 0; i < num_iters; ++i) {
        acGridSynchronizeStream(STREAM_ALL);
        timer_reset(&t);
        acGridSynchronizeStream(STREAM_ALL);
        acGridIntegrate(STREAM_DEFAULT, dt);
        acGridSynchronizeStream(STREAM_ALL);
        results.push_back(timer_diff_nsec(t) / 1e6);
    }

    // Write benchmark to file
    if (!pid) {
        std::sort(results.begin(), results.end(),
                  [](const double& a, const double& b) { return a < b; });
        fprintf(stdout,
                "Integration step time %g ms (%gth "
                "percentile)--------------------------------------\n",
                results[nth_percentile * num_iters], 100 * nth_percentile);

        char path[4096] = "";
        if (test == TEST_STRONG_SCALING)
            strncpy(path, "strong_scaling.csv", sizeof(path));
        else if (test == TEST_WEAK_SCALING)
            strncpy(path, "weak_scaling.csv", sizeof(path));
        else
            ERROR("Invalid test type");

        FILE* fp = fopen(path, "a");
        ERRCHK_ALWAYS(fp);
        // Format
        // nprocs, measured (ms)
        fprintf(fp, "%d, %g\n", nprocs, results[nth_percentile * num_iters]);

        fclose(fp);
    }

    acGridQuit();
    MPI_Finalize();
    return EXIT_SUCCESS;
}

#else
int
main(void)
{
    printf("The library was built without MPI support, cannot run mpitest. Rebuild Astaroth with "
           "cmake -DMPI_ENABLED=ON .. to enable.\n");
    return EXIT_FAILURE;
}
#endif // AC_MPI_ENABLES
