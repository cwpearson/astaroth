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

#include <stdint.h>

typedef struct {
    uint64_t x, y, z;
} uint3_64;

static uint3_64
operator+(const uint3_64& a, const uint3_64& b)
{
    return (uint3_64){a.x + b.x, a.y + b.y, a.z + b.z};
}

static uint3_64
morton3D(const uint64_t pid)
{
    uint64_t i, j, k;
    i = j = k = 0;

    for (int bit = 0; bit <= 21; ++bit) {
        const uint64_t mask = 0x1l << 3 * bit;
        k |= ((pid & (mask << 0)) >> 2 * bit) >> 0;
        j |= ((pid & (mask << 1)) >> 2 * bit) >> 1;
        i |= ((pid & (mask << 2)) >> 2 * bit) >> 2;
    }

    return (uint3_64){i, j, k};
}

static uint3_64
decompose(const uint64_t target)
{
    // This is just so beautifully elegant. Complex and efficient decomposition
    // in just one line of code.
    uint3_64 p = morton3D(target - 1) + (uint3_64){1, 1, 1};

    ERRCHK_ALWAYS(p.x * p.y * p.z == target);
    return p;
}

int
main(int argc, char** argv)
{
    MPI_Init(NULL, NULL);
    int nprocs, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    // CPU alloc
    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);

    if (argc > 1) {
        if (argc == 4) {
            const int nx           = atoi(argv[1]);
            const int ny           = atoi(argv[2]);
            const int nz           = atoi(argv[3]);
            info.int_params[AC_nx] = nx;
            info.int_params[AC_ny] = ny;
            info.int_params[AC_nz] = nz;
            acUpdateBuiltinParams(&info);
            printf("Benchmark mesh dimensions: (%d, %d, %d)\n", nx, ny, nz);
        }
        else {
            fprintf(stderr, "Could not parse arguments. Usage: ./benchmark <nx> <ny> <nz>.\n");
            exit(EXIT_FAILURE);
        }
    }

    const TestType test = TEST_WEAK_SCALING;
    if (test == TEST_WEAK_SCALING) {
        uint3_64 decomp = decompose(nprocs);
        info.int_params[AC_nx] *= decomp.x;
        info.int_params[AC_ny] *= decomp.y;
        info.int_params[AC_nz] *= decomp.z;
    }

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
    acGridRandomize();

    /*
    AcMesh model;
    acMeshCreate(info, &model);
    acMeshRandomize(&model);
    acGridLoadMesh(STREAM_DEFAULT, model);
    */

    /*
    acGridLoadMesh(STREAM_DEFAULT, model);

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

    // Percentiles
    const size_t num_iters      = 100;
    const double nth_percentile = 0.90;
    std::vector<double> results; // ms
    results.reserve(num_iters);

    // Warmup
    for (size_t i = 0; i < num_iters / 10; ++i)
        acGridIntegrate(STREAM_DEFAULT, FLT_EPSILON);

    // Benchmark
    Timer t;
    const AcReal dt = FLT_EPSILON;

    for (size_t i = 0; i < num_iters; ++i) {
        acGridSynchronizeStream(STREAM_ALL);
        timer_reset(&t);
        acGridSynchronizeStream(STREAM_ALL);
        acGridIntegrate(STREAM_DEFAULT, dt);
        acGridSynchronizeStream(STREAM_ALL);
        results.push_back(timer_diff_nsec(t) / 1e6);
        acGridSynchronizeStream(STREAM_ALL);
    }

    if (!pid) {
        std::sort(results.begin(), results.end(),
                  [](const double& a, const double& b) { return a < b; });
        fprintf(stdout,
                "Integration step time %g ms (%gth "
                "percentile)--------------------------------------\n",
                results[nth_percentile * num_iters], 100 * nth_percentile);

        char path[4096] = "";
        sprintf(path, "%s_%d.csv", test == TEST_STRONG_SCALING ? "strong" : "weak", nprocs);

        FILE* fp = fopen(path, "a");
        ERRCHK_ALWAYS(fp);
        // Format
        // nprocs, min, 50th perc, 90th perc, max
        fprintf(fp, "%d, %g, %g, %g, %g\n", nprocs, results[0], results[0.5 * num_iters],
                results[nth_percentile * num_iters], results[num_iters - 1]);
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
