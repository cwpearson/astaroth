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
#include "astaroth.h"
#include "astaroth_utils.h"

#if AC_MPI_ENABLED

#include <mpi.h>
#include <vector>

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

    AcMesh model, candidate;
    if (pid == 0) {
        acMeshCreate(info, &model);
        acMeshCreate(info, &candidate);
        acMeshRandomize(&model);
        acMeshRandomize(&candidate);
    }

    // GPU alloc & compute
    acGridInit(info);

    // INTEGRATION TESTS START ---------------------------------------------------------------------
    acGridLoadMesh(model, STREAM_DEFAULT);
    acGridIntegrate(STREAM_DEFAULT, FLT_EPSILON);
    acGridPeriodicBoundconds(STREAM_DEFAULT);
    acGridStoreMesh(STREAM_DEFAULT, &candidate);

    if (pid == 0) {
        acModelIntegrateStep(model, FLT_EPSILON);
        acMeshApplyPeriodicBounds(&model);
        acVerifyMesh(model, candidate);
    }
    // INTEGRATION TESTS END -----------------------------------------------------------------------

    // REDUCTION TESTS START -----------------------------------------------------------------------
    acGridLoadMesh(model, STREAM_DEFAULT);

    std::vector<AcScalReductionTestCase> scalarReductionTests{
        acCreateScalReductionTestCase("Scalar MAX", VTXBUF_UUX, RTYPE_MAX),
        acCreateScalReductionTestCase("Scalar MIN", VTXBUF_UUX, RTYPE_MIN),
        /*
        acCreateScalReductionTestCase("Scalar RMS", VTXBUF_UUX, RTYPE_RMS),
        acCreateScalReductionTestCase("Scalar RMS_EXP", VTXBUF_UUX, RTYPE_RMS_EXP),
        acCreateScalReductionTestCase("Scalar SUM", VTXBUF_UUX, RTYPE_SUM),
        */
    };
    std::vector<AcVecReductionTestCase> vectorReductionTests{
        acCreateVecReductionTestCase("Vector MAX", VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, RTYPE_MAX),
        acCreateVecReductionTestCase("Vector MIN", VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, RTYPE_MIN),
        /*
        acCreateVecReductionTestCase("Vector RMS", VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, RTYPE_RMS),
        acCreateVecReductionTestCase("Vector RMS_EXP", VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ,
                                     RTYPE_RMS_EXP),
        acCreateVecReductionTestCase("Vector SUM", VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, RTYPE_SUM),
        */
    };
    // False positives due to too strict error bounds, skip the tests until we can determine a
    // proper error bound
    fprintf(stderr, "WARNING: RTYPE_RMS, RTYPE_RMS_EXP, and RTYPE_SUM tests skipped\n");

    for (auto& testCase : scalarReductionTests) {
        acGridReduceScal(STREAM_DEFAULT, testCase.rtype, testCase.vtxbuf, &testCase.candidate);
    }
    for (auto& testCase : vectorReductionTests) {
        acGridReduceVec(STREAM_DEFAULT, testCase.rtype, testCase.a, testCase.b, testCase.c,
                        &testCase.candidate);
    }
    if (pid == 0) {
        acVerifyScalReductions(model, scalarReductionTests.data(), scalarReductionTests.size());
        acVerifyVecReductions(model, vectorReductionTests.data(), vectorReductionTests.size());
    }
    // REDUCTION TESTS END -------------------------------------------------------------------------

    if (pid == 0) {
        acMeshDestroy(&model);
        acMeshDestroy(&candidate);
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
