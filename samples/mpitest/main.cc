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
    acGridLoadMesh(model, STREAM_DEFAULT);

    acGridIntegrate(STREAM_DEFAULT, FLT_EPSILON);
    acGridPeriodicBoundconds(STREAM_DEFAULT);

    // clang-format off
    // Define scalar reduction tests here
    std::vector<AcScalReductionTestCase> scalarReductionTests{
        AcScalReductionTestCase{"Scalar MAX",     VTXBUF_UUX, RTYPE_MAX,     0},
        AcScalReductionTestCase{"Scalar MIN",     VTXBUF_UUX, RTYPE_MIN,     0},
        AcScalReductionTestCase{"Scalar RMS",     VTXBUF_UUX, RTYPE_RMS,     0},
        AcScalReductionTestCase{"Scalar RMS_EXP", VTXBUF_UUX, RTYPE_RMS_EXP, 0},
        AcScalReductionTestCase{"Scalar SUM",     VTXBUF_UUX, RTYPE_SUM,     0}
    };
    // Define vector reduction tests here
    std::vector<AcVecReductionTestCase> vectorReductionTests{
        AcVecReductionTestCase{"Vector MAX",     VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, RTYPE_MAX,     0},
        AcVecReductionTestCase{"Vector MIN",     VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, RTYPE_MIN,     0},
        AcVecReductionTestCase{"Vector RMS",     VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, RTYPE_RMS,     0},
        AcVecReductionTestCase{"Vector RMS_EXP", VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, RTYPE_RMS_EXP, 0},
        AcVecReductionTestCase{"Vector SUM",     VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, RTYPE_SUM,     0}
    };
    // clang-format on

    for (auto& testCase : scalarReductionTests) {
        acGridReduceScal(STREAM_DEFAULT, testCase.rtype, testCase.vtxbuf, &testCase.candidate);
    }
    for (auto& testCase : vectorReductionTests) {
        acGridReduceVec(STREAM_DEFAULT, testCase.rtype, testCase.a, testCase.b, testCase.c,
                        &testCase.candidate);
    }

    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    acGridQuit();

    // Verify
    if (pid == 0) {
        acModelIntegrateStep(model, FLT_EPSILON);
        acMeshApplyPeriodicBounds(&model);

        acVerifyMesh(model, candidate);

        // Check reductions
        acVerifyScalReductions(model, scalarReductionTests.data(), scalarReductionTests.size());
        acVerifyVecReductions(model, vectorReductionTests.data(), vectorReductionTests.size());

        acMeshDestroy(&model);
        acMeshDestroy(&candidate);
    }

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
