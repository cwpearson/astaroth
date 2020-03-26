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

#include <mpi.h>

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

    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    acGridQuit();

    // Verify
    if (pid == 0) {
        acModelIntegrateStep(model, FLT_EPSILON);
        acMeshApplyPeriodicBounds(&model);

        acVerifyMesh(model, candidate);
        acMeshDestroy(&model);
        acMeshDestroy(&candidate);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
