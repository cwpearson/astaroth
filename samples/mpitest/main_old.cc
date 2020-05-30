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
    info.real_params[AC_inv_dsx]   = AcReal(1.0) / info.real_params[AC_dsx];
    info.real_params[AC_inv_dsy]   = AcReal(1.0) / info.real_params[AC_dsy];
    info.real_params[AC_inv_dsz]   = AcReal(1.0) / info.real_params[AC_dsz];
    info.real_params[AC_cs2_sound] = info.real_params[AC_cs_sound] * info.real_params[AC_cs_sound];

    AcMesh model, candidate;
    if (pid == 0) {
        acMeshCreate(info, &model);
        acMeshCreate(info, &candidate);
        acMeshRandomize(&model);
        acMeshRandomize(&candidate);
    }

    // GPU alloc & compute
    Grid grid;
    acGridCreateMPI(info, &grid);

    acGridLoadMeshMPI(grid, STREAM_DEFAULT, model);
    acGridSynchronizeStreamMPI(grid, STREAM_ALL);

    acGridIntegrateMPI(grid, FLT_EPSILON);
    acGridSynchronizeStreamMPI(grid, STREAM_ALL);
    acGridSynchronizeMeshMPI(grid, STREAM_DEFAULT);
    acGridSynchronizeStreamMPI(grid, STREAM_ALL);

    acGridStoreMeshMPI(grid, STREAM_DEFAULT, &candidate);
    acGridSynchronizeStreamMPI(grid, STREAM_ALL);

    acGridDestroyMPI(grid);

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
