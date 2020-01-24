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
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "astaroth.h"
#include "astaroth_utils.h"

int
main(void)
{
    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);
    // Some real params must be calculated (for the MHD case) // TODO DANGEROUS
    info.real_params[AC_inv_dsx]   = (AcReal)(1.0) / info.real_params[AC_dsx];
    info.real_params[AC_inv_dsy]   = (AcReal)(1.0) / info.real_params[AC_dsy];
    info.real_params[AC_inv_dsz]   = (AcReal)(1.0) / info.real_params[AC_dsz];
    info.real_params[AC_cs2_sound] = info.real_params[AC_cs_sound] * info.real_params[AC_cs_sound];

    // Alloc
    AcMesh model, candidate;
    acMeshCreate(info, &model);
    acMeshCreate(info, &candidate);

    // Init
    acMeshRandomize(&model);
    acMeshApplyPeriodicBounds(&model);

    // Verify that the mesh was loaded and stored correctly
    acInit(info);
    acLoad(model);
    acStore(&candidate);
    acVerifyMesh(model, candidate);

    // Attempt to integrate and check max and min
    printf("Integrating... ");
    acIntegrate(FLT_EPSILON);
    printf("Done.\nVTXBUF ranges after one integration step:\n");
    for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i)
        printf("\t%-15s... [%.3g, %.3g]\n", vtxbuf_names[i],
               (double)acReduceScal(RTYPE_MIN, (VertexBufferHandle)i),
               (double)acReduceScal(RTYPE_MAX, (VertexBufferHandle)i));

    // Destroy
    acQuit();
    acMeshDestroy(&model);
    acMeshDestroy(&candidate);

    puts("cpptest complete.");
    return EXIT_SUCCESS;
}