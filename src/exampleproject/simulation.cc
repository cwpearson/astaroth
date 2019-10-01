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
#include "src/utils/config_loader.h"
#include "src/utils/memory.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

int
run_simulation(const char* config_path)
{
    AcMeshInfo info;
    acLoadConfig(config_path, &info);

    AcMesh mesh;
    acMeshCreate(info, &mesh);
    acMeshClear(&mesh);

    acInit(info);
    acLoad(mesh);

    const size_t num_steps = 10;
    for (size_t i = 0; i < num_steps; ++i) {
        const AcReal dt = 1; // JP: TODO! Set timestep here!

        // JP: TODO! Make sure that AcMeshInfo parameters are properly initialized before calling
        // acIntegrate()
        // NANs indicate that either dt is too large or something was uninitalized
        acIntegrate(dt);
    }
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        printf("%s:\n", vtxbuf_names[i]);
        printf("\tmax: %g\n", (double)acReduceScal(RTYPE_MAX, VertexBufferHandle(i)));
        printf("\tmin: %g\n", (double)acReduceScal(RTYPE_MIN, VertexBufferHandle(i)));
        printf("\trms: %g\n", (double)acReduceScal(RTYPE_RMS, VertexBufferHandle(i)));
        printf("\texp rms: %g\n", (double)acReduceScal(RTYPE_RMS_EXP, VertexBufferHandle(i)));
    }
    acStore(&mesh);
    acQuit();

    acMeshDestroy(&mesh);
    return EXIT_SUCCESS;
}

int
main(int argc, char* argv[])
{
    printf("Args: \n");
    for (int i = 0; i < argc; ++i)
        printf("%d: %s\n", i, argv[i]);

    char* config_path;
    (argc == 3) ? config_path = strdup(argv[2]) : config_path = strdup(AC_DEFAULT_CONFIG);

    printf("Config path: %s\n", config_path);
    assert(config_path);
    run_simulation(config_path);
    free(config_path);
    return EXIT_SUCCESS;
}
