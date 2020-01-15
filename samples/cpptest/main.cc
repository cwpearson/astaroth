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

// From Astaroth Utils
#include "src/utils/config_loader.h"
#include "src/utils/memory.h"
#include "src/utils/verification.h"

int
main(void)
{
    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);

    // Alloc
    AcMesh model, candidate;
    acMeshCreate(info, &model);
    acMeshCreate(info, &candidate);

    // Init
    acMeshRandomize(&model);
    acMeshApplyPeriodicBounds(&model);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    acInit(info);
    acLoad(model);
    acStore(&candidate);
    acQuit();
    ////////////////////////////////////////////////////////////////////////////////////////////////

    // Verify and destroy
    acVerifyMesh(model, candidate);
    acMeshDestroy(&model);
    acMeshDestroy(&candidate);

    return EXIT_SUCCESS;
}
