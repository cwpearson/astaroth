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
#include <stdio.h>
#include <stdlib.h>

#include "astaroth.h"

int
main(void)
{
    AcMeshInfo info = {
        .int_params[AC_nx] = 128,
        .int_params[AC_ny] = 64,
        .int_params[AC_nz] = 32,
    };
    acInit(info);
    acIntegrate(0.1f);
    acQuit();
    return EXIT_SUCCESS;
}
