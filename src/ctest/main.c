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
