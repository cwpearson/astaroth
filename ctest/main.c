#include <stdio.h>
#include <stdlib.h>

#include "astaroth.h"

int
main(void)
{
    AcMeshInfo info = {
        .int_params[AC_mx] = 128,
        .int_params[AC_my] = 64,
        .int_params[AC_mz] = 32,
    };
    acInit(info);
    acQuit();
    return EXIT_SUCCESS;
}
