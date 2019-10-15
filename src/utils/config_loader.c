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

/**
 * @file
 * \brief Brief info.
 *
 * Detailed info.
 *
 */
#include "config_loader.h"

#include <assert.h>
#include <limits.h> // UINT_MAX
#include <math.h>
#include <stdint.h> // uint8_t, uint32_t
#include <stdio.h>  // print
#include <string.h> // memset
//#include "src/core/math_utils.h"

/**
 \brief Find the index of the keyword in names
 \return Index in range 0...n if the keyword is in names. -1 if the keyword was
 not found.
 */
static int
find_str(const char keyword[], const char* names[], const int n)
{
    for (int i = 0; i < n; ++i)
        if (!strcmp(keyword, names[i]))
            return i;

    return -1;
}

static void
parse_config(const char* path, AcMeshInfo* config)
{
    FILE* fp;
    fp = fopen(path, "r");
    // For knowing which .conf file will be used
    printf("Config file path: \n %s \n ", path);
    assert(fp != NULL);

    const size_t BUF_SIZE = 128;
    char keyword[BUF_SIZE];
    char value[BUF_SIZE];
    int items_matched;
    while ((items_matched = fscanf(fp, "%s = %s", keyword, value)) != EOF) {

        if (items_matched < 2)
            continue;

        int idx = -1;
        if ((idx = find_str(keyword, intparam_names, NUM_INT_PARAMS)) >= 0)
            config->int_params[idx] = atoi(value);
        else if ((idx = find_str(keyword, realparam_names, NUM_REAL_PARAMS)) >= 0)
            config->real_params[idx] = (AcReal)(atof(value));
    }

    fclose(fp);
}

AcResult
acUpdateConfig(AcMeshInfo* config)
{
    config->int_params[AC_mx] = config->int_params[AC_nx] + STENCIL_ORDER;
    ///////////// PAD TEST
    // config->int_params[AC_mx] = config->int_params[AC_nx] + STENCIL_ORDER + PAD_SIZE;
    ///////////// PAD TEST
    config->int_params[AC_my] = config->int_params[AC_ny] + STENCIL_ORDER;
    config->int_params[AC_mz] = config->int_params[AC_nz] + STENCIL_ORDER;

    // Bounds for the computational domain, i.e. nx_min <= i < nx_max
    config->int_params[AC_nx_min] = STENCIL_ORDER / 2;
    config->int_params[AC_nx_max] = config->int_params[AC_nx_min] + config->int_params[AC_nx];
    config->int_params[AC_ny_min] = STENCIL_ORDER / 2;
    config->int_params[AC_ny_max] = config->int_params[AC_ny] + STENCIL_ORDER / 2;
    config->int_params[AC_nz_min] = STENCIL_ORDER / 2;
    config->int_params[AC_nz_max] = config->int_params[AC_nz] + STENCIL_ORDER / 2;

    /*
    // DEPRECATED: Spacing
    // These do not have to be defined by empty projects any more.
    // These should be set only if stdderiv.h is included
    config->real_params[AC_inv_dsx] = (AcReal)(1.) / config->real_params[AC_dsx];
    config->real_params[AC_inv_dsy] = (AcReal)(1.) / config->real_params[AC_dsy];
    config->real_params[AC_inv_dsz] = (AcReal)(1.) / config->real_params[AC_dsz];
    */

    /* Additional helper params */
    // Int helpers
    config->int_params[AC_mxy]  = config->int_params[AC_mx] * config->int_params[AC_my];
    config->int_params[AC_nxy]  = config->int_params[AC_nx] * config->int_params[AC_ny];
    config->int_params[AC_nxyz] = config->int_params[AC_nxy] * config->int_params[AC_nz];

    return AC_SUCCESS;
}

/**
\brief Loads data from astaroth.conf into a config struct.
\return AC_SUCCESS on success, AC_FAILURE if there are potentially uninitialized values.
*/
AcResult
acLoadConfig(const char* config_path, AcMeshInfo* config)
{
    int retval = AC_SUCCESS;
    assert(config_path);

    // memset reads the second parameter as a byte even though it says int in
    // the function declaration
    memset(config, (uint8_t)0xFF, sizeof(*config));

    parse_config(config_path, config);
    acUpdateConfig(config);
#if VERBOSE_PRINTING // Defined in astaroth.h
    printf("###############################################################\n");
    printf("Config dimensions recalculated:\n");
    acPrintMeshInfo(*config);
    printf("###############################################################\n");
#endif

    // sizeof(config) must be a multiple of 4 bytes for this to work
    assert(sizeof(*config) % sizeof(uint32_t) == 0);
    for (size_t i = 0; i < sizeof(*config) / sizeof(uint32_t); ++i) {
        if (((uint32_t*)config)[i] == (uint32_t)0xFFFFFFFF) {
            fprintf(stderr, "Some config values may be uninitialized. "
                            "See that all are defined in astaroth.conf\n");
            retval = AC_FAILURE;
        }
    }
    return retval;
}
