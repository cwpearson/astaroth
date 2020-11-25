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
 * @file
 * \brief Brief info.
 *
 * Detailed info.
 *
 */
#include "astaroth_utils.h"

#include <stdint.h> // uint8_t, uint32_t
#include <string.h>

#include "errchk.h"

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
    printf("Config file path: %s\n", path);
    ERRCHK_ALWAYS(fp != NULL);

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

/**
\brief Loads data from astaroth.conf into a config struct.
\return AC_SUCCESS on success, AC_FAILURE if there are potentially uninitialized values.
*/
AcResult
acLoadConfig(const char* config_path, AcMeshInfo* config)
{
    int retval = AC_SUCCESS;
    ERRCHK_ALWAYS(config_path);

    // memset reads the second parameter as a byte even though it says int in
    // the function declaration
    memset(config, (uint8_t)0xFF, sizeof(*config));

    parse_config(config_path, config);
    acHostUpdateBuiltinParams(config);
#if AC_VERBOSE
    printf("###############################################################\n");
    printf("Config dimensions loaded:\n");
    acPrintMeshInfo(*config);
    printf("###############################################################\n");
#endif

    // sizeof(config) must be a multiple of 4 bytes for this to work
    ERRCHK_ALWAYS(sizeof(*config) % sizeof(uint32_t) == 0);
    for (size_t i = 0; i < sizeof(*config) / sizeof(uint32_t); ++i) {
        if (((uint32_t*)config)[i] == (uint32_t)0xFFFFFFFF) {
#if AC_VERBOSE
            fprintf(stderr, "Some config values may be uninitialized. "
                            "See that all are defined in astaroth.conf\n");
#endif
            retval = AC_FAILURE;
        }
    }
    return retval;
}
