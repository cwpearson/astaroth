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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "run.h"
#include "src/core/errchk.h"

// Write all errors from stderr to an <errorlog_name> in the current working
// directory
static const bool write_log_to_a_file = false;
static const char* errorlog_name      = "error.log";

static void
errorlog_init(void)
{
    FILE* fp = freopen(errorlog_name, "w", stderr); // Log errors to a file
    if (!fp)
        perror("Error redirecting stderr to a file");
}

static void
errorlog_quit(void)
{
    fclose(stderr);

    // Print contents of the latest errorlog to screen
    FILE* fp = fopen(errorlog_name, "r");
    if (fp) {
        for (int c = getc(fp); c != EOF; c = getc(fp))
            putchar(c);
        fclose(fp);
    }
    else {
        perror("Error opening error log");
    }
}

int
main(int argc, char* argv[])
{
    if (write_log_to_a_file) {
        errorlog_init();
        atexit(errorlog_quit);
    }

    printf("Args: \n");
    for (int i = 0; i < argc; ++i)
        printf("%d: %s\n", i, argv[i]);

    char* config_path;
    (argc == 3) ? config_path = strdup(argv[2])
                : config_path = strdup(AC_DEFAULT_CONFIG);

    printf("Config path: %s\n", config_path);
    ERRCHK(config_path);

    if (argc == 1) {
        return run_renderer(config_path);
    }
    else if (argc == 2 || argc == 3) {
        if (strcmp(argv[1], "-t") == 0)
            return run_autotest(config_path);
        else if (strcmp(argv[1], "-b") == 0)
            return run_benchmark(config_path);
        else if (strcmp(argv[1], "-s") == 0)
            return run_simulation(config_path);
        else if (strcmp(argv[1], "-r") == 0)
            return run_renderer(config_path);
        else
            WARNING("Unrecognized option");
    }
    else {
        WARNING("Too many options given");
    }

    free(config_path);
    return EXIT_FAILURE;
}
