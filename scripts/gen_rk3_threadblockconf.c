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
 * \brief Generates a threadblock config file for RK3 using the given parameters.
 *
 *
 */
#include <stdio.h>
#include <stdlib.h>

#include <assert.h>

const char* rk3_threadblockconf_path = "rk3_threadblock.conf";

int
write_to_file(int threads_x, int threads_y, int threads_z, int elems_per_thread, int launch_bound)
{
    FILE* fp;
    fp = fopen(rk3_threadblockconf_path, "w");

    if (fp != NULL) {
        fprintf(fp, "#define RK_THREADS_X (%d)\n", threads_x);
        fprintf(fp, "#define RK_THREADS_Y (%d)\n", threads_y);
        fprintf(fp, "#define RK_THREADS_Z (%d)\n", threads_z);
        fprintf(fp, "#define RK_ELEMS_PER_THREAD (%d)\n", elems_per_thread);
        fprintf(fp, "#define RK_LAUNCH_BOUND_MIN_BLOCKS (%d)\n", launch_bound);
        fclose(fp);
        return EXIT_SUCCESS;
    }
    return EXIT_FAILURE;
}

// Takes arguments and writes them into a file
// RK_THREADS_X, RK_THREADS_Y, RK_THREADS_Z, RK_ELEMS_PER_THREAD, RK_LAUNCH_BOUND_MIN_BLOCKS
int
main(int argc, char* argv[])
{
    assert(argc == 6);

    return write_to_file(atoi(argv[1]), atoi(argv[2]),atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
}
