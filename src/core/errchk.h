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
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// clang-format off
/*
 * =============================================================================
 * General error checking
 * =============================================================================
 */
#define ERROR(str) \
{ \
    time_t t; time(&t); \
    fprintf(stderr, "%s", ctime(&t)); \
    fprintf(stderr, "\tError in file %s line %d: %s\n", \
                    __FILE__, __LINE__, str); \
    fflush(stderr); \
    exit(EXIT_FAILURE); \
    abort(); \
}

#define WARNING(str) \
{ \
    time_t t; time(&t); \
    fprintf(stderr, "%s", ctime(&t)); \
    fprintf(stderr, "\tWarning in file %s line %d: %s\n", \
                    __FILE__, __LINE__, str); \
    fflush(stderr); \
}

// DO NOT REMOVE BRACKETS AROUND RETVAL. F.ex. if (!a < b) vs if (!(a < b)).
#define ERRCHK(retval)  { if (!(retval)) ERROR(#retval " was false"); }
#define WARNCHK(retval) { if (!(retval)) WARNING(#retval " was false"); }
#define ERRCHK_ALWAYS(retval) { if (!(retval)) ERROR(#retval " was false"); }

/*
 * =============================================================================
 * CUDA-specific error checking
 * =============================================================================
 */
#ifdef __CUDACC__
static inline void
cuda_assert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        time_t t; time(&t); \
        fprintf(stderr, "%s", ctime(&t)); \
        fprintf(stderr, "\tCUDA error in file %s line %d: %s\n", \
                        file, line, cudaGetErrorString(code)); \
        fflush(stderr); \

        if (abort)
            exit(code);
    }
}

#ifdef NDEBUG
    #undef ERRCHK
    #undef WARNCHK
    #define ERRCHK(params)
    #define WARNCHK(params)
    #define ERRCHK_CUDA(params) params;
    #define WARNCHK_CUDA(params) params;
    #define ERRCHK_CUDA_KERNEL() {}
#else
    #define ERRCHK_CUDA(params) { cuda_assert((params), __FILE__, __LINE__); }
    #define WARNCHK_CUDA(params) { cuda_assert((params), __FILE__, __LINE__, false); }

    #define ERRCHK_CUDA_KERNEL()                                               \
    {                                                                          \
        ERRCHK_CUDA(cudaPeekAtLastError());                                    \
        ERRCHK_CUDA(cudaDeviceSynchronize());                                  \
    }
    #endif

#endif

#define ERRCHK_CUDA_ALWAYS(params) { cuda_assert((params), __FILE__, __LINE__); }

#define ERRCHK_CUDA_KERNEL_ALWAYS()                                               \
{                                                                          \
    ERRCHK_CUDA_ALWAYS(cudaPeekAtLastError());                                    \
    ERRCHK_CUDA_ALWAYS(cudaDeviceSynchronize());                                  \
}
// clang-format on
