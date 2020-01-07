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
#include "astaroth.h"
#include "common.cuh"

/*
AcResult acKernelPackData(const cudaStream_t stream, const AcReal* unpacked,
                          const int3 unpacked_start, const int3 packed_dimensions, AcReal* packed);

AcResult acKernelUnpackData(const cudaStream_t stream, const AcReal* packed,
                            const int3 packed_dimensions, const int3 unpacked_start,
                            AcReal* unpacked);
*/

typedef struct {
    int3 dims;
    AcReal* data;
} PackedData;

AcResult acKernelPackData(const cudaStream_t stream, const VertexBufferArray vba,
                          const int3 vba_start, PackedData packed);

AcResult acKernelUnpackData(const cudaStream_t stream, const PackedData packed,
                            const int3 vba_start, VertexBufferArray vba);

AcResult acKernelPackCorner(void);
AcResult acKernelUnpackCorner(void);

AcResult acKernelPackEdge(void);
AcResult acKernelUnpackEdge(void);

AcResult acKernelPackSide(void);
AcResult acKernelUnpackSide(void);
