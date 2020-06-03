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
 * \brief Functions for loading and updating AcMeshInfo.
 *
 */
#pragma once
#include "astaroth.h"

#ifdef __cplusplus
extern "C" {
#endif

 #include <stdbool.h>

typedef struct {
    VertexBufferHandle handle;
    AcReal model;
    AcReal candidate;
    long double abs_error;
    long double ulp_error;
    long double rel_error;
    AcReal maximum_magnitude;
    AcReal minimum_magnitude;
} Error;

/** TODO comment */
Error acGetError(AcReal model, AcReal candidate);

/** TODO comment */
bool printErrorToScreen(const Error error);

/** Loads data from the config file */
AcResult acLoadConfig(const char* config_path, AcMeshInfo* config);

/** Updates the built-in parameters based on nx, ny and nz */
AcResult acUpdateBuiltinParams(AcMeshInfo* config);

/** */
AcResult acMeshCreate(const AcMeshInfo mesh_info, AcMesh* mesh);

/** */
AcResult acMeshDestroy(AcMesh* mesh);

/** */
AcResult acMeshSet(const AcReal value, AcMesh* mesh);

/** */
AcResult acMeshRandomize(AcMesh* mesh);

/** */
AcResult acMeshApplyPeriodicBounds(AcMesh* mesh);

/** */
AcResult acMeshClear(AcMesh* mesh);

/** */
AcResult acModelIntegrateStep(AcMesh mesh, const AcReal dt);

/** TODO */
AcReal
acModelReduceScal(const AcMesh mesh, const ReductionType rtype, const VertexBufferHandle a);

/** TODO */
AcReal
acModelReduceVec(const AcMesh mesh, const ReductionType rtype, const VertexBufferHandle a,
                 const VertexBufferHandle b, const VertexBufferHandle c);

/** */
AcResult acVerifyMesh(const AcMesh model, const AcMesh candidate);

#ifdef __cplusplus
} // extern "C"
#endif
