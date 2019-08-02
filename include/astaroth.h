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
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "astaroth_defines.h"

#include "astaroth_device.h"
#include "astaroth_node.h"

/*
#include "astaroth_grid.h"
#define acInit(x) acGridInit(x)
#define acQuit() acGridQuit()
#define acLoad(x) acGridLoadMesh(STREAM_DEFAULT, x)
#define acReduceScal(x, y) acGridReduceScal(STREAM_DEFAULT, x, y)
#define acReduceVec(x, y, z, w) acGridReduceVec(STREAM_DEFAULT, x, y, z, w)
#define acBoundcondStep() acGridPeriodicBoundcondStep(STREAM_DEFAULT)
#define acIntegrate(x) acGridIntegrateStep(STREAM_DEFAULT, x)
#define acStore(x) acGridStoreMesh(STREAM_DEFAULT, x)
#define acSynchronizeStream(x) acGridSynchronizeStream(x)
#define acLoadDeviceConstant(x, y) acGridLoadConstant(STREAM_DEFAULT, x, y)
*/

AcResult acInit(const AcMeshInfo mesh_info);

AcResult acQuit(void);

AcResult acSynchronizeStream(const Stream stream);

AcResult acLoadDeviceConstant(const AcRealParam param, const AcReal value);

AcResult acLoad(const AcMesh host_mesh);

AcResult acStore(AcMesh* host_mesh);

AcResult acIntegrate(const AcReal dt);

AcResult acBoundcondStep(void);

AcReal acReduceScal(const ReductionType rtype, const VertexBufferHandle vtxbuf_handle);

AcReal acReduceVec(const ReductionType rtype, const VertexBufferHandle a,
                   const VertexBufferHandle b, const VertexBufferHandle c);

#ifdef __cplusplus
} // extern "C"
#endif
