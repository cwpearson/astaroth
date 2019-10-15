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
    Running: mpirun -np <num processes> <executable>
*/
#undef NDEBUG // Assert always
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "astaroth.h"

#include <mpi.h>

// From Astaroth Utils
#include "src/utils/config_loader.h"
#include "src/utils/memory.h"
#include "src/utils/verification.h"

static void
distribute_mesh(const AcMesh src, AcMesh* dst)
{
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Distributing mesh...\n");

    MPI_Datatype datatype = MPI_FLOAT;
    if (sizeof(AcReal) == 8)
        datatype = MPI_DOUBLE;

    int pid, num_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    const size_t count = acVertexBufferSize(dst->info);
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {

        if (pid == 0) {
            // Communicate to self
            assert(dst);
            memcpy(&dst->vertex_buffer[i][0], //
                   &src.vertex_buffer[i][0],  //
                   count * sizeof(src.vertex_buffer[i][0]));

            // Communicate to others
            for (int j = 1; j < num_processes; ++j) {
                const size_t src_idx = acVertexBufferIdx(
                    0, 0, j * src.info.int_params[AC_nz] / num_processes, src.info);

                MPI_Send(&src.vertex_buffer[i][src_idx], count, datatype, j, 0, MPI_COMM_WORLD);
            }
        }
        else {
            assert(dst);

            // Recv
            const size_t dst_idx = 0;
            MPI_Status status;
            MPI_Recv(&dst->vertex_buffer[i][dst_idx], count, datatype, 0, 0, MPI_COMM_WORLD,
                     &status);
        }
    }
}

static void
gather_mesh(const AcMesh src, AcMesh* dst)
{
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Gathering mesh...\n");
    MPI_Datatype datatype = MPI_FLOAT;
    if (sizeof(AcReal) == 8)
        datatype = MPI_DOUBLE;

    int pid, num_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    size_t count = acVertexBufferSize(src.info);

    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        // Communicate to self
        if (pid == 0) {
            assert(dst);
            memcpy(&dst->vertex_buffer[i][0], //
                   &src.vertex_buffer[i][0],  //
                   count * sizeof(src.vertex_buffer[i][0]));

            for (int j = 1; j < num_processes; ++j) {
                // Recv
                const size_t dst_idx = acVertexBufferIdx(
                    0, 0, j * dst->info.int_params[AC_nz] / num_processes, dst->info);

                assert(dst_idx + count <= acVertexBufferSize(dst->info));
                MPI_Status status;
                MPI_Recv(&dst->vertex_buffer[i][dst_idx], count, datatype, j, 0, MPI_COMM_WORLD,
                         &status);
            }
        }
        else {
            // Send
            const size_t src_idx = 0;

            assert(src_idx + count <= acVertexBufferSize(src.info));
            MPI_Send(&src.vertex_buffer[i][src_idx], count, datatype, 0, 0, MPI_COMM_WORLD);
        }
    }
}

static void
communicate_halos(AcMesh* submesh)
{
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Communicating bounds...\n");
    MPI_Datatype datatype = MPI_FLOAT;
    if (sizeof(AcReal) == 8)
        datatype = MPI_DOUBLE;

    int pid, num_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    const size_t count = submesh->info.int_params[AC_mx] * submesh->info.int_params[AC_my] * NGHOST;

    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        { // Front
            // ...|ooooxxx|... -> xxx|ooooooo|...
            const size_t src_idx = acVertexBufferIdx(0, 0, submesh->info.int_params[AC_nz],
                                                     submesh->info);
            const size_t dst_idx = acVertexBufferIdx(0, 0, 0, submesh->info);
            const int send_pid   = (pid + 1) % num_processes;
            const int recv_pid   = (pid + num_processes - 1) % num_processes;

            MPI_Request request;
            MPI_Isend(&submesh->vertex_buffer[i][src_idx], count, datatype, send_pid, i,
                      MPI_COMM_WORLD, &request);
            fflush(stdout);

            MPI_Status status;
            MPI_Recv(&submesh->vertex_buffer[i][dst_idx], count, datatype, recv_pid, i,
                     MPI_COMM_WORLD, &status);

            MPI_Wait(&request, &status);
        }
        { // Back
            // ...|ooooooo|xxx <- ...|xxxoooo|...
            const size_t src_idx = acVertexBufferIdx(0, 0, NGHOST, submesh->info);
            const size_t dst_idx = acVertexBufferIdx(0, 0, NGHOST + submesh->info.int_params[AC_nz],
                                                     submesh->info);
            const int send_pid   = (pid + num_processes - 1) % num_processes;
            const int recv_pid   = (pid + 1) % num_processes;

            MPI_Request request;
            MPI_Isend(&submesh->vertex_buffer[i][src_idx], count, datatype, send_pid,
                      NUM_VTXBUF_HANDLES + i, MPI_COMM_WORLD, &request);

            MPI_Status status;
            MPI_Recv(&submesh->vertex_buffer[i][dst_idx], count, datatype, recv_pid,
                     NUM_VTXBUF_HANDLES + i, MPI_COMM_WORLD, &status);

            MPI_Wait(&request, &status);
        }
    }
}

int
main(void)
{
    int num_processes, pid;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    printf("Processor %s. Process %d of %d.\n", processor_name, pid, num_processes);

    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);

    AcMesh model, candidate, submesh;

    // Master CPU
    if (pid == 0) {
        acMeshCreate(info, &model);
        acMeshCreate(info, &candidate);

        acMeshRandomize(&model);
        acMeshApplyPeriodicBounds(&model);
    }

    assert(info.int_params[AC_nz] % num_processes == 0);

    AcMeshInfo submesh_info = info;
    submesh_info.int_params[AC_nz] /= num_processes;
    acUpdateConfig(&submesh_info);
    acMeshCreate(submesh_info, &submesh);

    distribute_mesh(model, &submesh);

    // GPU-GPU communication
    /*
    const int device_id = pid % acGetNumDevicesPerNode();

    Device device;
    acDeviceCreate(device_id, submesh_info, &device);

    acDeviceLoadMesh(device, STREAM_DEFAULT, submesh);
    acDeviceCommunicateHalosMPI(device);
    acDeviceStoreMesh(device, STREAM_DEFAULT, &submesh);

    acDeviceDestroy(device);
    */

    // GPU-CPU-CPU-GPU communication
    const int device_id = pid % acGetNumDevicesPerNode();

    Device device;
    acDeviceCreate(device_id, submesh_info, &device);

    acDeviceLoadMesh(device, STREAM_DEFAULT, submesh);
    acDevicePeriodicBoundconds(device, STREAM_DEFAULT, (int3){0, 0, 0},
                               (int3){submesh_info.int_params[AC_mx],
                                      submesh_info.int_params[AC_my],
                                      submesh_info.int_params[AC_mz]});
    acDeviceStoreMesh(device, STREAM_DEFAULT, &submesh);
    communicate_halos(&submesh);

    acDeviceDestroy(device);
    //

    //
    // CPU-CPU communication
    // communicate_halos(&submesh);
    //
    gather_mesh(submesh, &candidate);

    acMeshDestroy(&submesh);
    // Master CPU
    if (pid == 0) {
        acVerifyMesh(model, candidate);
        acMeshDestroy(&model);
        acMeshDestroy(&candidate);
    }

    // GPU
    /*
    Device device;
    acDeviceCreate(pid, info, &device);

    acDeviceLoadMesh(device, STREAM_DEFAULT, model);

    acDeviceStoreMesh(device, STREAM_DEFAULT, &candidate);
    acDeviceDestroy(device);
    */
    //

    MPI_Finalize();
    return EXIT_SUCCESS;
}
