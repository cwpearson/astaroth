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

static void
distribute_mesh(const AcMesh* src, AcMesh* dst)
{
    MPI_Datatype datatype = MPI_FLOAT;
    if (sizeof(AcReal) == 8)
        datatype = MPI_DOUBLE;

    int process_id, num_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    const size_t count = acVertexBufferSize(dst->info);
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {

        // Communicate to self
        if (process_id == 0) {
            assert(src);
            assert(dst);
            memcpy(&dst->vertex_buffer[i][0], //
                   &src->vertex_buffer[i][0], //
                   count * sizeof(src->vertex_buffer[i][0]));
        }
        // Communicate to others
        for (int j = 1; j < num_processes; ++j) {
            if (process_id == 0) {
                assert(src);

                // Send
                // TODO RECHECK THESE j INDICES
                const size_t src_idx = j * dst->info.int_params[AC_mx] *
                                       dst->info.int_params[AC_my] * src->info.int_params[AC_nz] /
                                       num_processes;

                MPI_Send(&src->vertex_buffer[i][src_idx], count, datatype, j, 0, MPI_COMM_WORLD);
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
}

static void
gather_mesh(const AcMesh* src, AcMesh* dst)
{
    MPI_Datatype datatype = MPI_FLOAT;
    if (sizeof(AcReal) == 8)
        datatype = MPI_DOUBLE;

    int process_id, num_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    size_t count = acVertexBufferSize(src->info);

    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        // Communicate to self
        if (process_id == 0) {
            assert(src);
            assert(dst);
            memcpy(&dst->vertex_buffer[i][0], //
                   &src->vertex_buffer[i][0], //
                   count * sizeof(AcReal));
        }

        // Communicate to others
        for (int j = 1; j < num_processes; ++j) {
            if (process_id == 0) {
                // Recv
                // const size_t dst_idx = j * acVertexBufferCompdomainSize(dst->info);
                const size_t dst_idx = j * dst->info.int_params[AC_mx] *
                                       dst->info.int_params[AC_my] * dst->info.int_params[AC_nz] /
                                       num_processes;

                assert(dst_idx + count <= acVertexBufferSize(dst->info));
                MPI_Status status;
                MPI_Recv(&dst->vertex_buffer[i][dst_idx], count, datatype, j, 0, MPI_COMM_WORLD,
                         &status);
            }
            else {
                // Send
                const size_t src_idx = 0;

                assert(src_idx + count <= acVertexBufferSize(src->info));
                MPI_Send(&src->vertex_buffer[i][src_idx], count, datatype, 0, 0, MPI_COMM_WORLD);
            }
        }
    }
}

int
main(void)
{
    MPI_Init(NULL, NULL);

    int num_processes, process_id;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    printf("Processor %s. Process %d of %d.\n", processor_name, process_id, num_processes);

    AcMeshInfo mesh_info;
    acLoadConfig(AC_DEFAULT_CONFIG, &mesh_info);

    AcMesh* main_mesh     = NULL;
    ModelMesh* model_mesh = NULL;
    if (process_id == 0) {
        main_mesh = acmesh_create(mesh_info);
        acmesh_init_to(INIT_TYPE_RANDOM, main_mesh);
        model_mesh = modelmesh_create(mesh_info);
        acmesh_to_modelmesh(*main_mesh, model_mesh);
    }

    AcMeshInfo submesh_info = mesh_info;
    submesh_info.int_params[AC_nz] /= num_processes;
    update_config(&submesh_info);

    AcMesh* submesh = acmesh_create(submesh_info);

    /////////////////////
    distribute_mesh(main_mesh, submesh);
    gather_mesh(submesh, main_mesh);
    /////////////////////////
    // Autotest
    bool is_acceptable = verify_meshes(*model_mesh, *main_mesh);
    /////

    acmesh_destroy(submesh);

    if (process_id == 0) {
        modelmesh_destroy(model_mesh);
        acmesh_destroy(main_mesh);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
