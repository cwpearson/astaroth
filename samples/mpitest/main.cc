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
    Running: mpirun -np <num processes> <executable>
*/
#include "astaroth.h"
#include "astaroth_utils.h"
#include "errchk.h"

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

void
gaussian_radial_explosion(AcMesh* mesh)
{
    AcReal* uu_x     = mesh->vertex_buffer[VTXBUF_UUX];
    AcReal* uu_y     = mesh->vertex_buffer[VTXBUF_UUY];
    AcReal* uu_z     = mesh->vertex_buffer[VTXBUF_UUZ];
    const int mx     = mesh->info.int_params[AC_mx];
    const int my     = mesh->info.int_params[AC_my];
    const int nx_min = mesh->info.int_params[AC_nx_min];
    const int nx_max = mesh->info.int_params[AC_nx_max];
    const int ny_min = mesh->info.int_params[AC_nx_min];
    const int ny_max = mesh->info.int_params[AC_nx_max];
    const int nz_min = mesh->info.int_params[AC_nx_min];
    const int nz_max = mesh->info.int_params[AC_nx_max];
    const AcReal DX  = mesh->info.real_params[AC_dsx];
    const AcReal DY  = mesh->info.real_params[AC_dsy];
    const AcReal DZ  = mesh->info.real_params[AC_dsz];


    const AcReal xorig = 0.00001;
    const AcReal yorig = 0.00001;
    const AcReal zorig = 0.00001;

    const AcReal INIT_LOC_UU_X = 0.01;
    const AcReal INIT_LOC_UU_Y = 32 * DY;
    const AcReal INIT_LOC_UU_Z = 32 * DZ;
    const AcReal AMPL_UU       = 1;
    const AcReal UU_SHELL_R    = 0.8;
    const AcReal WIDTH_UU      = 0.2;
    // Outward explosion with gaussian initial velocity profile.
    int idx;
    AcReal xx, yy, zz, rr2, rr, theta = 0.0, phi = 0.0;
    AcReal uu_radial;
    // real theta_old = 0.0;
    for (int k = nz_min; k < nz_max; k++) {
        for (int j = ny_min; j < ny_max; j++) {
            for (int i = nx_min; i < nx_max; i++) {
                // Calculate the value of velocity in a particular radius.
                idx = i + j * mx + k * mx * my;
                // Determine the coordinates
                xx  = DX * (i - nx_min) - xorig;
                xx  = xx - INIT_LOC_UU_X;
                yy  = DY * (j - ny_min) - yorig;
                yy  = yy - INIT_LOC_UU_Y;
                zz  = DZ * (k - nz_min) - zorig;
                zz  = zz - INIT_LOC_UU_Z;
                rr2 = pow(xx, 2.0) + pow(yy, 2.0) + pow(zz, 2.0);
                rr  = sqrt(rr2);

                // printf("[%d %d %d] %e %e %e\n", i, j, k, DX, DY, DZ);

                // Origin is different!
                AcReal xx_abs, yy_abs, zz_abs;
                if (rr > 0.0) {
                    // theta range [0, PI]
                    if (zz >= 0.0) {
                        theta = acos(min(1.0, zz / rr));
                        if (theta > M_PI / 2.0 || theta < 0.0) {
                            printf("Explosion THETA WRONG: zz = %.3f, rr = %.3f, theta = %.3e/PI, "
                                   "M_PI = %.3e\n",
                                   zz, rr, theta / M_PI, M_PI);
                        }
                    }
                    else {
                        zz_abs = -zz; // Needs a posite value for acos
                        theta  = M_PI - acos(zz_abs / rr);
                        if (theta < M_PI / 2.0 || theta > 2 * M_PI) {
                            printf("Explosion THETA WRONG: zz = %.3f, rr = %.3f, theta = %.3e/PI, "
                                   "M_PI = %.3e\n",
                                   zz, rr, theta / M_PI, M_PI);
                        }
                    }
                    // phi range [0, 2*PI]i
                    if (xx != 0.0) {
                        if (xx < 0.0 && yy >= 0.0) {
                            //-+
                            xx_abs = -xx; // Needs a posite value for atan
                            phi    = M_PI - atan(yy / xx_abs);
                            if (phi < (M_PI / 2.0) || phi > M_PI) {
                                printf("Explosion PHI WRONG -+: xx = %.3f, yy = %.3f, phi = "
                                       "%.3e/PI, M_PI = %.3e\n",
                                       xx, yy, phi / M_PI, M_PI);
                            }
                        }
                        else if (xx > 0.0 && yy < 0.0) {
                            //+-
                            yy_abs = -yy;
                            phi    = 2.0 * M_PI - atan(yy_abs / xx);
                            if (phi < (3.0 * M_PI) / 2.0 || phi > (2.0 * M_PI + 1e-6)) {
                                printf("Explosion PHI WRONG +-: xx = %.3f, yy = %.3f, phi = "
                                       "%.3e/PI, M_PI = %.3e\n",
                                       xx, yy, phi / M_PI, M_PI);
                            }
                        }
                        else if (xx < 0.0 && yy < 0.0) {
                            //--
                            yy_abs = -yy;
                            xx_abs = -xx;
                            phi    = M_PI + atan(yy_abs / xx_abs);
                            if (phi < M_PI || phi > ((3.0 * M_PI) / 2.0 + 1e-6)) {
                                printf("Explosion PHI WRONG --: xx = %.3f, yy = %.3f, xx_abs = "
                                       "%.3f, yy_abs = %.3f, phi = %.3e, (3.0*M_PI)/2.0 = %.3e\n",
                                       xx, yy, xx_abs, yy_abs, phi, (3.0 * M_PI) / 2.0);
                            }
                        }
                        else {
                            //++
                            phi = atan(yy / xx);
                            if (phi < 0 || phi > M_PI / 2.0) {
                                printf("Explosion PHI WRONG --: xx = %.3f, yy = %.3f, phi = %.3e, "
                                       "(3.0*M_PI)/2.0 = %.3e\n",
                                       xx, yy, phi, (3.0 * M_PI) / 2.0);
                            }
                        }
                    }
                    else { // To avoid div by zero with atan
                        if (yy > 0.0) {
                            phi = M_PI / 2.0;
                        }
                        else if (yy < 0.0) {
                            phi = (3.0 * M_PI) / 2.0;
                        }
                        else {
                            phi = 0.0;
                        }
                    }
                    // Set zero for explicit safekeeping
                    if (xx == 0.0 && yy == 0.0) {
                        phi = 0.0;
                    }
                    // Gaussian velocity
                    // uu_radial = AMPL_UU*exp( -rr2 / (2.0*pow(WIDTH_UU, 2.0)) );
                    // New distribution, where that gaussion wave is not in the exact centre
                    // coordinates uu_radial = AMPL_UU*exp( -pow((rr - 4.0*WIDTH_UU),2.0) /
                    // (2.0*pow(WIDTH_UU, 2.0)) ); //TODO: Parametrize the peak location.
                    uu_radial = AMPL_UU *
                                exp(-pow((rr - UU_SHELL_R), 2.0) / (2.0 * pow(WIDTH_UU, 2.0)));
                }
                else {
                    uu_radial = 0.0; // TODO: There will be a discontinuity in the origin... Should
                                     // the shape of the distribution be different?
                }

                // if (rr > 0.2 && rr < 1.0) {
                //     printf("%e\n", uu_radial);
                // }

                // Determine the carthesian velocity components and lnrho
                uu_x[idx] = uu_radial * sin(theta) * cos(phi);
                uu_y[idx] = uu_radial * sin(theta) * sin(phi);
                uu_z[idx] = uu_radial * cos(theta);
                // Temporary diagnosticv output (TODO: Remove after not needed)
                // if (theta > theta_old) {
                // if (theta > M_PI || theta < 0.0 || phi < 0.0 || phi > 2*M_PI) {
                /*  printf("Explosion: xx = %.3f, yy = %.3f, zz = %.3f, rr = %.3f, phi = %.3e/PI,
                   theta = %.3e/PI\n, M_PI = %.3e", xx, yy, zz, rr, phi/M_PI, theta/M_PI, M_PI);
                    printf("           uu_radial = %.3e, uu_x[%i] = %.3e, uu_y[%i] = %.3e, uu_z[%i]
                   = %.3e \n", uu_radial, idx, uu_x[idx], idx, uu_y[idx], idx, uu_z[idx]); theta_old
                   = theta;
                */
            }
        }
    }
}

static AcReal
randf(void)
{
    return (AcReal)rand() / (AcReal)RAND_MAX;
}

AcResult
meshRadial(AcMesh* mesh)
{

    const int n = acVertexBufferSize(mesh->info);

    // lnrho to constant
    for (int i = 0; i < n; ++i)
        mesh->vertex_buffer[VTXBUF_LNRHO][i] = 0.5;

    // A to random
    for (int i = 0; i < n; ++i)
        mesh->vertex_buffer[VTXBUF_AX][i] = randf();
    for (int i = 0; i < n; ++i)
        mesh->vertex_buffer[VTXBUF_AY][i] = randf();
    for (int i = 0; i < n; ++i)
        mesh->vertex_buffer[VTXBUF_AZ][i] = randf();

    // entropy random
    for (int i = 0; i < n; ++i)
        mesh->vertex_buffer[VTXBUF_ENTROPY][i] = randf();

    // velocity is radial explosion
    gaussian_radial_explosion(mesh);

    return AC_SUCCESS;
}

void
dumpMesh(AcMesh* mesh, const std::string& path)
{
    const char delim[] = ",";

    FILE* outf = fopen(path.c_str(), "w");
    if (!outf) {
        std::cerr << "unable to open \"" << path << "\" for writing\n";
        exit(1);
    }

    // column headers
    fprintf(outf, "Z%sY%sX", delim, delim);
    for (int64_t qi = 0; qi < NUM_VTXBUF_HANDLES; ++qi) {
        std::string colName = "data" + std::to_string(qi);
        fprintf(outf, "%s%s", delim, colName.c_str());
    }
    fprintf(outf, "\n");

    // is this right?
    int3 origin = mesh->info.int3_params[AC_multigpu_offset];
    origin.x += 1;
    origin.y += 1;
    origin.z += 1;

    const int nz = mesh->info.int_params[AC_nz];
    const int ny = mesh->info.int_params[AC_ny];
    const int nx = mesh->info.int_params[AC_nx];
    // const int mz = mesh->info.int_params[AC_mz];
    const int my = mesh->info.int_params[AC_my];
    const int mx = mesh->info.int_params[AC_mx];

    // rows
    for (int lz = 0; lz < nz; ++lz) {
        for (int ly = 0; ly < ny; ++ly) {
            for (int lx = 0; lx < nx; ++lx) {

                const int3 pos{origin.x + lx, origin.y + ly, origin.z + lz};

                fprintf(outf, "%d%s%d%s%d", pos.z, delim, pos.y, delim, pos.x);

                for (int64_t qi = 0; qi < NUM_VTXBUF_HANDLES; ++qi) {
                    AcReal val = mesh->vertex_buffer[qi][lz * (my * mx) + ly * mx + lx];
                    if (8 == sizeof(AcReal)) {
                        fprintf(outf, "%s%.17f", delim, val);
                    }
                    else if (4 == sizeof(AcReal)) {
                        fprintf(outf, "%s%.9f", delim, val);
                    }
                }

                fprintf(outf, "\n");
            }
        }
    }

    fclose(outf);
}

#if AC_MPI_ENABLED

#include <mpi.h>
#include <vector>

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(*arr))

int
main(void)
{
    MPI_Init(NULL, NULL);
    int nprocs, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    // Set random seed for reproducibility
    srand(321654987);

    // CPU alloc
    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);

    AcMesh model, candidate;
    if (pid == 0) {
        acHostMeshCreate(info, &model);
        acHostMeshCreate(info, &candidate);
        meshRadial(&model);
        meshRadial(&candidate);
    }

    {
        std::stringstream ss;
        ss << "candidate_init_";
        ss << pid << ".txt";
        std::cerr << "dump to " << ss.str() << "\n";
        dumpMesh(&candidate, ss.str());
    }

    // GPU alloc & compute
    acGridInit(info);

    // Boundconds
    acGridLoadMesh(STREAM_DEFAULT, model);
    acGridPeriodicBoundconds(STREAM_DEFAULT);
    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    if (pid == 0) {
        acHostMeshApplyPeriodicBounds(&model);
        const AcResult res = acVerifyMesh("Boundconds", model, candidate);
        ERRCHK_ALWAYS(res == AC_SUCCESS);
        meshRadial(&model);
    }

    // Integration
    acGridLoadMesh(STREAM_DEFAULT, model);
    acGridIntegrate(STREAM_DEFAULT, FLT_EPSILON);
    acGridPeriodicBoundconds(STREAM_DEFAULT);
    acGridStoreMesh(STREAM_DEFAULT, &candidate);

    {
        std::stringstream ss;
        ss << "candidate_final_";
        ss << pid << ".txt";
        std::cerr << "dump to " << ss.str() << "\n";
        dumpMesh(&candidate, ss.str());
    }

    if (pid == 0) {
        acHostIntegrateStep(model, FLT_EPSILON);
        acHostMeshApplyPeriodicBounds(&model);
        const AcResult res = acVerifyMesh("Integration", model, candidate);
        ERRCHK_ALWAYS(res == AC_SUCCESS);
        meshRadial(&model);
    }

    // Scalar reductions
    acGridLoadMesh(STREAM_DEFAULT, model);

    if (pid == 0) {
        printf("---Test: Scalar reductions---\n");
        printf("Warning: testing only RTYPE_MAX and RTYPE_MIN\n");
        fflush(stdout);
    }
    for (size_t i = 0; i < 2; ++i) { // NOTE: 2 instead of NUM_RTYPES
        const VertexBufferHandle v0 = VTXBUF_UUX;
        AcReal candval;
        acGridReduceScal(STREAM_DEFAULT, (ReductionType)i, v0, &candval);
        if (pid == 0) {
            const AcReal modelval   = acHostReduceScal(model, (ReductionType)i, v0);
            Error error             = acGetError(modelval, candval);
            error.maximum_magnitude = acHostReduceScal(model, RTYPE_MAX, v0);
            error.minimum_magnitude = acHostReduceScal(model, RTYPE_MIN, v0);
            ERRCHK_ALWAYS(acEvalError(rtype_names[i], error));
        }
    }

    // Vector reductions
    if (pid == 0) {
        printf("---Test: Vector reductions---\n");
        printf("Warning: testing only RTYPE_MAX and RTYPE_MIN\n");
        fflush(stdout);
    }
    for (size_t i = 0; i < 2; ++i) { // NOTE: 2 instead of NUM_RTYPES
        const VertexBufferHandle v0 = VTXBUF_UUX;
        const VertexBufferHandle v1 = VTXBUF_UUY;
        const VertexBufferHandle v2 = VTXBUF_UUZ;
        AcReal candval;
        acGridReduceVec(STREAM_DEFAULT, (ReductionType)i, v0, v1, v2, &candval);
        if (pid == 0) {
            const AcReal modelval   = acHostReduceVec(model, (ReductionType)i, v0, v1, v2);
            Error error             = acGetError(modelval, candval);
            error.maximum_magnitude = acHostReduceVec(model, RTYPE_MAX, v0, v1, v2);
            error.minimum_magnitude = acHostReduceVec(model, RTYPE_MIN, v0, v1, v1);
            ERRCHK_ALWAYS(acEvalError(rtype_names[i], error));
        }
    }

    if (pid == 0) {
        acHostMeshDestroy(&model);
        acHostMeshDestroy(&candidate);
    }

    acGridQuit();
    MPI_Finalize();
    return EXIT_SUCCESS;
}

#else
int
main(void)
{
    printf("The library was built without MPI support, cannot run mpitest. Rebuild Astaroth with "
           "cmake -DMPI_ENABLED=ON .. to enable.\n");
    return EXIT_FAILURE;
}
#endif // AC_MPI_ENABLES
