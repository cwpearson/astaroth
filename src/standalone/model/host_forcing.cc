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
#include "host_forcing.h"

#include "core/math_utils.h"

// The is a wrapper for genering random numbers with a chosen system.
AcReal
get_random_number_01()
{
    // TODO: Implement better randon number generator http://www.cplusplus.com/reference/random/
    return AcReal(rand()) / AcReal(RAND_MAX);
}

AcReal3
cross(const AcReal3& a, const AcReal3& b)
{
    AcReal3 c;

    c.x = a.y * b.z - a.z * b.y;
    c.y = a.z * b.x - a.x * b.z;
    c.z = a.x * b.y - a.y * b.x;

    return c;
}

AcReal
dot(const AcReal3& a, const AcReal3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

AcReal3
vec_norm(const AcReal3& a)
{
    AcReal3 c;
    AcReal norm = dot(a, a);

    c.x = a.x / sqrt(norm);
    c.y = a.y / sqrt(norm);
    c.z = a.z / sqrt(norm);

    return c;
}

AcReal3
vec_multi_scal(const AcReal scal, const AcReal3& a)
{
    AcReal3 c;

    c.x = a.x * scal;
    c.y = a.y * scal;
    c.z = a.z * scal;

    return c;
}

// Generate forcing wave vector k_force
AcReal3
helical_forcing_k_generator(const AcReal kmax, const AcReal kmin)
{
    AcReal phi, theta, kk; // Spherical direction coordinates
    AcReal3 k_force;       // forcing wave vector

    AcReal delta_k = kmax - kmin;

    // Generate vector in spherical coordinates
    phi   = AcReal(2.0) * AcReal(M_PI) * get_random_number_01();
    theta = AcReal(M_PI) * get_random_number_01();
    kk    = delta_k * get_random_number_01() + kmin;

    // Cast into Cartesian form
    k_force = (AcReal3){kk * sin(theta) * cos(phi), //
                        kk * sin(theta) * sin(phi), //
                        kk * cos(theta)};

    // printf("k_force.x %f, k_force.y %f, k_force.z %f \n", k_force.x, k_force.y, k_force.z);

    // Round the numbers. In that way k(x/y/z) will get complete waves.
    k_force.x = round(k_force.x);
    k_force.y = round(k_force.y);
    k_force.z = round(k_force.z);

    // printf("After rounding --> k_force.x %f, k_force.y %f, k_force.z %f \n", k_force.x,
    // k_force.y, k_force.z);

    return k_force;
}

// Generate the unit perpendicular unit vector e required for helical forcing
// Addapted from Pencil code forcing.f90 hel_vec() subroutine.
void
helical_forcing_e_generator(AcReal3* e_force, const AcReal3 k_force)
{

    AcReal3 k_cross_e         = cross(k_force, *e_force);
    k_cross_e                 = vec_norm(k_cross_e);
    AcReal3 k_cross_k_cross_e = cross(k_force, k_cross_e);
    k_cross_k_cross_e         = vec_norm(k_cross_k_cross_e);
    AcReal phi                = AcReal(2.0) * AcReal(M_PI) * get_random_number_01();
    AcReal3 ee_tmp1           = vec_multi_scal(cos(phi), k_cross_e);
    AcReal3 ee_tmp2           = vec_multi_scal(sin(phi), k_cross_k_cross_e);

    *e_force = (AcReal3){ee_tmp1.x + ee_tmp2.x, ee_tmp1.y + ee_tmp2.y, ee_tmp1.z + ee_tmp2.z};
}

// PC Manual Eq. 223
void
helical_forcing_special_vector(AcReal3* ff_hel_re, AcReal3* ff_hel_im, const AcReal3 k_force,
                               const AcReal3 e_force, const AcReal relhel)
{

    // k dot e
    AcReal3 kdote;
    kdote.x = k_force.x * e_force.x;
    kdote.y = k_force.y * e_force.y;
    kdote.z = k_force.z * e_force.z;

    // k cross e
    AcReal3 k_cross_e;
    k_cross_e.x = k_force.y * e_force.z - k_force.z * e_force.y;
    k_cross_e.y = k_force.z * e_force.x - k_force.x * e_force.z;
    k_cross_e.z = k_force.x * e_force.y - k_force.y * e_force.x;

    // k cross k cross e
    AcReal3 k_cross_k_cross_e;
    k_cross_k_cross_e.x = k_force.y * k_cross_e.z - k_force.z * k_cross_e.y;
    k_cross_k_cross_e.y = k_force.z * k_cross_e.x - k_force.x * k_cross_e.z;
    k_cross_k_cross_e.z = k_force.x * k_cross_e.y - k_force.y * k_cross_e.x;

    // abs(k)
    AcReal kabs = sqrt(k_force.x * k_force.x + k_force.y * k_force.y + k_force.z * k_force.z);

    AcReal denominator = sqrt(AcReal(1.0) + relhel * relhel) * kabs *
                         sqrt(kabs * kabs -
                              (kdote.x * kdote.x + kdote.y * kdote.y + kdote.z * kdote.z));

    // MV: I suspect there is a typo in the Pencil Code manual!
    //*ff_hel_re = (AcReal3){-relhel*kabs*k_cross_e.x/denominator,
    //                       -relhel*kabs*k_cross_e.y/denominator,
    //                       -relhel*kabs*k_cross_e.z/denominator};

    //*ff_hel_im = (AcReal3){k_cross_k_cross_e.x/denominator,
    //                       k_cross_k_cross_e.y/denominator,
    //                       k_cross_k_cross_e.z/denominator};

    // See PC forcing.f90 forcing_hel_both()
    *ff_hel_re = (AcReal3){kabs * k_cross_e.x / denominator, kabs * k_cross_e.y,
                           kabs * k_cross_e.z};

    *ff_hel_im = (AcReal3){relhel * k_cross_k_cross_e.x / denominator, relhel * k_cross_k_cross_e.y,
                           relhel * k_cross_k_cross_e.z};
}
