
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

AcReal get_random_number_01();

AcReal3 cross(const AcReal3& a, const AcReal3& b);

AcReal dot(const AcReal3& a, const AcReal3& b);

AcReal3 vec_norm(const AcReal3& a);

AcReal3 vec_multi_scal(const AcReal scal, const AcReal3& a);

AcReal3 helical_forcing_k_generator(const AcReal kmax, const AcReal kmin);

void helical_forcing_e_generator(AcReal3* e_force, const AcReal3 k_force);

void helical_forcing_special_vector(AcReal3* ff_hel_re, AcReal3* ff_hel_im, const AcReal3 k_force, const AcReal3 e_force, const AcReal relhel);
