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
#include "astaroth_defines.h"

#define AC_GEN_STR(X) #X
const char* intparam_names[]   = {AC_FOR_BUILTIN_INT_PARAM_TYPES(AC_GEN_STR) //
                                AC_FOR_USER_INT_PARAM_TYPES(AC_GEN_STR)};
const char* int3param_names[]  = {AC_FOR_BUILTIN_INT3_PARAM_TYPES(AC_GEN_STR) //
                                 AC_FOR_USER_INT3_PARAM_TYPES(AC_GEN_STR)};
const char* realparam_names[]  = {AC_FOR_BUILTIN_REAL_PARAM_TYPES(AC_GEN_STR) //
                                 AC_FOR_USER_REAL_PARAM_TYPES(AC_GEN_STR)};
const char* real3param_names[] = {AC_FOR_BUILTIN_REAL3_PARAM_TYPES(AC_GEN_STR) //
                                  AC_FOR_USER_REAL3_PARAM_TYPES(AC_GEN_STR)};
const char* vtxbuf_names[]     = {AC_FOR_VTXBUF_HANDLES(AC_GEN_STR)};
#undef AC_GEN_STR
