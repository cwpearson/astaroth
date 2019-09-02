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
//#include <cmath>
// using namespace std; // Potentially bad practice to declare namespace std here
#include <math.h> // isnan, isinf // Overloads incorrect/bugged with GCC <= 6.0
//#include <tgmath.h> // Even this does not work
#include <stdlib.h> // rand

template <class T>
static inline const T
max(const T& a, const T& b)
{
    return a > b ? a : b;
}

template <class T>
static inline const T
min(const T& a, const T& b)
{
    return a < b ? a : b;
}

static inline const int3
max(const int3& a, const int3& b)
{
    return (int3){max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)};
}

static inline const int3
min(const int3& a, const int3& b)
{
    return (int3){min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)};
}

template <class T>
static inline const T
sum(const T& a, const T& b)
{
    return a + b;
}

template <class T>
static inline const T
clamp(const T& val, const T& min, const T& max)
{
    return val < min ? min : val > max ? max : val;
}

static inline AcReal
randr()
{
    return AcReal(rand()) / AcReal(RAND_MAX);
}

static inline bool
is_power_of_two(const unsigned val)
{
    return val && !(val & (val - 1));
}

#ifdef __CUDACC__
#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#else
#define HOST_DEVICE_INLINE inline
#endif // __CUDACC__

static HOST_DEVICE_INLINE AcReal3
operator+(const AcReal3& a, const AcReal3& b)
{
    return (AcReal3){a.x + b.x, a.y + b.y, a.z + b.z};
}

static HOST_DEVICE_INLINE int3
operator+(const int3& a, const int3& b)
{
    return (int3){a.x + b.x, a.y + b.y, a.z + b.z};
}

static HOST_DEVICE_INLINE AcReal3
operator-(const AcReal3& a, const AcReal3& b)
{
    return (AcReal3){a.x - b.x, a.y - b.y, a.z - b.z};
}

static HOST_DEVICE_INLINE int3
operator-(const int3& a, const int3& b)
{
    return (int3){a.x - b.x, a.y - b.y, a.z - b.z};
}

static HOST_DEVICE_INLINE AcReal3
operator-(const AcReal3& a)
{
    return (AcReal3){-a.x, -a.y, -a.z};
}

static HOST_DEVICE_INLINE AcReal3 operator*(const AcReal& a, const AcReal3& b)
{
    return (AcReal3){a * b.x, a * b.y, a * b.z};
}

static HOST_DEVICE_INLINE AcReal3 operator*(const AcReal3& b, const AcReal& a)
{
    return (AcReal3){a * b.x, a * b.y, a * b.z};
}

static HOST_DEVICE_INLINE AcReal
dot(const AcReal3& a, const AcReal3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static HOST_DEVICE_INLINE AcReal3
mul(const AcMatrix& aa, const AcReal3& x)
{
    return (AcReal3){dot(aa.row[0], x), dot(aa.row[1], x), dot(aa.row[2], x)};
}

static HOST_DEVICE_INLINE AcReal3
cross(const AcReal3& a, const AcReal3& b)
{
    AcReal3 c;

    c.x = a.y * b.z - a.z * b.y;
    c.y = a.z * b.x - a.x * b.z;
    c.z = a.x * b.y - a.y * b.x;

    return c;
}

static HOST_DEVICE_INLINE bool
is_valid(const AcReal a)
{
    return !isnan(a) && !isinf(a);
}

static HOST_DEVICE_INLINE bool
is_valid(const AcReal3& a)
{
    return is_valid(a.x) && is_valid(a.y) && is_valid(a.z);
}
