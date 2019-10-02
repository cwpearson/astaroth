#ifndef STENCIL_ORDER
#define STENCIL_ORDER (6)
#endif

Scalar
first_derivative(Scalar pencil[], Scalar inv_ds)
{
#if STENCIL_ORDER == 2
    Scalar coefficients[] = {0, 1.0 / 2.0};
#elif STENCIL_ORDER == 4
    Scalar coefficients[] = {0, 2.0 / 3.0, -1.0 / 12.0};
#elif STENCIL_ORDER == 6
    Scalar coefficients[] = {0, 3.0 / 4.0, -3.0 / 20.0, 1.0 / 60.0};
#elif STENCIL_ORDER == 8
    Scalar coefficients[] = {0, 4.0 / 5.0, -1.0 / 5.0, 4.0 / 105.0, -1.0 / 280.0};
#endif

#define MID (STENCIL_ORDER / 2)

    Scalar res = 0;

    for (int i = 1; i <= MID; ++i) {
        res = res + coefficients[i] * (pencil[MID + i] - pencil[MID - i]);
    }

    return res * inv_ds;
}

Scalar
second_derivative(Scalar pencil[], Scalar inv_ds)
{
#if STENCIL_ORDER == 2
    Scalar coefficients[] = {-2.0, 1.0};
#elif STENCIL_ORDER == 4
    Scalar coefficients[] = {-5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0};
#elif STENCIL_ORDER == 6
    Scalar coefficients[] = {-49.0 / 18.0, 3.0 / 2.0, -3.0 / 20.0, 1.0 / 90.0};
#elif STENCIL_ORDER == 8
    Scalar coefficients[] = {-205.0 / 72.0, 8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0};
#endif

#define MID (STENCIL_ORDER / 2)
    Scalar res = coefficients[0] * pencil[MID];

    for (int i = 1; i <= MID; ++i) {
        res = res + coefficients[i] * (pencil[MID + i] + pencil[MID - i]);
    }

    return res * inv_ds * inv_ds;
}

Scalar
cross_derivative(Scalar pencil_a[], Scalar pencil_b[], Scalar inv_ds_a, Scalar inv_ds_b)
{
#if STENCIL_ORDER == 2
    Scalar coefficients[] = {0, 1.0 / 4.0};
#elif STENCIL_ORDER == 4
    Scalar coefficients[] = {0, 1.0 / 32.0,
                             1.0 / 64.0}; // TODO correct coefficients, these are just placeholders
#elif STENCIL_ORDER == 6
    Scalar fac            = 1.0 / 720.0;
    Scalar coefficients[] = {0.0 * fac, 270.0 * fac, -27.0 * fac, 2.0 * fac};
#elif STENCIL_ORDER == 8
    Scalar fac            = (1.0 / 20160.0);
    Scalar coefficients[] = {0.0 * fac, 8064.0 * fac, -1008.0 * fac, 128.0 * fac, -9.0 * fac};
#endif

#define MID (STENCIL_ORDER / 2)
    Scalar res = 0.0;

    for (int i = 1; i <= MID; ++i) {
        res = res + coefficients[i] * (pencil_a[MID + i] + pencil_a[MID - i] - pencil_b[MID + i] -
                                       pencil_b[MID - i]);
    }
    return res * inv_ds_a * inv_ds_b;
}

Scalar
derx(int3 vertexIdx, in ScalarField arr)
{
    Scalar pencil[STENCIL_ORDER + 1];

    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset) {
        pencil[offset] = arr[vertexIdx.x + offset - STENCIL_ORDER / 2, vertexIdx.y, vertexIdx.z];
    }

    return first_derivative(pencil, AC_inv_dsx);
}

Scalar
derxx(int3 vertexIdx, in ScalarField arr)
{
    Scalar pencil[STENCIL_ORDER + 1];

    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset) {
        pencil[offset] = arr[vertexIdx.x + offset - STENCIL_ORDER / 2, vertexIdx.y, vertexIdx.z];
    }

    return second_derivative(pencil, AC_inv_dsx);
}

Scalar
derxy(int3 vertexIdx, in ScalarField arr)
{
    Scalar pencil_a[STENCIL_ORDER + 1];

    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset) {
        pencil_a[offset] = arr[vertexIdx.x + offset - STENCIL_ORDER / 2,
                               vertexIdx.y + offset - STENCIL_ORDER / 2, vertexIdx.z];
    }

    Scalar pencil_b[STENCIL_ORDER + 1];

    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset) {
        pencil_b[offset] = arr[vertexIdx.x + offset - STENCIL_ORDER / 2,
                               vertexIdx.y + STENCIL_ORDER / 2 - offset, vertexIdx.z];
    }

    return cross_derivative(pencil_a, pencil_b, AC_inv_dsx, AC_inv_dsy);
}

Scalar
derxz(int3 vertexIdx, in ScalarField arr)
{
    Scalar pencil_a[STENCIL_ORDER + 1];

    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset) {
        pencil_a[offset] = arr[vertexIdx.x + offset - STENCIL_ORDER / 2, vertexIdx.y,
                               vertexIdx.z + offset - STENCIL_ORDER / 2];
    }

    Scalar pencil_b[STENCIL_ORDER + 1];
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset) {
        pencil_b[offset] = arr[vertexIdx.x + offset - STENCIL_ORDER / 2, vertexIdx.y,
                               vertexIdx.z + STENCIL_ORDER / 2 - offset];
    }

    return cross_derivative(pencil_a, pencil_b, AC_inv_dsx, AC_inv_dsz);
}

Scalar
dery(int3 vertexIdx, in ScalarField arr)
{
    Scalar pencil[STENCIL_ORDER + 1];

    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset) {
        pencil[offset] = arr[vertexIdx.x, vertexIdx.y + offset - STENCIL_ORDER / 2, vertexIdx.z];
    }

    return first_derivative(pencil, AC_inv_dsy);
}

Scalar
deryy(int3 vertexIdx, in ScalarField arr)
{
    Scalar pencil[STENCIL_ORDER + 1];

    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset) {
        pencil[offset] = arr[vertexIdx.x, vertexIdx.y + offset - STENCIL_ORDER / 2, vertexIdx.z];
    }

    return second_derivative(pencil, AC_inv_dsy);
}

Scalar
deryz(int3 vertexIdx, in ScalarField arr)
{
    Scalar pencil_a[STENCIL_ORDER + 1];

    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset) {
        pencil_a[offset] = arr[vertexIdx.x, vertexIdx.y + offset - STENCIL_ORDER / 2,
                               vertexIdx.z + offset - STENCIL_ORDER / 2];
    }

    Scalar pencil_b[STENCIL_ORDER + 1];

    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset) {
        pencil_b[offset] = arr[vertexIdx.x, vertexIdx.y + offset - STENCIL_ORDER / 2,
                               vertexIdx.z + STENCIL_ORDER / 2 - offset];
    }

    return cross_derivative(pencil_a, pencil_b, AC_inv_dsy, AC_inv_dsz);
}

Scalar
derz(int3 vertexIdx, in ScalarField arr)
{
    Scalar pencil[STENCIL_ORDER + 1];

    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset) {
        pencil[offset] = arr[vertexIdx.x, vertexIdx.y, vertexIdx.z + offset - STENCIL_ORDER / 2];
    }

    return first_derivative(pencil, AC_inv_dsz);
}

Scalar
derzz(int3 vertexIdx, in ScalarField arr)
{
    Scalar pencil[STENCIL_ORDER + 1];

    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset) {
        pencil[offset] = arr[vertexIdx.x, vertexIdx.y, vertexIdx.z + offset - STENCIL_ORDER / 2];
    }

    return second_derivative(pencil, AC_inv_dsz);
}

Preprocessed Scalar
value(in ScalarField vertex)
{
    return vertex[vertexIdx];
}

Preprocessed Vector
gradient(in ScalarField vertex)
{
    return (Vector){derx(vertexIdx, vertex), dery(vertexIdx, vertex), derz(vertexIdx, vertex)};
}

Preprocessed Matrix
hessian(in ScalarField vertex)
{
    Matrix hessian;

    hessian.row[0] = (Vector){derxx(vertexIdx, vertex), derxy(vertexIdx, vertex),
                              derxz(vertexIdx, vertex)};
    hessian.row[1] = (Vector){hessian.row[0].y, deryy(vertexIdx, vertex), deryz(vertexIdx, vertex)};
    hessian.row[2] = (Vector){hessian.row[0].z, hessian.row[1].z, derzz(vertexIdx, vertex)};

    return hessian;
}
