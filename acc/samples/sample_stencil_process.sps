// TODO comments and reformatting

uniform Scalar dsx;
uniform Scalar dsy;
uniform Scalar dsz;

uniform Scalar GM_star;
// Other uniforms types than Scalar or int not yet supported

// BUILTIN
//Scalar dot(...){}

// BUILTIN
//Scalar distance(Vector a, Vector b) { return sqrt(dot(a, b)); }

// BUILTIN
// Scalar first_derivative(Scalar pencil[], Scalar inv_ds) { return pencil[3] * inv_ds; }

Scalar first_derivative(Scalar pencil[], Scalar inv_ds)
{
    Scalar res = 0;
    for (int i = 0; i < STENCIL_ORDER+1; ++i) {
        res = res + pencil[i];
    }
    return inv_ds * res;
}

Scalar distance(Vector a, Vector b)
{
    return sqrt(a.x * b.x + a.y * b.y + a.z * b.z); 
}

Scalar
gravity_potential(int i, int j, int k)
{
    Vector star_pos = (Vector){0, 0, 0};
    Vector vertex_pos = (Vector){dsx * i, dsy * j, dsz * k};
    return GM_star / distance(star_pos, vertex_pos);
}

Scalar
gradx_gravity_potential(int i, int j, int k)
{
    Scalar pencil[STENCIL_ORDER + 1];
    for (int offset = -STENCIL_ORDER; offset <= STENCIL_ORDER; ++offset) {
        pencil[offset+STENCIL_ORDER] = gravity_potential(i + offset, j, k);
    }

    Scalar inv_ds = Scalar(1.) / dsx;
    return first_derivative(pencil, inv_ds);
}

Scalar
grady_gravity_potential(int i, int j, int k)
{
    Scalar pencil[STENCIL_ORDER + 1];
    for (int offset = -STENCIL_ORDER; offset <= STENCIL_ORDER; ++offset) {
        pencil[offset+STENCIL_ORDER] = gravity_potential(i, j + offset, k);
    }

    Scalar inv_ds = Scalar(1.) / dsy;
    return first_derivative(pencil, inv_ds);
}

Scalar
gradz_gravity_potential(int i, int j, int k)
{
    Scalar pencil[STENCIL_ORDER + 1];
    for (int offset = -STENCIL_ORDER; offset <= STENCIL_ORDER; ++offset) {
        pencil[offset+STENCIL_ORDER] = gravity_potential(i, j, k + offset);
    }

    Scalar inv_ds = Scalar(1.) / dsz;
    return first_derivative(pencil, inv_ds);
}

Vector
momentum(int i, int j, int k, in Vector uu)
{

    Vector gravity_potential = (Vector){gradx_gravity_potential(i, j, k),
                                      grady_gravity_potential(i, j, k),
                                      gradz_gravity_potential(i, j, k)};


    return gravity_potential;
}






























































