#include "stencil_definition.sdh"

Vector
value(in VectorField uu)
{
    return (Vector){value(uu.x), value(uu.y), value(uu.z)};
}

in VectorField uu(VTXBUF_A, VTXBUF_B, VTXBUF_C);
out VectorField out_uu(VTXBUF_A, VTXBUF_B, VTXBUF_C);

Kernel void
solve()
{
    Scalar dt = AC_dt;
    Vector rate_of_change = (Vector){1, 2, 3};
    out_uu = rk3(out_uu, uu, rate_of_change, dt);
}
