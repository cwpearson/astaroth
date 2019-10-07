// PC-style helical forcing with profiles
uniform Vector AC_kk;
uniform Vector AC_coef1;
uniform Vector AC_coef2;
uniform Vector AC_coef3;
uniform Vector AC_fda;

uniform Scalar AC_phase;
uniform Scalar AC_fact;
uniform Scalar AC_k1_ff;

uniform ScalarArray AC_profx;
uniform ScalarArray AC_profy;
uniform ScalarArray AC_profz;
uniform ScalarArray AC_profx_hel;
uniform ScalarArray AC_profy_hel;
uniform ScalarArray AC_profz_hel;

uniform int AC_iforcing_zsym;

// PC-style helical forcing with support for profiles
Vector
pcforcing(int3 vertexIdx, int3 globalVertexIdx, Scalar dt, ScalarArray profx, ScalarArray profy,
          ScalarArray profz, ScalarArray profx_hel, ScalarArray profy_hel, ScalarArray profz_hel)
{
    Vector pos = (Vector){(globalVertexIdx.x - AC_nx_min) * AC_dsx,
                          (globalVertexIdx.y - AC_ny_min) * AC_dsy,
                          (globalVertexIdx.z - AC_nz_min) * AC_dsz};

    Complex fx = AC_fact * exp(Complex(0, AC_kk.x * AC_k1_ff * pos.z + AC_phase));
    Complex fy = exp(Complex(0, AC_kk.y * AC_k1_ff * pos.y));

    Complex fz;
    if (AC_iforcing_zsym == 0) {
        fz = exp(Complex(0.0, AC_kk.z * AC_k1_ff * pos.z));
    }
    else if (AC_iforcing_zsym == 1) {
        fz = Complex(cos(AC_kk.z * AC_k1_ff * pos.z), 0);
    }
    else if (AC_iforcing_zsym == -1) {
        fz = Complex(sin(AC_kk.z * AC_k1_ff * pos.z), 0);
    }
    else {
        // Failure
    }

    Complex fxyz = fx * fy * fz;

    // TODO recheck indices
    Scalar force_ampl    = profx[vertexIdx.x - NGHOST] * profy[vertexIdx.y] * profz[vertexIdx.z];
    Scalar prof_hel_ampl = profx_hel[vertexIdx.x - NGHOST] * profy_hel[vertexIdx.y] *
                           profz_hel[vertexIdx.z];

    return force_ampl * AC_fda * (Complex(AC_coef1.x, prof_hel_ampl * AC_coef2.x) * fxyz).x;
}
