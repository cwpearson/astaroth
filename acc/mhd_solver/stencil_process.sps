#define LINDUCTION (1)
#define LENTROPY (1)
#define LTEMPERATURE (0)
#define LGRAVITY (0)


// Declare uniforms (i.e. device constants)
uniform Scalar cs2_sound;
uniform Scalar nu_visc;
uniform Scalar cp_sound;
uniform Scalar cv_sound;
uniform Scalar mu0;
uniform Scalar eta;
uniform Scalar gamma;
uniform Scalar zeta;

uniform int nx_min;
uniform int ny_min;
uniform int nz_min;
uniform int nx;
uniform int ny;
uniform int nz;

Vector
value(in Vector uu)
{
    return (Vector){value(uu.x), value(uu.y), value(uu.z)};
}

Matrix
gradients(in Vector uu)
{
    return (Matrix){gradient(uu.x), gradient(uu.y), gradient(uu.z)};
}

Scalar
continuity(in Vector uu, in Scalar lnrho) {
    return -dot(value(uu), gradient(lnrho)) - divergence(uu);
}

#if LENTROPY
Vector
momentum(in Vector uu, in Scalar lnrho, in Scalar ss, in Vector aa) {
    const Matrix S = stress_tensor(uu);
    const Scalar cs2 = cs2_sound * exp(gamma * value(ss) / cp_sound + (gamma - 1) * (value(lnrho) - LNRHO0));
    const Vector  j = (Scalar(1.) / mu0) * (gradient_of_divergence(aa) - laplace_vec(aa)); // Current density
    const Vector B = curl(aa);
    const Scalar inv_rho = Scalar(1.) / exp(value(lnrho));

    // Regex replace CPU constants with get\(AC_([a-zA-Z_0-9]*)\)
    // \1
    const Vector mom = - mul(gradients(uu), value(uu)) 
                                                       - cs2 * ((Scalar(1.) / cp_sound) * gradient(ss) + gradient(lnrho))
                                                       + inv_rho * cross(j, B)
                                                       + nu_visc * (
                                                            laplace_vec(uu) 
                                                        + Scalar(1. / 3.) * gradient_of_divergence(uu) 
                                                        + Scalar(2.) * mul(S, gradient(lnrho))
                                                        )
                                                        + zeta * gradient_of_divergence(uu);
    return mom;
}
#elif LTEMPERATURE
Vector
momentum(in Vector uu, in Scalar lnrho, in Scalar tt) {
  Vector mom;

  const Matrix S = stress_tensor(uu);

    const Vector pressure_term = (cp_sound - cv_sound) * (gradient(tt) + value(tt) * gradient(lnrho));

  mom = -mul(gradients(uu), value(uu)) -
    pressure_term +
    nu_visc *
    (laplace_vec(uu) + Scalar(1. / 3.) * gradient_of_divergence(uu) +
      Scalar(2.) * mul(S, gradient(lnrho))) + zeta * gradient_of_divergence(uu);

  #if LGRAVITY
  mom = mom - (Vector){0, 0, -10.0};
  #endif

  return mom;
}
#else
Vector
momentum(in Vector uu, in Scalar lnrho) {
  Vector mom;

  const Matrix S = stress_tensor(uu);

    // Isothermal: we have constant speed of sound

  mom = -mul(gradients(uu), value(uu)) -
    cs2_sound * gradient(lnrho) +
    nu_visc *
    (laplace_vec(uu) + Scalar(1. / 3.) * gradient_of_divergence(uu) +
      Scalar(2.) * mul(S, gradient(lnrho))) + zeta * gradient_of_divergence(uu);

  #if LGRAVITY
  mom = mom - (Vector){0, 0, -10.0};
  #endif

  return mom;
}
#endif


Vector
induction(in Vector uu, in Vector aa) {
  // Note: We do (-nabla^2 A + nabla(nabla dot A)) instead of (nabla x (nabla
  // x A)) in order to avoid taking the first derivative twice (did the math,
  // yes this actually works. See pg.28 in arXiv:astro-ph/0109497)
  // u cross B - ETA * mu0 * (mu0^-1 * [- laplace A + grad div A ])
  const Vector B = curl(aa);
  const Vector grad_div = gradient_of_divergence(aa);
  const Vector lap = laplace_vec(aa);

  // Note, mu0 is cancelled out
  const Vector ind = cross(value(uu), B) - eta * (grad_div - lap);

  return ind;
}


#if LENTROPY
Scalar
lnT( in Scalar ss, in Scalar lnrho) {
  const Scalar lnT = LNT0 + gamma * value(ss) / cp_sound +
    (gamma - Scalar(1.)) * (value(lnrho) - LNRHO0);
  return lnT;
}

// Nabla dot (K nabla T) / (rho T)
Scalar
heat_conduction( in Scalar ss, in Scalar lnrho) {
  const Scalar inv_cp_sound = AcReal(1.) / cp_sound;

  const Vector grad_ln_chi = - gradient(lnrho);

  const Scalar first_term = gamma * inv_cp_sound * laplace(ss) +
    (gamma - AcReal(1.)) * laplace(lnrho);
  const Vector second_term = gamma * inv_cp_sound * gradient(ss) +
    (gamma - AcReal(1.)) * gradient(lnrho);
  const Vector third_term = gamma * (inv_cp_sound * gradient(ss) +
    gradient(lnrho)) + grad_ln_chi;

  const Scalar chi = AC_THERMAL_CONDUCTIVITY / (exp(value(lnrho)) * cp_sound);
  return cp_sound * chi * (first_term + dot(second_term, third_term));
}

Scalar
heating(const int i, const int j, const int k) {
  return 1;
}

Scalar
entropy(in Scalar ss, in Vector uu, in Scalar lnrho, in Vector aa) {
    const Matrix S = stress_tensor(uu);
    const Scalar inv_pT = Scalar(1.) / (exp(value(lnrho)) * exp(lnT(ss, lnrho)));
    const Vector  j = (Scalar(1.) / mu0) * (gradient_of_divergence(aa) - laplace_vec(aa)); // Current density
    const Scalar RHS = H_CONST - C_CONST
                                                + eta * (mu0) * dot(j, j) 
                                                + Scalar(2.) * exp(value(lnrho)) * nu_visc * contract(S)
                                                + zeta * exp(value(lnrho)) * divergence(uu) * divergence(uu);

    return - dot(value(uu), gradient(ss))
                  + inv_pT * RHS
                  + heat_conduction(ss, lnrho);
}
#endif

#if LTEMPERATURE
Scalar
heat_transfer(in Vector uu, in Scalar lnrho, in Scalar tt)
{
    const Matrix S = stress_tensor(uu);
    const Scalar heat_diffusivity_k = 0.0008; //8e-4;
    return -dot(value(uu), gradient(tt)) + heat_diffusivity_k * laplace(tt) + heat_diffusivity_k * dot(gradient(lnrho), gradient(tt)) + nu_visc * contract(S) * (Scalar(1.) / cv_sound) - (gamma - 1) * value(tt) * divergence(uu);
}
#endif

// Declare input and output arrays using locations specified in the
// array enum in astaroth.h
in Scalar lnrho = VTXBUF_LNRHO;
out Scalar out_lnrho = VTXBUF_LNRHO;

in Vector uu = (int3) {VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ};
out Vector out_uu = (int3) {VTXBUF_UUX,VTXBUF_UUY,VTXBUF_UUZ};


#if LINDUCTION
in Vector aa = (int3) {VTXBUF_AX,VTXBUF_AY,VTXBUF_AZ};
out Vector out_aa = (int3) {VTXBUF_AX,VTXBUF_AY,VTXBUF_AZ};
#endif

#if LENTROPY
in Scalar ss = VTXBUF_ENTROPY;
out Scalar out_ss = VTXBUF_ENTROPY;
#endif

#if LTEMPERATURE
in Scalar tt = VTXBUF_TEMPERATURE;
out Scalar out_tt = VTXBUF_TEMPERATURE;
#endif

Kernel void
solve(Scalar dt) {
    out_lnrho = rk3(out_lnrho, lnrho, continuity(uu, lnrho), dt);

    #if LINDUCTION
    out_aa = rk3(out_aa, aa, induction(uu, aa), dt);
    #endif

    #if LENTROPY
        out_uu = rk3(out_uu, uu, momentum(uu, lnrho, ss, aa), dt);
        out_ss  = rk3(out_ss, ss, entropy(ss, uu, lnrho, aa), dt);
    #elif LTEMPERATURE
        out_uu =rk3(out_uu, uu, momentum(uu, lnrho, tt), dt);
        out_tt = rk3(out_tt, tt, heat_transfer(uu, lnrho, tt), dt);
    #else
        out_uu = rk3(out_uu, uu, momentum(uu, lnrho), dt);
    #endif    
}










































