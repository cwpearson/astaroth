// Declare uniforms (i.e. device constants)
uniform Scalar cs2_sound;
uniform Scalar nu_visc;
uniform Scalar cp_sound;
uniform Scalar cv_sound;
uniform Scalar mu0;
uniform Scalar eta;
uniform Scalar gamma;
uniform Scalar zeta;

uniform Scalar dsx;
uniform Scalar dsy;
uniform Scalar dsz;

uniform Scalar lnT0;
uniform Scalar lnrho0;

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

#if LUPWD
Scalar
upwd_der6(in Vector uu, in Scalar lnrho)
{
    Scalar uux = fabs(value(uu).x);
    Scalar uuy = fabs(value(uu).y);
    Scalar uuz = fabs(value(uu).z);
    return (Scalar){uux*der6x_upwd(lnrho) + uuy*der6y_upwd(lnrho) + uuz*der6z_upwd(lnrho)};
}
#endif

Matrix
gradients(in Vector uu)
{
    return (Matrix){gradient(uu.x), gradient(uu.y), gradient(uu.z)};
}

Scalar
continuity(in Vector uu, in Scalar lnrho) {
    return -dot(value(uu), gradient(lnrho))
#if LUPWD
           //This is a corrective hyperdiffusion term for upwinding.
           + upwd_der6(uu, lnrho)
#endif
           - divergence(uu);
}

#if LENTROPY
Vector
momentum(in Vector uu, in Scalar lnrho, in Scalar ss, in Vector aa) {
    const Matrix S = stress_tensor(uu);
    const Scalar cs2 = cs2_sound * exp(gamma * value(ss) / cp_sound + (gamma - 1) * (value(lnrho) - lnrho0));
    const Vector  j = (Scalar(1.) / mu0) * (gradient_of_divergence(aa) - laplace_vec(aa)); // Current density
    const Vector B = curl(aa);
    //TODO: DOES INTHERMAL VERSTION INCLUDE THE MAGNETIC FIELD?
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
  const Scalar lnT = lnT0 + gamma * value(ss) / cp_sound +
    (gamma - Scalar(1.)) * (value(lnrho) - lnrho0);
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



#if LFORCING
Vector
simple_vortex_forcing(Vector a, Vector b, Scalar magnitude)
{
    return magnitude * cross(normalized(b - a), (Vector){0, 0, 1}); // Vortex
}

Vector
simple_outward_flow_forcing(Vector a, Vector b, Scalar magnitude)
{
    return magnitude * (1 / length(b - a)) * normalized(b - a); // Outward flow
}


// The Pencil Code forcing_hel_noshear(), manual Eq. 222, inspired forcing function with adjustable helicity
Vector
helical_forcing(Scalar magnitude, Vector k_force, Vector xx, Vector ff_re, Vector ff_im, Scalar phi)
{
    // JP: This looks wrong:
    //    1) Should it be dsx * nx instead of dsx * ny?
    //    2) Should you also use globalGrid.n instead of the local n?
    //    MV: You are rigth. Made a quickfix. I did not see the error  because multigpu is split 
    //        in z direction not y direction. 
    //    3) Also final point: can we do this with vectors/quaternions instead?
    //       Tringonometric functions are much more expensive and inaccurate/
    //    MV: Good idea. No an immediate priority.  
    // Fun related article:
    // https://randomascii.wordpress.com/2014/10/09/intel-underestimates-error-bounds-by-1-3-quintillion/
    xx.x = xx.x*(2.0*M_PI/(dsx*globalGrid.n.x));
    xx.y = xx.y*(2.0*M_PI/(dsy*globalGrid.n.y));
    xx.z = xx.z*(2.0*M_PI/(dsz*globalGrid.n.z));

    Scalar cos_phi = cos(phi);
    Scalar sin_phi = sin(phi);
    Scalar cos_k_dox_x     = cos(dot(k_force, xx));
    Scalar sin_k_dox_x     = sin(dot(k_force, xx));
    // Phase affect only the x-component
    //Scalar real_comp       = cos_k_dox_x;
    //Scalar imag_comp       = sin_k_dox_x;
    Scalar real_comp_phase = cos_k_dox_x*cos_phi - sin_k_dox_x*sin_phi;
    Scalar imag_comp_phase = cos_k_dox_x*sin_phi + sin_k_dox_x*cos_phi;


    Vector force = (Vector){ ff_re.x*real_comp_phase - ff_im.x*imag_comp_phase,
                             ff_re.y*real_comp_phase - ff_im.y*imag_comp_phase,
                             ff_re.z*real_comp_phase - ff_im.z*imag_comp_phase};

    return force;
}

Vector
forcing(int3 globalVertexIdx, Scalar dt)
{
    Vector a = Scalar(.5) * (Vector){globalGrid.n.x * dsx,
                                     globalGrid.n.y * dsy,
                                     globalGrid.n.z * dsz}; // source (origin)
    Vector xx = (Vector){(globalVertexIdx.x - nx_min) * dsx,
                        (globalVertexIdx.y - ny_min) * dsy,
                        (globalVertexIdx.z - nz_min) * dsz}; // sink (current index)
    const Scalar cs2 = cs2_sound;
    const Scalar cs = sqrt(cs2);

    //Placeholders until determined properly
    Scalar magnitude = DCONST_REAL(AC_forcing_magnitude);
    Scalar phase     = DCONST_REAL(AC_forcing_phase);
    Vector k_force   = (Vector){  DCONST_REAL(AC_k_forcex),   DCONST_REAL(AC_k_forcey),   DCONST_REAL(AC_k_forcez)};
    Vector ff_re     = (Vector){DCONST_REAL(AC_ff_hel_rex), DCONST_REAL(AC_ff_hel_rey), DCONST_REAL(AC_ff_hel_rez)};
    Vector ff_im     = (Vector){DCONST_REAL(AC_ff_hel_imx), DCONST_REAL(AC_ff_hel_imy), DCONST_REAL(AC_ff_hel_imz)};


    //Determine that forcing funtion type at this point.
    //Vector force = simple_vortex_forcing(a, xx, magnitude);
    //Vector force = simple_outward_flow_forcing(a, xx, magnitude);
    Vector force   = helical_forcing(magnitude, k_force, xx, ff_re,ff_im, phase);

    //Scaling N = magnitude*cs*sqrt(k*cs/dt)  * dt
    const Scalar NN = cs*sqrt(DCONST_REAL(AC_kaver)*cs);
    //MV: Like in the Pencil Code. I don't understandf the logic here.
    force.x = sqrt(dt)*NN*force.x;
    force.y = sqrt(dt)*NN*force.y;
    force.z = sqrt(dt)*NN*force.z;

    if (is_valid(force)) { return force; }
    else                 { return (Vector){0, 0, 0}; }
}
#endif // LFORCING

// Declare input and output arrays using locations specified in the
// array enum in astaroth.h
in Scalar lnrho = VTXBUF_LNRHO;
out Scalar out_lnrho = VTXBUF_LNRHO;

in Vector uu = (int3) {VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ};
out Vector out_uu = (int3) {VTXBUF_UUX,VTXBUF_UUY,VTXBUF_UUZ};


#if LMAGNETIC
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

    #if LMAGNETIC
    out_aa = rk3(out_aa, aa, induction(uu, aa), dt);
    #endif

    #if LENTROPY
        out_uu  = rk3(out_uu, uu, momentum(uu, lnrho, ss, aa), dt);
        out_ss  = rk3(out_ss, ss, entropy(ss, uu, lnrho, aa), dt);
    #elif LTEMPERATURE
        out_uu = rk3(out_uu, uu, momentum(uu, lnrho, tt), dt);
        out_tt = rk3(out_tt, tt, heat_transfer(uu, lnrho, tt), dt);
    #else
        out_uu = rk3(out_uu, uu, momentum(uu, lnrho), dt);
    #endif

    #if LFORCING
    if (step_number == 2) {
        out_uu = out_uu + forcing(globalVertexIdx, dt);
    }
    #endif
}
