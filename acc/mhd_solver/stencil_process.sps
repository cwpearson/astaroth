#include "stencil_definition.sdh"

Vector
value(in VectorField uu)
{
    return (Vector){value(uu.x), value(uu.y), value(uu.z)};
}

#if LUPWD
Scalar
upwd_der6(in VectorField uu, in ScalarField lnrho)
{
    Scalar uux = fabs(value(uu).x);
    Scalar uuy = fabs(value(uu).y);
    Scalar uuz = fabs(value(uu).z);
    return (Scalar){uux * der6x_upwd(lnrho) + uuy * der6y_upwd(lnrho) + uuz * der6z_upwd(lnrho)};
}
#endif

Matrix
gradients(in VectorField uu)
{
    return (Matrix){gradient(uu.x), gradient(uu.y), gradient(uu.z)};
}

#if LSINK 
Vector
sink_gravity(int3 globalVertexIdx){
    int accretion_switch = int(AC_switch_accretion);
    if (accretion_switch == 1){
        Vector force_gravity;
        const Vector grid_pos = (Vector){(globalVertexIdx.x - DCONST(AC_nx_min)) * AC_dsx,
                                         (globalVertexIdx.y - DCONST(AC_ny_min)) * AC_dsy,
                                         (globalVertexIdx.z - DCONST(AC_nz_min)) * AC_dsz};
        const Scalar sink_mass = AC_M_sink;
        const Vector sink_pos = (Vector){AC_sink_pos_x,
                                         AC_sink_pos_y,
                                         AC_sink_pos_z}; 
        const Scalar distance = length(grid_pos - sink_pos);
        const Scalar soft = AC_soft;
        //MV: The commit 083ff59 had AC_G_const defined wrong here in DSL making it exxessively strong.
        //MV: Scalar gravity_magnitude = ... below is correct! 
        const Scalar gravity_magnitude = (AC_G_const * sink_mass) / pow(((distance * distance) +  soft*soft), 1.5);
        const Vector direction = (Vector){(sink_pos.x - grid_pos.x) / distance,
                                          (sink_pos.y - grid_pos.y) / distance,
                                          (sink_pos.z - grid_pos.z) / distance};
        force_gravity = gravity_magnitude * direction;
        return force_gravity;
    } else {
        return (Vector){0.0, 0.0, 0.0};
    }
}
#endif


#if LSINK
// Give Truelove density
Scalar
truelove_density(in ScalarField lnrho){
    const Scalar rho = exp(value(lnrho));
    const Scalar Jeans_length_squared = (M_PI * AC_cs2_sound) / (AC_G_const * rho);
    const Scalar TJ_rho = ((M_PI) * ((AC_dsx * AC_dsx) / Jeans_length_squared) * AC_cs2_sound) / (AC_G_const * AC_dsx * AC_dsx);
    //TODO: AC_dsx will cancel out, deal with it later for optimization.      
 
    Scalar accretion_rho = TJ_rho;
       
    return accretion_rho;
}

// This controls accretion of density/mass to the sink particle. 
Scalar
sink_accretion(int3 globalVertexIdx, in ScalarField lnrho, Scalar dt){
    const Vector grid_pos = (Vector){(globalVertexIdx.x - DCONST(AC_nx_min)) * AC_dsx,
                                     (globalVertexIdx.y - DCONST(AC_ny_min)) * AC_dsy,
                                     (globalVertexIdx.z - DCONST(AC_nz_min)) * AC_dsz};
    const Vector sink_pos = (Vector){AC_sink_pos_x,
                                     AC_sink_pos_y,
                                     AC_sink_pos_z};
    const Scalar profile_range = AC_accretion_range;
    const Scalar accretion_distance = length(grid_pos - sink_pos);
    int accretion_switch = AC_switch_accretion; 
    Scalar accretion_density; 
    Scalar weight;

    if (accretion_switch == 1){   
        if ((accretion_distance) <= profile_range){
            //weight = Scalar(1.0);
            //Hann window function
            Scalar window_ratio = accretion_distance/profile_range;
            weight = Scalar(0.5)*(Scalar(1.0) - cos(Scalar(2.0)*M_PI*window_ratio));
        } else {
            weight = Scalar(0.0);
        }
        
        //Truelove criterion is used as a kind of arbitrary density floor. 
        const Scalar lnrho_min = log(truelove_density(lnrho));
        Scalar rate;     
        if (value(lnrho) > lnrho_min) {
            rate = (exp(value(lnrho)) - exp(lnrho_min)) / dt;
        } else { 
            rate = Scalar(0.0);
        }
        accretion_density = weight * rate ;
    } else {
        accretion_density = Scalar(0.0); 
    }
    return accretion_density;
} 

// This controls accretion of velocity to the sink particle. 
Vector
sink_accretion_velocity(int3 globalVertexIdx, in VectorField uu, Scalar dt) {
    const Vector grid_pos = (Vector){(globalVertexIdx.x - DCONST(AC_nx_min)) * AC_dsx,
                                     (globalVertexIdx.y - DCONST(AC_ny_min)) * AC_dsy,
                                     (globalVertexIdx.z - DCONST(AC_nz_min)) * AC_dsz};
    const Vector sink_pos = (Vector){AC_sink_pos_x,
                                     AC_sink_pos_y,
                                     AC_sink_pos_z};
    const Scalar profile_range = AC_accretion_range;
    const Scalar accretion_distance = length(grid_pos - sink_pos);   
    int accretion_switch = AC_switch_accretion; 
    Vector accretion_velocity; 

    if (accretion_switch == 1){
        Scalar weight;
        // Step function weighting
        // Arch of a cosine function? 
        // Cubic spline x^3 - x in range [-0.5 , 0.5] 
        if ((accretion_distance) <= profile_range){
            //weight = Scalar(1.0);
            //Hann window function
            Scalar window_ratio = accretion_distance/profile_range;
            weight = Scalar(0.5)*(Scalar(1.0) - cos(Scalar(2.0)*M_PI*window_ratio));
        } else {
            weight = Scalar(0.0);
        }


        Vector rate;     
        // MV: Could we use divergence here ephasize velocitie which are compressive and 
        // MV: not absorbins stuff that would not be accreted anyway? 
        if (length(value(uu)) > Scalar(0.0)) {
            rate = (Scalar(1.0)/dt) * value(uu);
        } else { 
            rate = (Vector){0.0, 0.0, 0.0};
        }
        accretion_velocity = weight * rate ;
    } else {
        accretion_velocity = (Vector){0.0, 0.0, 0.0};
    }
    return accretion_velocity;
}
#endif


Scalar
continuity(int3 globalVertexIdx, in VectorField uu, in ScalarField lnrho, Scalar dt) 
{
    return -dot(value(uu), gradient(lnrho))
#if LUPWD
           // This is a corrective hyperdiffusion term for upwinding.
           + upwd_der6(uu, lnrho)
#endif
#if LSINK
	   - sink_accretion(globalVertexIdx, lnrho, dt) / exp(value(lnrho)) 
#endif
           - divergence(uu);
}



#if LENTROPY
Vector
momentum(int3 globalVertexIdx, in VectorField uu, in ScalarField lnrho, in ScalarField ss, in VectorField aa, Scalar dt) 
{
    const Matrix S   = stress_tensor(uu);
    const Scalar cs2 = AC_cs2_sound * exp(AC_gamma * value(ss) / AC_cp_sound +
                                          (AC_gamma - 1) * (value(lnrho) - AC_lnrho0));
    const Vector j   = (Scalar(1.) / AC_mu0) *
                     (gradient_of_divergence(aa) - laplace_vec(aa)); // Current density
    const Vector B = curl(aa);
    // TODO: DOES INTHERMAL VERSTION INCLUDE THE MAGNETIC FIELD?
    const Scalar inv_rho = Scalar(1.) / exp(value(lnrho));

    // Regex replace CPU constants with get\(AC_([a-zA-Z_0-9]*)\)
    // \1
    const Vector mom = -mul(gradients(uu), value(uu)) -
                       cs2 * ((Scalar(1.) / AC_cp_sound) * gradient(ss) + gradient(lnrho)) +
                       inv_rho * cross(j, B) +
                       AC_nu_visc *
                           (laplace_vec(uu) + Scalar(1. / 3.) * gradient_of_divergence(uu) +
                            Scalar(2.) * mul(S, gradient(lnrho))) +
                       AC_zeta * gradient_of_divergence(uu)
    #if LSINK
                       //Gravity term
                       + sink_gravity(globalVertexIdx) 
                       //Corresponding loss of momentum
                       -     //(Scalar(1.0) / Scalar( (AC_dsx*AC_dsy*AC_dsz) * exp(value(lnrho)))) *  // Correction factor by unit mass
    	                 sink_accretion_velocity(globalVertexIdx, uu, dt) // As in Lee et al.(2014)
                       ; 
    #else
                       ;
    #endif
    return mom;
}
#elif LTEMPERATURE
Vector
momentum(int3 globalVertexIdx, in VectorField uu, in ScalarField lnrho, in ScalarField tt) 
{
    Vector mom;

    const Matrix S = stress_tensor(uu);

    const Vector pressure_term = (AC_cp_sound - AC_cv_sound) *
                                 (gradient(tt) + value(tt) * gradient(lnrho));

    mom = -mul(gradients(uu), value(uu)) - pressure_term +
          AC_nu_visc * (laplace_vec(uu) + Scalar(1. / 3.) * gradient_of_divergence(uu) +
                        Scalar(2.) * mul(S, gradient(lnrho))) +
          AC_zeta * gradient_of_divergence(uu)
    #if LSINK
          + sink_gravity(globalVertexIdx);
    #else
                                         ;
    #endif

#if LGRAVITY
    mom = mom - (Vector){0, 0, -10.0};
#endif
    return mom;
}
#else
Vector
momentum(int3 globalVertexIdx, in VectorField uu, in ScalarField lnrho, Scalar dt) 
{
    Vector mom;

    const Matrix S = stress_tensor(uu);

    // Isothermal: we have constant speed of sound

    mom = -mul(gradients(uu), value(uu)) - AC_cs2_sound * gradient(lnrho) +
          AC_nu_visc * (laplace_vec(uu) + Scalar(1. / 3.) * gradient_of_divergence(uu) +
                        Scalar(2.) * mul(S, gradient(lnrho))) +
          AC_zeta * gradient_of_divergence(uu)
    #if LSINK
          + sink_gravity(globalVertexIdx)
          //Corresponding loss of momentum
          -     //(Scalar(1.0) / Scalar( (AC_dsx*AC_dsy*AC_dsz) * exp(value(lnrho)))) *  // Correction factor by unit mass
    	    sink_accretion_velocity(globalVertexIdx, uu, dt) // As in Lee et al.(2014)
                                              ; 
    #else
                                              ;
    #endif

#if LGRAVITY
    mom = mom - (Vector){0, 0, -10.0};
#endif

    return mom;
}
#endif

Vector
induction(in VectorField uu, in VectorField aa)
{
    // Note: We do (-nabla^2 A + nabla(nabla dot A)) instead of (nabla x (nabla
    // x A)) in order to avoid taking the first derivative twice (did the math,
    // yes this actually works. See pg.28 in arXiv:astro-ph/0109497)
    // u cross B - AC_eta * AC_mu0 * (AC_mu0^-1 * [- laplace A + grad div A ])
    const Vector B        = curl(aa);
    const Vector grad_div = gradient_of_divergence(aa);
    const Vector lap      = laplace_vec(aa);

    // Note, AC_mu0 is cancelled out
    const Vector ind = cross(value(uu), B) - AC_eta * (grad_div - lap);

    return ind;
}

#if LENTROPY
Scalar
lnT(in ScalarField ss, in ScalarField lnrho)
{
    const Scalar lnT = AC_lnT0 + AC_gamma * value(ss) / AC_cp_sound +
                       (AC_gamma - Scalar(1.)) * (value(lnrho) - AC_lnrho0);
    return lnT;
}

// Nabla dot (K nabla T) / (rho T)
Scalar
heat_conduction(in ScalarField ss, in ScalarField lnrho)
{
    const Scalar inv_AC_cp_sound = AcReal(1.) / AC_cp_sound;

    const Vector grad_ln_chi = -gradient(lnrho);

    const Scalar first_term = AC_gamma * inv_AC_cp_sound * laplace(ss) +
                              (AC_gamma - AcReal(1.)) * laplace(lnrho);
    const Vector second_term = AC_gamma * inv_AC_cp_sound * gradient(ss) +
                               (AC_gamma - AcReal(1.)) * gradient(lnrho);
    const Vector third_term = AC_gamma * (inv_AC_cp_sound * gradient(ss) + gradient(lnrho)) +
                              grad_ln_chi;

    const Scalar chi = AC_THERMAL_CONDUCTIVITY / (exp(value(lnrho)) * AC_cp_sound);
    return AC_cp_sound * chi * (first_term + dot(second_term, third_term));
}

Scalar
heating(const int i, const int j, const int k)
{
    return 1;
}

Scalar
entropy(in ScalarField ss, in VectorField uu, in ScalarField lnrho, in VectorField aa)
{
    const Matrix S      = stress_tensor(uu);
    const Scalar inv_pT = Scalar(1.) / (exp(value(lnrho)) * exp(lnT(ss, lnrho)));
    const Vector j      = (Scalar(1.) / AC_mu0) *
                     (gradient_of_divergence(aa) - laplace_vec(aa)); // Current density
    const Scalar RHS = H_CONST - C_CONST + AC_eta * (AC_mu0)*dot(j, j) +
                       Scalar(2.) * exp(value(lnrho)) * AC_nu_visc * contract(S) +
                       AC_zeta * exp(value(lnrho)) * divergence(uu) * divergence(uu);

    return -dot(value(uu), gradient(ss)) + inv_pT * RHS + heat_conduction(ss, lnrho);
}
#endif

#if LTEMPERATURE
Scalar
heat_transfer(in VectorField uu, in ScalarField lnrho, in ScalarField tt)
{
    const Matrix S                  = stress_tensor(uu);
    const Scalar heat_diffusivity_k = 0.0008; // 8e-4;
    return -dot(value(uu), gradient(tt)) + heat_diffusivity_k * laplace(tt) +
           heat_diffusivity_k * dot(gradient(lnrho), gradient(tt)) +
           AC_nu_visc * contract(S) * (Scalar(1.) / AC_cv_sound) -
           (AC_gamma - 1) * value(tt) * divergence(uu);
}
#endif

#if LFORCING
Vector
    simple_vortex_forcing(Vector a, Vector b, Scalar magnitude){
    int accretion_switch = AC_switch_accretion; 
    
    if (accretion_switch == 0){
        return magnitude * cross(normalized(b - a), (Vector){ 0, 0, 1}); // Vortex
    } else { 
        return (Vector){0,0,0};  
    }
}        
Vector
    simple_outward_flow_forcing(Vector a, Vector b, Scalar magnitude){
    int accretion_switch = AC_switch_accretion;
    if (accretion_switch == 0){            
        return magnitude * (1 / length(b - a)) * normalized(b - a); // Outward flow
    } else {
        return (Vector){0,0,0};
    }
}

// The Pencil Code forcing_hel_noshear(), manual Eq. 222, inspired forcing function with adjustable
// helicity
Vector
helical_forcing(Scalar magnitude, Vector k_force, Vector xx, Vector ff_re, Vector ff_im, Scalar phi)
{
    // JP: This looks wrong:
    //    1) Should it be AC_dsx * AC_nx instead of AC_dsx * AC_ny?
    //    2) Should you also use globalGrid.n instead of the local n?
    //    MV: You are rigth. Made a quickfix. I did not see the error  because multigpu is split
    //        in z direction not y direction.
    //    3) Also final point: can we do this with vectors/quaternions instead?
    //       Tringonometric functions are much more expensive and inaccurate/
    //    MV: Good idea. No an immediate priority.
    // Fun related article:
    // https://randomascii.wordpress.com/2014/10/09/intel-underestimates-error-bounds-by-1-3-quintillion/
    xx.x = xx.x * (2.0 * M_PI / (AC_dsx * globalGridN.x));
    xx.y = xx.y * (2.0 * M_PI / (AC_dsy * globalGridN.y));
    xx.z = xx.z * (2.0 * M_PI / (AC_dsz * globalGridN.z));

    Scalar cos_phi     = cos(phi);
    Scalar sin_phi     = sin(phi);
    Scalar cos_k_dot_x = cos(dot(k_force, xx));
    Scalar sin_k_dot_x = sin(dot(k_force, xx));
    // Phase affect only the x-component
    // Scalar real_comp       = cos_k_dot_x;
    // Scalar imag_comp       = sin_k_dot_x;
    Scalar real_comp_phase = cos_k_dot_x * cos_phi - sin_k_dot_x * sin_phi;
    Scalar imag_comp_phase = cos_k_dot_x * sin_phi + sin_k_dot_x * cos_phi;

    Vector force = (Vector){ff_re.x * real_comp_phase - ff_im.x * imag_comp_phase,
                            ff_re.y * real_comp_phase - ff_im.y * imag_comp_phase,
                            ff_re.z * real_comp_phase - ff_im.z * imag_comp_phase};

    return force;
}

Vector
forcing(int3 globalVertexIdx, Scalar dt)
{
    int accretion_switch = AC_switch_accretion;
    if (accretion_switch == 0){

        Vector a = Scalar(.5) * (Vector){globalGridN.x * AC_dsx,
                                         globalGridN.y * AC_dsy,
                                         globalGridN.z * AC_dsz}; // source (origin)
        Vector xx = (Vector){(globalVertexIdx.x - DCONST(AC_nx_min)) * AC_dsx,
                             (globalVertexIdx.y - DCONST(AC_ny_min)) * AC_dsy,
                             (globalVertexIdx.z - DCONST(AC_nz_min)) * AC_dsz}; // sink (current index)
        const Scalar cs2 = AC_cs2_sound;
        const Scalar cs = sqrt(cs2);
    
        //Placeholders until determined properly
        Scalar magnitude = AC_forcing_magnitude;
        Scalar phase     = AC_forcing_phase;
        Vector k_force   = (Vector){AC_k_forcex,   AC_k_forcey,   AC_k_forcez};
        Vector ff_re     = (Vector){AC_ff_hel_rex, AC_ff_hel_rey, AC_ff_hel_rez};
        Vector ff_im     = (Vector){AC_ff_hel_imx, AC_ff_hel_imy, AC_ff_hel_imz};
    
    
        //Determine that forcing funtion type at this point.
        //Vector force = simple_vortex_forcing(a, xx, magnitude);
        //Vector force = simple_outward_flow_forcing(a, xx, magnitude);
        Vector force   = helical_forcing(magnitude, k_force, xx, ff_re,ff_im, phase);
    
        //Scaling N = magnitude*cs*sqrt(k*cs/dt)  * dt
        const Scalar NN = cs*sqrt(AC_kaver*cs);
        //MV: Like in the Pencil Code. I don't understandf the logic here.
        force.x = sqrt(dt)*NN*force.x;
        force.y = sqrt(dt)*NN*force.y;
        force.z = sqrt(dt)*NN*force.z;
    
        if (is_valid(force)) { return force; }
        else                 { return (Vector){0, 0, 0}; }
    } else {
          return (Vector){0,0,0};
    }
}
#endif // LFORCING

// Declare input and output arrays using locations specified in the
// array enum in astaroth.h
in ScalarField lnrho(VTXBUF_LNRHO);
out ScalarField out_lnrho(VTXBUF_LNRHO);

in VectorField uu(VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ);
out VectorField out_uu(VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ);

#if LMAGNETIC
in VectorField aa(VTXBUF_AX, VTXBUF_AY, VTXBUF_AZ);
out VectorField out_aa(VTXBUF_AX, VTXBUF_AY, VTXBUF_AZ);
#endif

#if LENTROPY
in ScalarField ss(VTXBUF_ENTROPY);
out ScalarField out_ss(VTXBUF_ENTROPY);
#endif

#if LTEMPERATURE
in ScalarField tt(VTXBUF_TEMPERATURE);
out ScalarField out_tt(VTXBUF_TEMPERATURE);
#endif

#if LSINK
in ScalarField accretion(VTXBUF_ACCRETION);
out ScalarField out_accretion(VTXBUF_ACCRETION);
#endif

Kernel void
solve()
{
    Scalar dt = AC_dt;
    out_lnrho = rk3(out_lnrho, lnrho, continuity(globalVertexIdx, uu, lnrho, dt), dt);

#if LMAGNETIC
    out_aa = rk3(out_aa, aa, induction(uu, aa), dt);
#endif

#if LENTROPY
    out_uu = rk3(out_uu, uu, momentum(globalVertexIdx, uu, lnrho, ss, aa, dt), dt);
    out_ss = rk3(out_ss, ss, entropy(ss, uu, lnrho, aa), dt);
#elif LTEMPERATURE
    out_uu = rk3(out_uu, uu, momentum(globalVertexIdx, uu, lnrho, tt, dt), dt);
    out_tt = rk3(out_tt, tt, heat_transfer(uu, lnrho, tt), dt);
#else
    out_uu = rk3(out_uu, uu, momentum(globalVertexIdx, uu, lnrho, dt), dt);
#endif

#if LFORCING
    if (step_number == 2) {
        out_uu = out_uu + forcing(globalVertexIdx, dt);
    }
#endif

#if LSINK
    out_accretion = rk3(out_accretion, accretion, sink_accretion(globalVertexIdx, lnrho, dt), dt);// unit now is rho!
          
    if (step_number == 2) {
        out_accretion = out_accretion * AC_dsx * AC_dsy * AC_dsz;// unit is now mass!
    }
#endif
}
