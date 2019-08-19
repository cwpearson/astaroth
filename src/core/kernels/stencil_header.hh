#define LDENSITY (1)
#define LHYDRO (1)
#define LMAGNETIC (1)
#define LENTROPY (1)
#define LTEMPERATURE (0)
#define LFORCING (1)
#define LUPWD (1)

#define AC_THERMAL_CONDUCTIVITY (AcReal(0.001)) // TODO: make an actual config parameter

// Declare uniforms (i.e. device constants)
uniform Scalar AC_cs2_sound;
uniform Scalar AC_nu_visc;
uniform Scalar AC_cp_sound;
uniform Scalar AC_cv_sound;
uniform Scalar AC_mu0;
uniform Scalar AC_eta;
uniform Scalar AC_gamma;
uniform Scalar AC_zeta;

uniform Scalar AC_dsx;
uniform Scalar AC_dsy;
uniform Scalar AC_dsz;

uniform Scalar AC_lnT0;
uniform Scalar AC_lnrho0;

uniform int AC_nx_min;
uniform int AC_ny_min;
uniform int AC_nz_min;
uniform int AC_nx;
uniform int AC_ny;
uniform int AC_nz;
