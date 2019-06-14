'''
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
'''
import numpy as np
import pylab as plt
import scipy as scp

import matplotlib.colors as colors

G_newton = 6.674e-8 #cm**3 g**-1 s**-2  

# Time to convert to physical quantities
yr  = 3.154e+7 #s 
kyr = 1000.0*yr
km = 1e5 #cm
AU = 1.496e+13 #cm
Msun = 1.98847e33 #g

#cs0 = 20000.0   #cs cm/s "a" in Shu notation
cs0 = 35000.0   #cs cm/s "a" in Shu notation
B0  = 30e-6   #G 
ksii = 11.3 # 

#GS Eq. 10
ttm = 9.03e12*(cs0/35000.0)/(B0/30e-6) 


CM_INFERNO = plt.cm.get_cmap('inferno')






def P_harmonics(theta, J=666):
    #Vector spherical harmonics in e_r direction
    if J == 0: 
        P = np.ones_like(theta)  # 1.0 
    elif J == 2:
        cos_theta = np.cos(theta)
        P = (1.0/2.0)*(3.0*(cos_theta**2.0) - 1.0)
    else:
        P = 0.0
  
    #print("P_2", P) 
    return P 
    

def B_harmonics(theta, J=666):
    #Vector spherical harmonics in e_theta direction
    #print("B_harmonics theta", theta)
    if J == 2:
        sin_theta = np.abs(np.sin(theta))
        cos_theta = np.cos(theta)
        #B = -(3.0/np.sqrt(6.0))*cos_theta*sin_theta #Morse & Feshbach 1953 book
        B = -3.0*cos_theta*sin_theta #GS93 Appendix B
    else:
        B = 0.0*theta

    #print("B_harmonics", B)
   
    return B 

def get_tau(tt): 
    return tt/ttm

def get_SHU77_potential(xx_point):
    #Copied here again for convenience
    m0 = 0.975 #Shu 77 core reduced mass
    xx_SHU_table   = np.array([ 0.05,  0.10,  0.15,  0.20,  0.25, 
                          0.30,  0.35,  0.40,  0.45,  0.50, 
                          0.55,  0.60,  0.65,  0.70,  0.75, 
                          0.80,  0.85,  0.90,  0.95,  1.00]) 
    
    mm_SHU77_table = np.array([0.981, 0.993,  1.01,  1.03,  1.05, 
                          1.08,  1.12,  1.16,  1.20,  1.25, 
                          1.30,  1.36,  1.42,  1.49,  1.56, 
                          1.64,  1.72,  1.81,  1.90,  2.00]) 
 
    xx = xx_SHU_table[  np.where(xx_SHU <= xx_point)]
    mm = mm_SHU77_table[np.where(xx_SHU <= xx_point)]

    psi = - m0/xx_point + np.trapz(mm/(xx**2.0), xx)

    return psi


def psi2(xx_SHU, mm_term, pp_term, J=666):
    #GS93 Eq. 113
    if J == 0: 
        psi2 = - mm_term/xx_SHU + pp_term
    elif J == 2:
        psi2 = - mm_term/(xx_SHU**3.0) + (xx_SHU**2.0)*pp_term 
    else:
        psi2 = 0.0

    #print('psi2', psi2, 'J', J, 'mm_term', mm_term, 'xx_SHU', xx_SHU, 'pp_term', pp_term)

    return psi2

# Calculate the directional parameter
def dv_dx(xx,vv, alpha):
    EE = alpha*(xx-vv) - 2.0/xx 
    HH = (xx-vv)**2.0 - 1.0
    return (EE/HH)*(xx-vv)

def dalpha_dx(xx,vv, alpha):
    EE = alpha*(alpha - (2.0/xx)*(xx-vv))
    HH = (xx-vv)**2.0 - 1.0
    return (EE/HH)*(xx-vv)

def dpsi_dx(xx, mm):
    return mm/(xx**2.0)

def dmm_dx(xx, alpha):
    return (xx**2.0)*alpha

def dphi_dx(xx, alpha, mm, theta):
    ff_zero_der = 0.5*mm*dmm_dx(xx, alpha)
    sin_theta = np.sin(theta)
    return ff_zero_der*(sin_theta*2.0) 


def deltaspace(theta, tau):
    #Assuming J= 0, 2 only
    v0 = -2.222e-1
    v2 = 2.177e-1
    deltaJ2 = -(1.0/3.0)*((v0+2.0/3.0)*P_harmonics(theta, J=0) + (v2 - 2.0/3.0)*P_harmonics(theta, J=2))
    delta   = 1 + (tau**2.0)*deltaJ2 
    return delta

def delta2(theta, tau):
    #Assuming J= 0, 2 only
    return deltaspace(theta, tau)**2.0

def yy_transform(xx_SHU, alpha_SHU77, alpha_mono_GS93, alpha_quad_GS93):
    
    

    return alpha_mono_GS93, alpha_quad_GS93 

# Calculating the perturbation stage
def alpha_perturb(tau, xx_SHU, vv_SHU77, alpha_SHU77, alpha_mono_GS93, alpha_quad_GS93, theta):
    #Assuming J= 0, 2 only
    directional = xx_SHU*dalpha_dx(xx_SHU, vv_SHU77, alpha_SHU77)*delta2(theta, tau)
    directional = 0.0 # 
    alpha       = alpha_mono_GS93*P_harmonics(theta, J=0) + alpha_quad_GS93*P_harmonics(theta, J=2) + directional
    return alpha

def vv_perturb(tau, xx_SHU, vv_SHU77, alpha_SHU77, vv_ww_mono_GS93, vv_ww_quad_GS93, theta):
    #Assuming J= 0, 2 only
    directional = xx_SHU*dv_dx(xx_SHU, vv_SHU77, alpha_SHU77)*delta2(theta, tau)
    directional = 0.0 # 
    vv_mono  = vv_ww_mono_GS93[0]
    vv_quad  = vv_ww_quad_GS93[0]
    ww_mono  = vv_ww_mono_GS93[1]
    ww_quad  = vv_ww_quad_GS93[1]
    #print('vv_mono, vv_quad, ww_mono, ww_quad', vv_mono, vv_quad, ww_mono, ww_quad)
    vv_r     = vv_mono*P_harmonics(theta, J=0) + vv_quad*P_harmonics(theta, J=2) + directional ## vv
    vv_theta = ww_mono*B_harmonics(theta, J=0) + ww_quad*B_harmonics(theta, J=2) + directional ## ww
    #print("vv_r, vv_theta", vv_r, vv_theta)
    vv       = np.array([vv_r, vv_theta])
    return vv

def psi_perturb(tau, xx_SHU, mm_SHU77, mm_pp_mono_GS93, mm_pp_quad_GS93, theta):
    #Assuming J= 0, 2 only
    directional = xx_SHU*dpsi_dx(xx_SHU, mm_SHU77)*delta2(theta, tau)
    directional = 0.0 # 
    mm_mono  = mm_pp_mono_GS93[0]
    mm_quad  = mm_pp_quad_GS93[0]
    pp_mono  = mm_pp_mono_GS93[1]
    pp_quad  = mm_pp_quad_GS93[1]

    #print('mm_pp_mono_GS93', mm_pp_mono_GS93)
    #print('mm_mono', mm_mono)
    
    psi      =   psi2(xx_SHU, mm_mono, pp_mono, J=0)*P_harmonics(theta, J=0) \
               + psi2(xx_SHU, mm_quad, pp_quad, J=0)*P_harmonics(theta, J=2) \
               + directional
    
    #print('psi_perturb', psi)
 
    return psi

def phi_vecpot_second_order(tau, xx_SHU, mm_SHU77, alpha_SHU77, FF_DD_mono_GS93, FF_DD_quad_GS93, theta):
    directional = xx_SHU*dphi_dx(xx_SHU, alpha_SHU77, mm_SHU77, theta)*delta2(theta, tau)
    directional = 0.0 # 
    sin_theta = np.sin(theta)
    #print(FF_DD_mono_GS93)
    #print(FF_DD_quad_GS93)
    #print(ksii, P_harmonics(theta, J=0), P_harmonics(theta, J=2))
    mono_term = (FF_DD_mono_GS93[0] + (1.0/ksii)*FF_DD_mono_GS93[1])
    quad_term = (FF_DD_quad_GS93[0] + (1.0/ksii)*FF_DD_quad_GS93[1])
    phi_vecpot_second = (sin_theta**2.0)*( mono_term*P_harmonics(theta, J=0) \
                                           + quad_term*P_harmonics(theta, J=2) ) \
                                           + directional
    return phi_vecpot_second

def phi_vecpot_zero_order(xx_SHU, mm_SHU77, theta):
    ff_zero = 0.25*(mm_SHU77**2.0)
    sin_theta = np.sin(theta)
    phi_vecpot_zero = ff_zero*(sin_theta*2.0)
    return phi_vecpot_zero


# Combining the perturbation stage.
def alpha_xvec_tau(tau, xx_SHU, vv_SHU77, alpha_SHU77, alpha_mono_GS93, alpha_quad_GS93, theta):
    alpha = alpha_SHU77 + (tau**2.0)*alpha_perturb(tau, xx_SHU, vv_SHU77, alpha_SHU77, alpha_mono_GS93, alpha_quad_GS93, theta)
    return alpha

def vv_xvec_tau(tau, xx_SHU, vv_SHU77, alpha_SHU77, vv_ww_mono_GS93, vv_ww_quad_GS93, theta):
    vv = (tau**2.0)*vv_perturb(tau, xx_SHU, vv_SHU77, alpha_SHU77, vv_ww_mono_GS93, vv_ww_quad_GS93, theta)
    #print("BF",vv, vv_ww_mono_GS93, vv_ww_quad_GS93) 
    vv[0] = vv_SHU77 + vv[0]
    vv[1] = 0.0      + vv[1]   #No poloidal velocity in Shu77
    #print("AF",vv)
    return vv 

def psi_xvec_tau(tau, xx_SHU, mm_SHU77, mm_pp_mono, mm_pp_quad, theta):
    #print("psi_xvec_tau --- tau, xx_SHU, mm_SHU7, mm_pp_mono, mm_pp_quad, theta", tau, xx_SHU, mm_SHU77, mm_pp_mono, mm_pp_quad, theta)
    psi = (tau**2.0)*psi_perturb(tau, xx_SHU, mm_SHU77, mm_pp_mono, mm_pp_quad, theta)
    psi77 = get_SHU77_potential(xx_SHU)
    #print('psi77', psi77)
    psi = psi77 + psi  
    #print('psi_xvec_tau', psi)
    return psi 


def phi_vecpot_xvec_tau(tau, xx_SHU, mm_SHU77, alpha_SHU77, FF_DD_mono_GS93, FF_DD_quad_GS93, theta):
    phi_vecpot_second = (tau**2.0)*phi_vecpot_second_order(tau, xx_SHU, mm_SHU77, alpha_SHU77, FF_DD_mono_GS93, FF_DD_quad_GS93, theta)
    phi_vecpot_zero = phi_vecpot_zero_order(xx_SHU, mm_SHU77, theta)
    phi_vecpot = phi_vecpot_zero + phi_vecpot_second 
    return phi_vecpot 

#Physical unit converion stage
def rho_rt(tt, xx_SHU, vv_SHU77, alpha_SHU77, alpha_mono_GS93, alpha_quad_GS93, theta):
    tau = get_tau(tt)
    alpha_xvec = alpha_xvec_tau(tau, xx_SHU, vv_SHU77, alpha_SHU77, alpha_mono_GS93, alpha_quad_GS93, theta)
    rho = (1.0/(4.0*np.pi*G_newton*(tt**2.0))) * alpha_xvec
    return rho, alpha_xvec

def uu_rt(tt, xx_SHU, vv_SHU77, alpha_SHU77, vv_ww_mono_GS93, vv_ww_quad_GS93, theta):
    tau = get_tau(tt)
    vv_xvec = vv_xvec_tau(tau, xx_SHU, vv_SHU77, alpha_SHU77, vv_ww_mono_GS93, vv_ww_quad_GS93, theta)
    uu = cs0*vv_xvec
    return uu, vv_xvec

def grav_psi_rt(tt, xx_SHU, mm_SHU77, mm_pp_mono, mm_pp_quad, theta):
    tau = get_tau(tt)
    #print("tt , xx_SHU, mm_SHU77, mm_pp_mono, mm_pp_quad, theta", tt, xx_SHU, mm_SHU77, mm_pp_mono, mm_pp_quad, theta)
    psi_xvec = psi_xvec_tau(tau, xx_SHU, mm_SHU77, mm_pp_mono, mm_pp_quad, theta)
    Vpot     = (cs0**2.0)*psi_xvec
    return Vpot, psi_xvec

def vectorpot_rt(tt, xx_SHU, mm_SHU77, alpha_SHU77, FF_DD_mono_GS93, FF_DD_quad_GS93, theta):
    tau = get_tau(tt)
    phi_vecpot_xvec = phi_vecpot_xvec_tau(tau, xx_SHU, mm_SHU77, alpha_SHU77, FF_DD_mono_GS93, FF_DD_quad_GS93, theta)
    Phi_flux = np.pi*B0*((cs0*tt)**2.0)*phi_vecpot_xvec
    return Phi_flux, phi_vecpot_xvec



###def match_xx(xx_rad, xx_SHU):
###    xx_buffer = np.empty_like(xx_rad)
###    stride = np.abs(xx_SHU[1] - xx_SHU[0])
###    for xx in xx_SHU:
###        #where  xx - stride <  xx_rad < xx + stride   -> xx_rad[i] = xx 
###        #loc = np.where((xx_rad <= (xx + stride) and xx_rad > (xx - stride) ))
###        loc = np.where(xx_rad <= (xx + stride) )
###        print(loc)


def get_shu_index(xx, xx_SHU):
    stride = np.abs(xx_SHU[1] - xx_SHU[0])/2.0

    #ishu = np.where((xx_SHU <= (xx + stride)) & (xx_SHU > (xx - stride)))[0]    


    #TODO Now a purkka version. Do better. 
    # Can be improve by taking the treatment of the actual low and high x cases. 
    if (xx > xx_SHU[xx_SHU.size-1]):
        ishu = xx_SHU.size-1 
    elif (xx < xx_SHU[0]):
        ishu = 0
    else:
        ishu = np.where((xx_SHU <= (xx + stride)) & (xx_SHU > (xx - stride)))[0]
        #print("get_shu_index", ishu, ishu.size)
        ishu = ishu[0]
        #print("get_shu_index", ishu, ishu.size)

    #print(ishu, xx_SHU[ishu], xx)

    return ishu

def plot_figure(tt, xx_horizontal_corners, xx_vertical_corners, xx_horizontal, xx_vertical, xxvar, physvar, 
                vv_hor=np.array(None), vv_ver=np.array(None), uu_hor=np.array(None), uu_ver=np.array(None), 
                title1=r"\alpha", title2=r"\rho", filetitle='density',
                var_min=[None, None], var_max=[None, None], colmap=CM_INFERNO, normtype='log', 
                streamlines = 0, contourplot = 0):

    if var_min[0] != None:
        if normtype == 'log':
            mynorm1 = colors.LogNorm( vmin=var_min[0], vmax=var_max[0] )
            mynorm2 = colors.LogNorm( vmin=var_min[1], vmax=var_max[1] )
        else:
            mynorm1 = colors.Normalize( vmin=var_min[0], vmax=var_max[0] )
            mynorm2 = colors.Normalize( vmin=var_min[1], vmax=var_max[1] )
    else:
        mynorm1 = colors.Normalize( )
        mynorm2 = colors.Normalize( )

    if contourplot: 
        if normtype =='cdensity':
            numbers = np.arange(0, 20, dtype=np.float64)
            contourlevs = 1e-20*(np.sqrt(2.0)**numbers)
            contournorm = colors.LogNorm( vmin=contourlevs.min(), vmax=contourlevs.max() )
        elif normtype =='cflux':
            contourlevs = np.linspace(1.0, 1e31, num=20)
            contournorm = colors.Normalize( vmin=contourlevs.min(), vmax=contourlevs.max() )
        else: 
            contourlevs = np.linspace(physvar.min(), physvar.max(), num=10)
            contournorm = colors.Normalize( vmin=contourlevs.min(), vmax=contourlevs.max() )


    ##rr_horizontal_corners = xx_horizontal_corners*(cs0*tt)/AU
    ##rr_vertical_corners   = xx_vertical_corners*  (cs0*tt)/AU
    ##rr_horizontal         = xx_horizontal*(cs0*tt)/AU
    ##rr_vertical           = xx_vertical*  (cs0*tt)/AU

    rr_horizontal_corners = xx_horizontal_corners*(cs0*tt)/1e17
    rr_vertical_corners   = xx_vertical_corners*  (cs0*tt)/1e17
    rr_horizontal         = xx_horizontal*(cs0*tt)/1e17
    rr_vertical           = xx_vertical*  (cs0*tt)/1e17



    figa, axa = plt.subplots(nrows=1, ncols=2, figsize=(16,6))
    if contourplot:
        mapa = axa[0].contourf(xx_horizontal, xx_vertical, xxvar, cmap=colmap, norm=mynorm1)
        maprho = axa[1].contourf(rr_horizontal, rr_vertical, physvar, contourlevs, cmap=colmap, norm=contournorm)
    else: 
        mapa = axa[0].pcolormesh(xx_horizontal_corners, xx_vertical_corners, xxvar, cmap=colmap, norm=mynorm1 )
        maprho = axa[1].pcolormesh(rr_horizontal_corners, rr_vertical_corners, physvar, cmap=colmap, norm=mynorm2)

    #mapa = axa[0].contourf(xx_horizontal, xx_vertical, alpha, cmap=CM_INFERNO, norm=colors.LogNorm(vmin=0.1, vmax=50.0))
    #maprho = axa[1].contourf(xx_horizontal*(cs0*tt)/AU, xx_vertical*(cs0*tt)/AU, rho, cmap=CM_INFERNO, norm=colors.LogNorm(vmin=1e15, vmax=1e20))

    if vv_hor.any() != None:
        if streamlines:
            #vv_tot = np.sqrt(vv_hor**2.0 + vv_ver**2.0)
            #vv_tot = np.log(vv_tot/vv_tot.max())
            axa[0].streamplot(xx_horizontal, xx_vertical, vv_hor, vv_ver, color  = 'k')
            axa[1].streamplot(rr_horizontal, rr_vertical, uu_hor, uu_ver, color = 'k' )
        else:
            axa[0].quiver(xx_horizontal, xx_vertical, vv_hor, vv_ver, pivot = 'middle')
            axa[1].quiver(rr_horizontal, rr_vertical, uu_hor, uu_ver, pivot = 'middle')

    fig.colorbar(mapa, ax=axa[0])
    fig.colorbar(maprho, ax=axa[1])

    tau    = get_tau(tt)
    tt_kyr = tt/kyr
    axa[0].set_title(r'$%s(x, \tau = %.3f)$ ' % (title1, tau))
    axa[1].set_title(r'$%s(r, t = %.3f \mathrm{kyr})$ ' % (title2, tt_kyr))

    axa[0].set_xlabel('x')
    axa[0].set_ylabel('x')
    #axa[1].set_xlabel('r (AU)')
    #axa[1].set_ylabel('r (AU)')
    axa[1].set_xlabel(r'r ($10^{17}$ cm)')
    axa[1].set_ylabel(r'r ($10^{17}$ cm)' )

    ##axa[1].set_xlim(0.0, 3e17/AU)
    ##axa[1].set_ylim(0.0, 3e17/AU)
    axa[1].set_xlim(0.0, 3.0)
    axa[1].set_ylim(0.0, 3.0)

    axa[0].set_aspect('equal', 'datalim')
    #axa[1].set_aspect('equal', 'datalim')

    figfile = '%s_%s.png' % (filetitle, str(numslice).zfill(6))
    print(figfile)
    figa.savefig(figfile)
    plt.close(figa)



xx_SHU      =  np.array([ 0.05,  0.10,  0.15,  0.20,  0.25, 
                          0.30,  0.35,  0.40,  0.45,  0.50, 
                          0.55,  0.60,  0.65,  0.70,  0.75, 
                          0.80,  0.85,  0.90,  0.95,  1.00]) 

alpha_SHU77 =  np.array([ 71.5,  27.8,  16.4,  11.5,  8.76, 
                          7.09,  5.95,  5.14,  4.52,  4.04, 
                          3.66,  3.35,  3.08,  2.86,  2.67, 
                          2.50,  2.35,  2.22,  2.10,  2.00]) 

vv_SHU77    = -np.array([ 5.44,  3.47,  2.58,  2.05,  1.68, 
                          1.40,  1.18,  1.01, 0.861, 0.735, 
                         0.625, 0.528, 0.442, 0.363, 0.291, 
                         0.225, 0.163, 0.106, 0.051,  0.00]) 

mm_SHU77    =  np.array([0.981, 0.993,  1.01,  1.03,  1.05, 
                          1.08,  1.12,  1.16,  1.20,  1.25, 
                          1.30,  1.36,  1.42,  1.49,  1.56, 
                          1.64,  1.72,  1.81,  1.90,  2.00]) 




#GS Table 1 

alpha_mono_GS93 = np.array([    6.304,     2.600,     1.652,     1.156,  9.005e-1, 
                             7.314e-1,  6.084e-1,  5.084e-1,  4.256e-1,  3.517e-1, 
                             2.829e-1,  2.172e-1,  1.488e-1,  8.091e-2,  8.360e-3, 
                            -6.826e-2, -1.512e-1, -2.406e-1, -3.382e-1, -4.444e-1]) 

vv_ww_mono_GS93 = np.array([[4.372e-1,  3.335e-1,  2.390e-1,  1.918e-1,  1.522e-1,
                             1.226e-1,  9.579e-2,  7.103e-2,  4.828e-2,  2.640e-2, 
                             5.058e-3, -1.588e-2, -3.791e-2, -5.975e-2, -8.293e-2,
                            -1.071e-1, -1.330e-1, -1.605e-1, -1.902e-1, -2.222e-1],
                           [      0.0,       0.0,       0.0,       0.0,       0.0,
                                  0.0,       0.0,       0.0,       0.0,       0.0,
                                  0.0,       0.0,       0.0,       0.0,       0.0, 
                                  0.0,       0.0,       0.0,       0.0,       0.0]])

mm_pp_mono_GS93 = np.array([[8.634e-4, 1.959e-3, 3.560e-3, 5.661e-3, 8.235e-3,
                             1.130e-2, 1.482e-2, 1.873e-2, 2.293e-2, 2.730e-2,
                             3.166e-2, 3.579e-2, 3.935e-2, 4.196e-2, 4.312e-2,
                             4.221e-2, 3.847e-2, 3.097e-2, 1.859e-2,      0.0],
                           [      0.0,      0.0,      0.0,      0.0,      0.0,
                                  0.0,      0.0,      0.0,      0.0,      0.0,
                                  0.0,      0.0,      0.0,      0.0,      0.0,
                                  0.0,      0.0,      0.0,      0.0,      0.0]])


FF_DD_mono_GS93 = np.array([[   -1.130, -3.275e-1, -1.355e-1, -6.415e-2, -2.889e-2, #F
                             -8.387e-3,  5.358e-3,  1.534e-2,  2.303e-2,  2.931e-2,
                              3.454e-2,  3.888e-2,  4.225e-2,  4.442e-2,  4.504e-2,
                              4.358e-2,  3.935e-2,  3.146e-2,  1.881e-2,      0.0],
                           [  -1.246e1,    -3.168,    -1.141, -5.740e-1, -3.178e-1, #D
                             -1.878e-1, -1.049e-1, -4.547e-2,  3.393e-4,  3.924e-2,
                              7.431e-2,  1.070e-1,  1.376e-1,  1.650e-1,  1.867e-1,
                              1.992e-1,  1.966e-1,  1.708e-1,  1.103e-1,       0.0]])



#GS Table 2

alpha_quad_GS93 = np.array([ -1.096e3, -1.191e2,  -3.148e1,  -1.158e1,    -5.105, 
                               -2.456,   -1.217, -5.889e-1, -2.569e-1, -7.024e-2, 
                             3.790e-2, 1.042e-1,  1.505e-1,  1.845e-1,  2.163e-1, 
                             2.492e-1, 2.865e-1,  3.302e-1,  3.823e-1,  4.437e-1])

vv_ww_quad_GS93 = np.array([[  -2.581,    -1.533, -8.072e-1, -5.666e-1, -3.905e-1, #v
                            -2.790e-1, -1.928e-1, -1.254e-1, -7.156e-2, -2.614e-2, 
                             1.267e-2,  4.650e-2,  7.724e-2,  1.042e-1,  1.288e-1,
                             1.510e-1,  1.711e-1,  1.889e-1,  2.045e-1,  2.177e-1],
                           [   -2.085,    -4.890,    -1.811, -8.842e-1, -4.816e-1, #w
                            -2.807e-1, -1.628e-1, -8.779e-2, -3.852e-2, -4.481e-3,
                             1.928e-2,  3.578e-2,  4.683e-2,  5.306e-2,  5.512e-2, 
                             5.312e-2,  4.704e-2,  3.670e-2,  2.179e-2,  1.898e-3]])

mm_pp_quad_GS93 = np.array([[-3.860e-5, -1.541e-4, -3.044e-4, -4.847e-4, -6.831e-4, #m
                             -8.874e-4, -1.083e-3, -1.253e-3, -1.385e-3, -1.462e-3,
                             -1.470e-3, -1.389e-3, -1.191e-3, -8.405e-4, -2.841e-4,
                              5.579e-4,  1.800e-3,  3.609e-3,  6.218e-3,  9.951e-3],
                            [ -7.539e1,    -7.275,    -1.730, -5.586e-1, -1.999e-1, #p
                             -6.591e-1, -1.062e-2,  1.294e-2,  2.267e-2,  2.600e-2,
                              2.625e-2,  2.500e-2,  2.294e-2,  2.046e-2,  1.769e-2,
                              1.469e-2,  1.146e-2,  7.941e-3,  4.102e-3, -1.214e-4]])

FF_DD_quad_GS93 = np.array([[   -2.253, -6.517e-1, -2.722e-1, -1.345e-1, -6.993e-2, #F
                             -3.593e-2, -1.660e-2, -5.864e-3, -6.809e-4,  8.213e-4,
                             -3.086e-4, -3.338e-3, -7.681e-3, -1.272e-2, -1.778e-2,
                             -2.191e-2, -2.392e-2, -2.219e-2, -1.457e-2,  1.729e-3],
                            [ -2.484e1,    -6.258,    -2.221,    -1.102, -6.127e-1, #D
                             -3.645e-1, -2.213e-1, -1.297e-1, -7.020e-2, -1.112e-2,
                             -2.139e-3, -1.615e-2,  2.744e-2,  3.252e-2,  3.269e-2,
                              2.839e-2,  2.104e-2,  1.199e-2,  3.732e-3,       0.0]])


tt = 0.3*ttm
theta = 0.5*np.pi


xx_SHU          = xx_SHU[:-1]  
vv_SHU77        = vv_SHU77[:-1]
alpha_SHU77     = alpha_SHU77[:-1]

alpha_mono_GS93 = alpha_mono_GS93[:-1]
alpha_quad_GS93 = alpha_quad_GS93[:-1]

vv_ww_mono_GS93 = np.array([vv_ww_mono_GS93[0][:-1], vv_ww_mono_GS93[1][:-1]])
vv_ww_quad_GS93 = np.array([vv_ww_quad_GS93[0][:-1], vv_ww_quad_GS93[1][:-1]])


rho, alpha_xvec = rho_rt(tt, xx_SHU, vv_SHU77, alpha_SHU77, alpha_mono_GS93, alpha_quad_GS93, theta)

rr = xx_SHU*cs0*tt 

np.set_printoptions(linewidth=200)

print(rho.shape)
print(xx_SHU.shape)

print(rho)
print(xx_SHU)

print(vv_ww_mono_GS93)
print(vv_ww_quad_GS93)
print(vv_ww_quad_GS93[0])
print(vv_ww_quad_GS93[1])

#plt.figure()
#plt.plot(rr, rho)
#
#plt.figure()
#plt.plot(xx_SHU, alpha_xvec, label = "GS93")
#plt.plot(xx_SHU, alpha_SHU77, label = "Shu77")
#plt.legend()


#alpha_mono_yy, alpha_quad_yy, alpha_mono_yy = yy_transform(xx_SHU, alpha_SHU77, alpha_mono_GS93, alpha_quad_GS93)


plt.figure()
plt.plot(xx_SHU, alpha_SHU77, label=r"$\alpha^{(0)}$")
plt.plot(xx_SHU, alpha_mono_GS93, label=r"$\alpha^{(2)}_0$")
plt.plot(xx_SHU, alpha_quad_GS93, label=r"$\alpha^{(2)}_2$")
plt.ylim([-5.0,5.0])
plt.legend()
plt.show()


'''
ii = 0
theta_axis = np.linspace(0.0, np.pi)
xx_theta = np.array([])

print("PIIP")


plt.figure()
for ii in range(0,xx_SHU.size):
    alpha_theta  = np.array([])
    alpha_shuref = np.array([])
    for theta in theta_axis: 
        rho, alpha_xvec = rho_rt(tt, xx_SHU[ii], vv_SHU77[ii], alpha_SHU77[ii], alpha_mono_GS93[ii], alpha_quad_GS93[ii])
        alpha_theta  = np.append(alpha_theta, alpha_xvec)
        alpha_shuref = np.append(alpha_shuref, alpha_SHU77[ii])

    plt.plot(alpha_theta, theta_axis, label = "GS93")
    #plt.plot(alpha_shuref, theta_axis, label = "GS93")
'''


#Interpolate a mesh. 

xx_SHU_GRID = np.insert(xx_SHU, 0, 0.0)
print(xx_SHU_GRID)

xx_horizontal, xx_vertical = np.meshgrid(xx_SHU_GRID, xx_SHU_GRID,  indexing='xy') 
theta = np.arctan2(xx_horizontal, xx_vertical)

#Take pcolormesh coordinate system into account, which marks corners instead of centre points. 
dxx = np.abs(xx_horizontal[0,1] - xx_horizontal[0,0])
    
print(dxx)
xx_horizontal_corners = xx_horizontal - dxx/2.0
xx_vertical_corners   = xx_vertical - dxx/2.0 

xx_rad = np.sqrt(xx_horizontal**2.0 +  xx_vertical**2.0)




fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,4))
        
map1 = ax[0].pcolormesh(xx_horizontal_corners, xx_vertical_corners, theta)
map2 = ax[1].pcolormesh(xx_horizontal_corners, xx_vertical_corners, xx_rad)

ax[0].set_title(r"$\theta$")
ax[1].set_title(r"$x_\mathrm{rad}$")

fig.colorbar(map1, ax=ax[0])
fig.colorbar(map2, ax=ax[1])

ax[0].set_aspect('equal', 'datalim')
ax[1].set_aspect('equal', 'datalim')




Pfig, Pax = plt.subplots(nrows=1, ncols=3, figsize=(16,4))

print("P_harmonics(theta, J=0)", P_harmonics(theta, J=0))

Pmap1 = Pax[0].pcolormesh(xx_horizontal_corners, xx_vertical_corners, P_harmonics(theta, J=0))
Pmap2 = Pax[1].pcolormesh(xx_horizontal_corners, xx_vertical_corners, P_harmonics(theta, J=2))
Pmap3 = Pax[2].pcolormesh(xx_horizontal_corners, xx_vertical_corners, deltaspace(theta, 0.5))

Pax[0].set_title(r"$P_0(\theta)$")
Pax[1].set_title(r"$P_2(\theta)$")
Pax[2].set_title(r"$\Delta(\theta, \tau = 0.5)$")


Pfig.colorbar(Pmap1, ax=Pax[0])
Pfig.colorbar(Pmap2, ax=Pax[1])
Pfig.colorbar(Pmap3, ax=Pax[2])

Pax[0].set_aspect('equal', 'datalim')
Pax[1].set_aspect('equal', 'datalim')
Pax[2].set_aspect('equal', 'datalim')




Bfig, Bax = plt.subplots(nrows=1, ncols=2, figsize=(16,4))

print("B_harmonics(theta, J=0)", B_harmonics(theta, J=0))

Bmap1 = Bax[0].pcolormesh(xx_horizontal_corners, xx_vertical_corners, B_harmonics(theta, J=0))
Bmap2 = Bax[1].pcolormesh(xx_horizontal_corners, xx_vertical_corners, B_harmonics(theta, J=2))

Bax[0].set_title(r"$B_0(\theta)$")
Bax[1].set_title(r"$B_2(\theta)$")

Bfig.colorbar(Bmap1, ax=Bax[0])
Bfig.colorbar(Bmap2, ax=Bax[1])

Bax[0].set_aspect('equal', 'datalim')
Bax[1].set_aspect('equal', 'datalim')


plt.show()



##xx_horizontal_corners = np.append(xx_horizontal_corners, (np.amax(xx_horizontal_corners)+dxx)*np.ones((xx_horizontal_corners.shape[1],1)), axis=1)

print(xx_horizontal_corners[-1,:])
print(xx_horizontal_corners)

##xx_horizontal_corners = np.vstack((xx_horizontal_corners, xx_horizontal_corners[-1,:]))
##print(xx_horizontal_corners)

##xx_vertical_corners   = np.append(xx_vertical_corners,   (np.amax(xx_vertical_corners)+dxx)*np.ones((1,xx_vertical_corners.shape[0])),   axis=0)

print(xx_vertical_corners[:, -1])
print(xx_vertical_corners)
##xx_vertical_corners   =  np.hstack((xx_vertical_corners, xx_vertical_corners[:,-1])) 
print(xx_vertical_corners)

numslice = 0
frametot = 201
#frametot = 101
#frametot = 11
for tt in np.linspace(0.1, ttm, num=frametot):
    
    alpha      = np.empty_like(xx_rad)
    alpha77    = np.empty_like(xx_rad)
    rho        = np.empty_like(xx_rad)

    vv_rad     = np.empty_like(xx_rad)
    vv_pol     = np.empty_like(xx_rad)
    uu_rad     = np.empty_like(xx_rad)
    uu_pol     = np.empty_like(xx_rad)

    psi        = np.empty_like(xx_rad)
    Vpot       = np.empty_like(xx_rad)

    Delta      = np.empty_like(xx_rad)

    Phi_flux = np.empty_like(xx_rad)
    phi_vecpot     = np.empty_like(xx_rad)


    alpha_2_J  = np.empty_like(xx_rad)

    for ii in range(xx_SHU_GRID.size):
        for kk in range(xx_SHU_GRID.size):
            xx    = xx_rad[ii,kk]
            th    = theta[ii,kk]
            ishu  = get_shu_index(xx, xx_SHU)
            rho[ii, kk], alpha[ii, kk] = rho_rt(tt, xx_SHU[ishu],
                                                vv_SHU77[ishu],
                                                alpha_SHU77[ishu],
                                                alpha_mono_GS93[ishu],
                                                alpha_quad_GS93[ishu], th)
            alpha77[ii, kk] = alpha_SHU77[ishu]

            vv_ww_mono_point = vv_ww_mono_GS93[:, ishu]
            vv_ww_quad_point = vv_ww_quad_GS93[:, ishu]
            uu_dump, vv_dump =  uu_rt(tt, xx_SHU[ishu], vv_SHU77[ishu], alpha_SHU77[ishu], vv_ww_mono_point, vv_ww_quad_point, th)
            vv_rad[ii, kk]  = vv_dump[0] 
            vv_pol[ii, kk]  = vv_dump[1] 
            uu_rad[ii, kk] = uu_dump[0] 
            uu_pol[ii, kk] = uu_dump[1] 

            mm_pp_mono_point = mm_pp_mono_GS93[:, ishu]
            mm_pp_quad_point = mm_pp_quad_GS93[:, ishu]
            Vpot[ii, kk], psi[ii, kk] = grav_psi_rt(tt, xx_SHU[ishu], mm_SHU77[ishu], mm_pp_mono_point, mm_pp_quad_point, th)

            Phi_flux[ii, kk], phi_vecpot[ii, kk] = vectorpot_rt(tt, xx_SHU[ishu], mm_SHU77[ishu], alpha_SHU77[ishu], 
                                                                FF_DD_mono_GS93[:, ishu], 
                                                                FF_DD_quad_GS93[:, ishu], th)

            Delta[ii, kk] = deltaspace(th, get_tau(tt))
            alpha_2_J[ii, kk] = alpha_mono_GS93[ishu]*P_harmonics(th, J=0) + alpha_quad_GS93[ishu]*P_harmonics(th, J=2) 


    vv_hor =   vv_pol*np.cos(theta) + vv_rad*np.sin(theta)
    vv_ver = - vv_pol*np.sin(theta) + vv_rad*np.cos(theta)
    uu_hor =   uu_pol*np.cos(theta) + uu_rad*np.sin(theta)
    uu_ver = - uu_pol*np.sin(theta) + uu_rad*np.cos(theta)


    rho77 = alpha77 * (1.0/(4.0*np.pi*G_newton)*tt) #TODO WRONG COEFFS!!! 


    #Apply mask
    
    rad_mask = 0.2

   
    alpha = np.ma.masked_where(xx_rad < rad_mask, alpha)
    rho   = np.ma.masked_where(xx_rad < rad_mask, rho)

    vv_rad = np.ma.masked_where(xx_rad < rad_mask, vv_rad) 
    uu_rad = np.ma.masked_where(xx_rad < rad_mask, uu_rad) 
    vv_pol = np.ma.masked_where(xx_rad < rad_mask, vv_pol) 
    uu_pol = np.ma.masked_where(xx_rad < rad_mask, uu_pol) 

    vv_hor = np.ma.masked_where(xx_rad < rad_mask, vv_hor)
    vv_ver = np.ma.masked_where(xx_rad < rad_mask, vv_ver)
    uu_hor = np.ma.masked_where(xx_rad < rad_mask, uu_hor)
    uu_ver = np.ma.masked_where(xx_rad < rad_mask, uu_ver)

    psi  = np.ma.masked_where(xx_rad < rad_mask, psi )
    Vpot = np.ma.masked_where(xx_rad < rad_mask, Vpot)

    phi_vecpot = np.ma.masked_where(xx_rad < rad_mask, phi_vecpot)
    Phi_flux   = np.ma.masked_where(xx_rad < rad_mask, Phi_flux  )

    alpha_2_J = np.ma.masked_where(xx_rad < rad_mask, alpha_2_J)
    Delta     = np.ma.masked_where(xx_rad < rad_mask, Delta    )

    plot_figure(tt, xx_horizontal_corners, xx_vertical_corners, xx_horizontal, xx_vertical, alpha, rho, 
                vv_hor=vv_hor, vv_ver=vv_ver, uu_hor=uu_hor, uu_ver=uu_ver,
                title1=r"\alpha", title2=r"\rho", filetitle='GS93density',
                streamlines = 1, contourplot=1, 
                var_min=[0.00, 1e15], var_max=[16, 1e21], 
                normtype = 'cdensity')

    plot_figure(tt, xx_horizontal_corners, xx_vertical_corners, xx_horizontal, xx_vertical, alpha77, rho77, 
                #var_min=[0.00, 0], var_max=[16, 1e20], 
                title1=r"\alpha", title2=r"\rho", filetitle='S77density')

    plot_figure(tt, xx_horizontal_corners, xx_vertical_corners, xx_horizontal, xx_vertical, vv_rad, uu_rad, 
                vv_hor=vv_hor, vv_ver=vv_ver, uu_hor=uu_hor, uu_ver=uu_ver,
                title1=r"v_r", title2=r"u_r", filetitle='GS93velocity_rad',
                var_min=[-2.5, -2.5*cs0], var_max=[0.0, 0.0*cs0], 
                normtype = 'lin')

    plot_figure(tt, xx_horizontal_corners, xx_vertical_corners, xx_horizontal, xx_vertical, vv_pol, uu_pol, 
                vv_hor=vv_hor, vv_ver=vv_ver, uu_hor=uu_hor, uu_ver=uu_ver,
                title1=r"v_\theta", title2=r"u_\theta", filetitle='GS93velocity_pol',
                var_min=[0.0, 0.0*cs0], var_max=[0.5, 0.5*cs0], 
                normtype = 'lin')

    
    plot_figure(tt, xx_horizontal_corners, xx_vertical_corners, xx_horizontal, xx_vertical, psi, Vpot, 
                vv_hor=vv_hor, vv_ver=vv_ver, uu_hor=uu_hor, uu_ver=uu_ver,
                title1=r"\psi", title2=r"V_\mathrm{pot}", filetitle='GS93gravpot',
                var_min=[12.0, 12.0*(cs0**2.0)], var_max=[21.0, 21.0*(cs0**2.0)], 
                normtype = 'lin')

    plot_figure(tt, xx_horizontal_corners, xx_vertical_corners, xx_horizontal, xx_vertical, phi_vecpot, Phi_flux, 
                title1=r"\phi", title2=r"\Phi_\mathrm{flux}", filetitle='GS93vecpot',
                vv_hor=vv_hor, vv_ver=vv_ver, uu_hor=uu_hor, uu_ver=uu_ver,
                streamlines = 1, contourplot=1,
                normtype = 'cflux')

    plot_figure(tt, xx_horizontal_corners, xx_vertical_corners, xx_horizontal, xx_vertical, np.sqrt(vv_hor**2.0 + vv_ver**2.0), np.sqrt(uu_hor**2.0 + uu_ver**2.0), 
                title1=r"|v|", title2=r"|u| (cm/s)", filetitle='GS93vel2',
                var_min=[0.0, 0.0*cs0], var_max=[2.5, 2.5*cs0], 
                vv_hor=vv_hor, vv_ver=vv_ver, uu_hor=uu_hor, uu_ver=uu_ver,
                streamlines = 1,  
                normtype = 'lin')

    
    ##plot_figure(tt, xx_horizontal_corners, xx_vertical_corners, xx_horizontal, xx_vertical, Delta, Delta,  
    ##            title1=r"\Delta", title2=r"\Delta", filetitle='Delta',
    ##            normtype = 'lin')

    ##plot_figure(tt, xx_horizontal_corners, xx_vertical_corners, xx_horizontal, xx_vertical, alpha_2_J, alpha_2_J,  
    ##            title1=r"\sum \alpha^{(2)}_J", title2=r"\sum \alpha^{(2)}_J", filetitle='alpha_2_J', 
    ##            normtype = 'lin')

    numslice += 1 



















