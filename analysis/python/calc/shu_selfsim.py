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

G_newton = 6.674e-8 #cm**3 g**-1 s**-2  

def dv_dx(xx,vv, alpha):
    EE = alpha*(xx-vv) - 2.0/xx 
    HH = (xx-vv)**2.0 - 1.0
    return (EE/HH)*(xx-vv)

def dalpha_dx(xx,vv, alpha):
    EE = alpha*(alpha - (2.0/xx)*(xx-vv))
    HH = (xx-vv)**2.0 - 1.0
    return (EE/HH)*(xx-vv)

###def dv_dx(xx,vv, alpha):
###    return 2.0*(xx-vv)
###
###def dalpha_dx(xx,vv, alpha):
###    return -1.0*(xx-vv)

def get_m(xx, vv, alpha): 
    mm = xx**2.0 * alpha * (xx-vv)
    return mm 

def alpha_to_rho(alpha, tt):
    rho = alpha/(4.0*np.pi*G_newton*(tt**2.0))
    return rho

def vv_to_uu(vv, cs0):
    uu = cs0*vv
    return uu

def mm_to_MM(mm, tt, cs0):
    MM = (((cs0**3.0)*tt)/G_newton)*mm
    return MM

def euler(xx_step, xx, vv, alpha, mm, target):
    diff = target - xx[-1]  
    if diff >= 0:         
        while xx[-1] <= target:
            vv_step    = vv[-1]    + xx_step*dv_dx(xx[-1], vv[-1], alpha[-1])
            alpha_step = alpha[-1] + xx_step*dalpha_dx(xx[-1], vv[-1], alpha[-1])
        
            xx = np.append(xx, xx[-1]+xx_step)
            alpha = np.append(alpha, alpha_step)
            vv = np.append(vv, vv_step)
            mm_step    = get_m(xx[-1], vv[-1], alpha[-1])
            mm = np.append(mm, mm_step)
    else: 
        while xx[-1] <= target:
            vv_step    = vv[-1]    + xx_step*dv_dx(xx[-1], vv[-1], alpha[-1])
            alpha_step = alpha[-1] + xx_step*dalpha_dx(xx[-1], vv[-1], alpha[-1])
        
            xx = np.append(xx, xx[-1]+xx_step)
            alpha = np.append(alpha, alpha_step)
            vv = np.append(vv, vv_step)
            mm_step    = get_m(xx[-1], vv[-1], alpha[-1])
            mm = np.append(mm, mm_step)
    return xx, vv, alpha, mm

def RK4_step(vv, xx, alpha, xx_step): 
    vv1    =     xx_step*dv_dx(xx[-1], vv[-1], alpha[-1]) 
    alpha1 = xx_step*dalpha_dx(xx[-1], vv[-1], alpha[-1])
    
    vv2 =        xx_step*dv_dx(xx[-1]+xx_step/2.0, vv[-1]+vv1/2.0, alpha[-1]+alpha1/2.0)
    alpha2 = xx_step*dalpha_dx(xx[-1]+xx_step/2.0, vv[-1]+vv1/2.0, alpha[-1]+alpha1/2.0)
    
    vv3 =        xx_step*dv_dx(xx[-1]+xx_step/2.0, vv[-1]+vv2/2.0, alpha[-1]+alpha2/2.0)
    alpha3 = xx_step*dalpha_dx(xx[-1]+xx_step/2.0, vv[-1]+vv2/2.0, alpha[-1]+alpha2/2.0)
    
    vv4 =        xx_step*dv_dx(xx[-1]+xx_step, vv[-1]+vv3, alpha[-1]+alpha3)
    alpha4 = xx_step*dalpha_dx(xx[-1]+xx_step, vv[-1]+vv3, alpha[-1]+alpha3)
    
    vv_step    = vv[-1]    + (1.0/6.0)*(vv1 + 2.0*vv2 + 2.0*vv3 + vv4) 
    alpha_step = alpha[-1] + (1.0/6.0)*(alpha1 + 2.0*alpha2 + 2.0*alpha3 + alpha4)

    return vv_step, alpha_step

def RK4(xx_step, xx, vv, alpha, mm, target, epsilon):
    #Runge-Kutta RK4
    diff = target - xx[-1]  
    #if diff < 0: 

    if diff >= 0:         
        while xx[-1] <= target:
            if (np.abs(xx[-1] - vv[-1] - 1.0) > epsilon):
                vv_step, alpha_step = RK4_step(vv, xx, alpha, xx_step)
                print( vv_step, alpha_step)
            else: 
                vv_step    = vv[-1]
                alpha_step = alpha[-1]
                print("PIIP") 

            #print(np.abs(xx[-1] - vv[-1]), epsilon)
 
            xx = np.append(xx, xx[-1]+xx_step)
            alpha = np.append(alpha, alpha_step)
            vv = np.append(vv, vv_step)
            mm_step    = get_m(xx[-1], vv[-1], alpha[-1])
            mm = np.append(mm, mm_step)
    else:         
        while xx[-1] >= target:
            if (np.abs(xx[-1] - vv[-1] - 1.0) > epsilon):
                vv_step, alpha_step = RK4_step(vv, xx, alpha, xx_step)
                print( vv_step, alpha_step)
            else: 
                vv_step    = vv[-1]
                alpha_step = alpha[-1]
                print("PIIP") 

            #print(np.abs(xx[-1] - vv[-1]), epsilon)
 
            xx = np.append(xx, xx[-1]+xx_step)
            alpha = np.append(alpha, alpha_step)
            vv = np.append(vv, vv_step)
            mm_step    = get_m(xx[-1], vv[-1], alpha[-1])
            mm = np.append(mm, mm_step)
            

    return xx, vv, alpha, mm

# From Shu 1977 TABLE II

xx_SHU    =  np.array([0.05 , 0.10 , 0.15 , 0.20 , 0.25 , 0.30 , 0.35 , 0.40 , 0.45 ,
		       0.50 , 0.55 , 0.60 , 0.65 , 0.70 , 0.75 , 0.80 , 0.85 ,
                       0.90 , 0.95 , 1.00]) 
alpha_SHU =  np.array([71.5 , 27.8 , 16.4 , 11.5 , 8.76 , 7.09 , 5.95 , 5.14 , 4.52 ,
		       4.04 , 3.66 , 3.35 , 3.08 , 2.86 , 2.67 , 2.50 , 2.35 ,
                       2.22 , 2.10 , 2.00]) 
vv_SHU    = -np.array([5.44 , 3.47 , 2.58 , 2.05 , 1.68 , 1.40 , 1.18 , 1.01 , 0.861,
		       0.735, 0.625, 0.528, 0.442, 0.363, 0.291, 0.225, 0.163,
                       0.106, 0.051, 0.00]) 
mm_SHU    =  np.array([0.981, 0.993, 1.01 , 1.03 , 1.05 , 1.08 , 1.12 , 1.16 , 1.20 ,
		       1.25 , 1.30 , 1.36 , 1.42 , 1.49 , 1.56 , 1.64 , 1.72 ,
                       1.81 , 1.90 , 2.00]) 


##From Shu (1977)
#AA = [  2.0,  2.2,  2.4,  2.6,  2.8,  3.0,  3.2,  3.4,  3.6,  3.8, 4.0]
#m0 = [0.975, 1.45, 1.88, 2.31, 2.74, 3.18, 3.63, 4.10, 4.58, 5.08, 5.58]
#AA = np.array(AA)
#m0 = np.array(m0)

#xx0    = xx_SHU[1] 
#alpha0 = alpha_SHU[1] 
#vv0    = vv_SHU[1]
#xx_step = 0.005
#target = 1.0

xx0    = xx_SHU[-3] 
alpha0 = alpha_SHU[-3] 
vv0    = vv_SHU[-3]
target = 0.05
xx_step = -0.005
xx_step = -0.001
             
print(get_m(xx0, alpha0, vv0))

xx = np.array([])
alpha = np.array([])
vv = np.array([])
mm = np.array([])

xx = np.append(xx, xx0)
alpha = np.append(alpha, alpha0)
vv = np.append(vv, vv0)
mm = np.append(mm, get_m(xx0, alpha0, vv0))

print(xx, alpha, vv, mm)


xx_EUL, vv_EUL, alpha_EUL, mm_EUL = euler(xx_step, xx, vv, alpha, mm, target)
xx_RK , vv_RK , alpha_RK , mm_RK  = RK4(xx_step, xx, vv, alpha, mm, target, epsilon = 0.000001)

mm_EUL = get_m(xx_EUL, alpha_EUL, vv_EUL)
mm_RK  = get_m(xx_RK , alpha_RK , vv_RK )
mm_SHU = get_m(xx_SHU, alpha_SHU, vv_SHU)

# Plotting time
 
figQ, axQ = plt.subplots(nrows=2, ncols=2, sharex=True)

axQ[0,0].plot(xx_EUL, alpha_EUL, label=r'$\alpha$ (Euler)', linewidth = 3.0)
axQ[0,0].plot(xx_RK , alpha_RK , label=r'$\alpha$ (RK4)', linewidth = 3.0)
axQ[0,0].plot(xx_SHU, alpha_SHU, 'd', label=r'$\alpha$ (Shu)', linewidth = 3.0)
axQ[0,0].set_xlabel(r'x')
axQ[0,0].set_ylabel(r'$\alpha$')
axQ[0,0].legend()

axQ[0,1].plot(xx_EUL, np.abs(vv_EUL), label='v (Euler)', linewidth = 3.0)
axQ[0,1].plot(xx_RK , np.abs(vv_RK ), label='v (RK4)', linewidth = 3.0)
axQ[0,1].plot(xx_SHU, np.abs(vv_SHU),'d', label='v (Shu)', linewidth = 3.0)
axQ[0,1].set_xlabel(r'x')
axQ[0,1].set_ylabel(r'-v')
axQ[0,1].legend()

axQ[1,0].plot(xx_EUL, mm_EUL, label='m (Euler)', linewidth = 3.0)
axQ[1,0].plot(xx_RK , mm_RK , label='m (RK4)', linewidth = 3.0)
axQ[1,0].plot(xx_SHU , mm_SHU , 'd', label='m (Shu)', linewidth = 3.0)
axQ[1,0].set_xlabel(r'x')
axQ[1,0].set_ylabel(r'm')
axQ[1,0].legend()


axQ[1,1].plot(xx_EUL, xx_EUL-vv_EUL, label='x-v (Euler)', linewidth = 3.0)
axQ[1,1].plot(xx_RK , xx_RK -vv_RK , label='x-v (RK4)', linewidth = 3.0)
axQ[1,1].plot(xx_SHU, xx_SHU-vv_SHU, 'd', label='x-v (Shu)', linewidth = 3.0)
axQ[1,1].set_xlabel(r'x')
axQ[1,1].set_ylabel(r'x-v')
axQ[1,1].legend()

# Time to convert to physical quantities
yr  = 3.154e+7 #s 
kyr = 1000.0*yr
km = 1e5 #cm
AU = 1.496e+13 #cm
Msun = 1.98847e33 #g

cs0 = 20000   #cs cm/s "a" in Shu notation

tt_list = np.linspace(10*kyr, 20.0*kyr, num=4)
mm = get_m(xx_RK, vv_RK, alpha_RK) 


fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True)

for tt in tt_list:
    rho = alpha_to_rho(alpha_RK, tt)
    RR = xx_RK*(cs0*tt)
    time = r'%.2f $\mathrm{kyr}$' % (tt/kyr) 
    
    ax[0].plot(RR/AU, rho, label= r'$\rho$, t = ' + time, linewidth = 3.0)
    ax[0].set_xlabel(r'R (AU)')
    ax[0].set_ylabel(r'$\rho$ (g/cm$^3$)')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].legend()

    uu = vv_to_uu(vv_RK, cs0)

    ax[1].plot(RR/AU, -uu/km, label= r'$u$, t = ' + time, linewidth = 3.0)
    ax[1].set_xlabel(r'R (AU)')
    ax[1].set_ylabel(r'-$u$ (km/s)')
    ax[1].set_yscale('log')
    ax[1].legend()

    MM = mm_to_MM(mm, tt, cs0)

    ax[2].plot(RR/AU, MM/Msun, label= r'$M$, t = ' + time, linewidth = 3.0)
    ax[2].set_xlabel(r'R (AU)')
    ax[2].set_ylabel(r'$M$ ($M_\odot}$)')
    ax[2].legend()

   

plt.show()




