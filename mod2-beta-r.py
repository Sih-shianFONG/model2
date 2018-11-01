"""Second model of molecular cloud with changing of Plummer scale and distance of binary star  """

import itertools

import numpy as np

from numba import autojit

from math import pi

from multiprocessing.dummy import Pool as ThreadPool

import csv


G = 1.0 #gravitational constant
m1 = float(raw_input('mass 1 :'))  # mass of star 1 
m2 = float(raw_input('mass 2 :'))  # mass of star 2
ismdensity = float(raw_input('density of ism :'))  # density of interstellar medium
mb0 = float(raw_input('mb initial mass i :'))  # mass of molecular cloud 
r01 = float(raw_input('r01:'))  # initial distanve 
r02 = float(raw_input('r02:'))  # final distance
r0step =  int(raw_input('r0step:')) # step of distance
r0 = np.linspace(r01, r02, r0step)
R0 = float(raw_input('initial plummer scale i:')) #initial Plummer scale
Rf = ( 3 * mb0 / ( 4 * pi * ismdensity ) ) ** ( 1. / 3 ) # final Plummer scale 
N = int(raw_input('totstep:'))  #total steps
s = float(raw_input('sstep:'))  #time steps
t0 = float(raw_input('time0:'))  #initial time
beta1 = float(raw_input('beta1:')) #multiple numer of velocity 1
beta2 = float(raw_input('beta2:'))  #multiple numer of velocity 1
betastep = int(raw_input('betastep:')) 
beta = np.linspace(beta1, beta2, betastep)
tau = float(raw_input('tau:'))  #changing time

 
        
@autojit

def f1(t, q1, q2, q3, p1, p2, p3):  # gravity 
    
    r = ( q1 ** 2 + q2 ** 2 + q3 ** 2 ) ** 0.5  #distance of binary star in spherical coordinate
    
    r12 = ( ( q1 - p1 ) ** 2 + ( q2 - p2 ) ** 2 + ( q3 - p3 ) ** 2 ) ** 0.5 # relative distance
    
    rism = f3(t) * ( ( Rf / f3 (t) ) ** ( 1.2 ) - 1. ) ** 0.5  # radius where the density of cloud is equal to the interstellar's
    
    mc = mb0 * ( ( ( rism ** 2 ) / ( rism ** 2 + f3(t) ** 2 ) ) ** 2.5 ) #effective cloud mass
     
    if r <= rism :
        
        return -G * m1 * ( q1 - p1 ) / ( r12 ** 3 ) - ( G * mb0 * q1 / ( ( f3(t) ** 2 + r ** 2 ) ** 1.5 ) - 4 * pi * G * ismdensity * q1 / 3. )
    
    else:
        
        return - G * m1 * ( q1 - p1 ) / ( r12 ** 3 ) - G * mc * q1 / ( r ** 3 )
    




@autojit

def f3(t): # expansion of molecular cloud
    
    if t <= tau:  #linear expansion in durial time
        
        return R0 + ( Rf - R0 ) * ( t - t0 ) / ( tau - t0 )
    
    else :
        
        return Rf


@autojit
def func(tmx):
    

    j,i= tmx[0],tmx[1]

    x1 = np.array([j]) ; y1 = np.array([0.0]) ; z1 = np.array([0.0]) #star 1
    x2 = np.array([-j]) ; y2 = np.array([0.0]) ; z2 = np.array([0.0]) #star 2, they are symmetrical
    r1 = ( x1 ** 2 + y1 ** 2 + z1 ** 2 ) ** 0.5
    r2 = (x2 ** 2 + y2 ** 2 + z2 ** 2 ) ** 0.5
    vx1 = np.array([0.0]) #velocity(x) of star 1
    vy1 = i * (G * (mb0 * (r1 ** 3) / ((R0 ** 2 + r1 ** 2) ** 1.5) - 4.0 * pi * ismdensity * (r1 ** 3) / 3 + m2 / 4.0 ) / r1) ** 0.5  #velocity(y) of star 1
    vz1 = np.array([0.0]) #velocity(z) of star 1
    vx2 = np.array([0.0]) 
    vy2 = - i * (G * (mb0 * (r2**3) / ((R0 **2 + r2**2)**1.5) -4.0*pi*ismdensity*(r2**3)/3+ m1 / 4.0 ) / r2)**0.5
    vz2 = np.array([0.0])
    t = np.array([t0])

    for k in xrange(N):  #rungekutta

        k1a = s * f1(t, x1, y1, z1, x2, y2, z2)
        k1b = s * vx1
        k1c = s * f1(t, y1, z1, x1, y2, z2, x2)
        k1d = s * vy1
        k1e = s * f1(t, z1, x1, y1, z2, x2, y2)
        k1f = s * vz1
        k1g = s * f1(t, x2, y2, z2, x1, y1, z1)
        k1h = s * vx2
        k1i = s * f1(t, y2, z2, x2, y1, z1, x1)
        k1j = s * vy2
        k1k = s * f1(t, z2, x2, y2, z1, x1, y1)
        k1l = s * vz2

        k2a = s * f1(t + 0.5 * s, x1 + 0.5 * k1b, y1 + 0.5 * k1d , z1 + 0.5 * k1f, x2 + 0.5 * k1h, y2 + 0.5 * k1j, z2 + 0.5 * k1l)
        k2b = s * (vx1 + 0.5 * k1a)
        k2c = s * f1(t + 0.5 * s, y1 + 0.5 * k1d, z1 + 0.5 * k1f, x1 + 0.5 * k1b,y2 + 0.5 * k1j, z2 + 0.5 * k1l, x2 + 0.5 * k1h)
        k2d = s * (vy1 + 0.5 * k1c)
        k2e = s * f1(t + 0.5 * s,z1 + 0.5 * k1f, x1 + 0.5 * k1b, y1 + 0.5 * k1d,z2 + 0.5 * k1l, x2 + 0.5 * k1h , y2 + 0.5 * k1j)
        k2f = s * (vz1 + 0.5 * k1e)
        k2g = s * f1(t + 0.5 * s,  x2 + 0.5 * k1h, y2 + 0.5 * k1j, z2 + 0.5 * k1l,x1 + 0.5 * k1b, y1 + 0.5 * k1d , z1 + 0.5 * k1f)
        k2h = s * (vx2 + 0.5 * k1g)
        k2i = s * f1(t + 0.5 * s,  y2 + 0.5 * k1j, z2 + 0.5 * k1l, x2 + 0.5 * k1h,y1 + 0.5 * k1d, z1 + 0.5 * k1f, x1 + 0.5 * k1b)
        k2j = s * (vy2 + 0.5 * k1i)
        k2k = s * f1(t + 0.5 * s, z2 + 0.5 * k1l, x2 + 0.5 * k1h, y2 + 0.5 * k1j,z1 + 0.5 * k1f, x1 + 0.5 * k1b, y1 + 0.5 * k1d)
        k2l = s * (vz2 + 0.5 * k1k)

        k3a = s * f1(t + 0.5 * s, x1 + 0.5 * k2b, y1 + 0.5 * k2d , z1 + 0.5 * k2f, x2 + 0.5 * k2h, y2 + 0.5 * k2j, z2 + 0.5 * k2l)
        k3b = s * (vx1 + 0.5 * k2a)
        k3c = s * f1(t + 0.5 * s, y1 + 0.5 * k2d, z1 + 0.5 * k2f, x1 + 0.5 * k2b, y2 + 0.5 * k2j, z2 + 0.5 * k2l, x2 + 0.5 * k2h)
        k3d = s * (vy1 + 0.5 * k2c)
        k3e = s * f1(t + 0.5 * s,z1 + 0.5 * k2f, x1 + 0.5 * k2b, y1 + 0.5 * k2d, z2 + 0.5 * k2l, x2 + 0.5 * k2h, y2 + 0.5 * k2j)
        k3f = s * (vz1 + 0.5 * k2e) 
        k3g = s * f1(t + 0.5 * s,  x2 + 0.5 * k2h, y2 + 0.5 * k2j, z2 + 0.5 * k2l,x1 + 0.5 * k2b, y1 + 0.5 * k2d , z1 + 0.5 * k2f) 
        k3h = s * (vx2 + 0.5 * k2g)
        k3i = s * f1(t + 0.5 * s,  y2 + 0.5 * k2j, z2 + 0.5 * k2l, x2 + 0.5 * k2h,y1 + 0.5 * k2d, z1 + 0.5 * k2f, x1 + 0.5 * k2b)
        k3j = s * (vy2 + 0.5 * k2i)
        k3k = s * f1(t + 0.5 * s, z2 + 0.5 * k2l, x2 + 0.5 * k2h,y2 + 0.5 * k2j,z1 + 0.5 * k2f,  x1 + 0.5 * k2b,y1 + 0.5 * k2d)
        k3l = s * (vz2 + 0.5 * k2k)
        

        k4a = s * f1(t + s, x1 + k3b, y1 + k3d, z1 + k3f, x2 + k3h, y2 + k3j, z2 + k3l)
        k4b = s * (vx1 + k3a)
        k4c = s * f1(t + s, y1 + k3d, z1 + k3f, x1 + k3b, y2 + k3j, z2 + k3l, x2 + k3h)
        k4d = s * (vy1 + k3c)
        k4e = s * f1(t + s, z1 + k3f, x1 + k3b, y1 + k3d, z2 + k3l, x2 + k3h, y2 + k3j)
        k4f = s * (vz1 + k3e) 
        k4g = s * f1(t + s, x2 + k3h, y2 + k3j, z2 + k3l, x1 + k3b, y1 + k3d, z1 + k3f)
        k4h = s * (vx2 + k3g)
        k4i = s * f1(t + s, y2 + k3j, z2 + k3l, x2 + k3h, y1 + k3d, z1 + k3f, x1 + k3b)
        k4j = s * (vy2 + k3i)
        k4k = s * f1(t + s, z2 + k3l,  x2 + k3h,y2 + k3j, z1 + k3f, x1 + k3b, y1 + k3d)
        k4l = s * (vz2 + k3k)
        
        np.add(vx1, (k1a + 2 * k2a + 2 * k3a + k4a) / 6.0, out=vx1)
        np.add(x1, (k1b + 2 * k2b + 2 * k3b + k4b) / 6.0, out =x1)
        np.add(vy1,  (k1c + 2 * k2c + 2 * k3c + k4c) / 6.0, out = vy1)
        np.add(y1, (k1d + 2 * k2d + 2 * k3d + k4d) / 6.0, out = y1)
        np.add(vz1, (k1e + 2 * k2e + 2 * k3e + k4e) / 6.0, out = vz1)
        np.add(z1, (k1f + 2 * k2f + 2 * k3f + k4f) / 6.0, out = z1)
        np.add(vx2, (k1g + 2 * k2g + 2 * k3g + k4g) / 6.0, out = vx2)
        np.add(x2, (k1h + 2 * k2h + 2 * k3h + k4h) / 6.0, out = x2)
        np.add(vy2, (k1i + 2 * k2i + 2 * k3i + k4i) / 6.0, out = vy2)
        np.add(y2, (k1j + 2 * k2j + 2 * k3j + k4j) / 6.0, out = y2)
        np.add(vz2, (k1k + 2 * k2k + 2 * k3k + k4k) / 6.0, out = vz2)
        np.add(z2, (k1l + 2 * k2l + 2 * k3l + k4l) / 6.0, out = z2)
        np.add(t, s, out = t)

        
    plumer = f3(t)
        

    return [vx1, x1, vy1, y1, vz1, z1, vx2, x2, vy2, y2, vz2, z2, plumer, t]


tm = np.rec.fromarrays(np.meshgrid(r0, beta)).T.flatten()
if __name__ == '__main__':
    pool = ThreadPool(1)
    ret_data = pool.map(func, tm)

rd = np.array(ret_data)

np.savetxt("mod2-r0-beta-rd.csv", rd, delimiter = ",")
tauRmassvalure = np.vstack((r0, beta))
file2 = open('mod2-r0-beta-xyvalure.csv','w')
wr = csv.writer(file2)
wr.writerows(tauRmassvalure)
file2.close()

