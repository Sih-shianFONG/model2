"""use the data of mod2-beta-r.py to plot the diagram """

import numpy as np

from numpy import genfromtxt

import matplotlib.pyplot as plt

import itertools

from math import pi

import matplotlib.cm as cm


rd = genfromtxt('mod2-r0-beta-rd.csv', delimiter=',')  #read the csv
tau_m = genfromtxt('mod2-r0-beta-xyvalure.csv', delimiter=',')
betastep = int(raw_input('betastep:'))
r0step =  int(raw_input('r0step:'))
G = 1.0  #gravity
m1 = float(raw_input('mass 1 :')) #mass of star 1
m2 = float(raw_input('mass 2 :'))#mass of star 2
ismdensity = float(raw_input('density of ism :')) #density of interstellar medium
mb0 = float(raw_input('mb initial mass i :')) #mass of cloud
R0 = float(raw_input('initial plummer scale i:')) #initial Plummer scale
Rf = (3 * mb0 / ( 4 * pi * ismdensity)) ** (1. / 3) #final Plummer scale
ratio = 3 * mb0 / (4 * pi * ( R0 ** 3)) / ismdensity # ratio of density(cloud/interstellar medium)
tau = float(raw_input('tau:')) #changing time
r0 = tau_m[0]
beta = tau_m[1]
vx1 = rd[:,0]
x1 = rd[:,1]
vy1 = rd[:,2]
y1  = rd[:,3]
vz1 = rd[:,4]
z1 = rd[:,5]
vx2 = rd[:,6]
x2 = rd[:,7]
vy2 = rd[:,8]
y2 = rd[:,9]
vz2 = rd[:,10]
z2 = rd[:,11]
plu = rd[:,12]
time = rd[:,13]
r12 = ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) ** 0.5
r1 = (x1 ** 2 + y1 ** 2 + z1 ** 2) ** 0.5
r2 = (x2 ** 2 + y2 ** 2 + z2 ** 2) ** 0.5
xcm = (m1 * x1 + m2 * x2) / (m1 + m2)    #center of x
ycm = (m1 * y1 + m2 * y2) / (m1 + m2)    #center of y
zcm = (m1 * z1 + m2 * z2) / (m1 + m2)    #center of z
vxcm = (m1 * vx1 + m2 * vx2) / (m1 + m2) #center velocity of x
vycm = (m1 * vy1 + m2 * vy2) / (m1 + m2) #center velocity of y
vzcm = (m1 * vz1 + m2 * vz2) / (m1 + m2) #center velocity of z
energybind = 0.5 * m1 * m2 / (m1 + m2) * ((vx2 - vx1) ** 2 + (vy2 - vy1) ** 2 + (vz2 - vz1) ** 2) - G * m1 * m2 / r12  #binding energy
energycm = 0.5 * (m1 + m2) * (vxcm ** 2 + vycm ** 2 + vzcm ** 2 )  #center energy
energy12 = energybind + energycm       #interbal energy 
rism = plu * ((Rf / plu) ** (1.2) - 1.) ** 0.5  #radius where the density of cloud is eaqual to ism
r1re = r1.reshape(r0step * betastep, 1)
r2re = r2.reshape(r0step * betastep, 1)
rismre = rism.reshape(r0step * betastep, 1)
pp = plu.reshape(r0step * betastep, 1)

eplu1 = [];eplu2=[]  #calculate the energy of cloud

for q,w,b in itertools.izip(r1re,rismre,pp):
    if q <= w:
        ep1 = - G * mb0 * m1/((b**2 + q**2)**0.5) - 2.*pi*G*m1*ismdensity*(q**2)/3. +15.*G*m1*mb0*((((b**2)/(w**2+b**2))**1.5)/3-(((b**2)/(w**2+b**2))**2.5)/5)/(2*b)
        eplu1 = np.append(eplu1,ep1)
    else:
        mc=mb0*((w**2)/(w**2+b**2))**2.5
        ep1 = -G*mc/q
        eplu1 = np.append(eplu1,ep1)
for u,o,f in itertools.izip(r2re,rismre,pp):
    if u <= o:
        ep2 = - G * mb0 * m2/((f**2 + u**2)**0.5) - 2.*pi*G*m2*ismdensity*(u**2)/3. +15.*G*m2*mb0*((((f**2)/(o**2+f**2))**1.5)/3-(((f**2)/(o**2+f**2))**2.5)/5)/(2*f)
        eplu2 = np.append(eplu2,ep2)
    else:
        mc=mb0*((o**2)/(o**2+f**2))**2.5
        ep2 = -G*mc/u
        eplu2 = np.append(eplu2,ep2)

energytotal = energy12 + eplu1 + eplu2
en=energytotal.reshape(r0step,betastep)
"plot the contour"
origin='lower' 
levels=np.linspace(-0.5,0.5,11)
co=plt.contourf(beta,r0,en,levels=levels,cmap=cm.jet,origin=origin,extend='both')
co.cmap.set_under('darkblue')
co.cmap.set_over('maroon')
co1=plt.contour(beta,r0,en,levels=[0],colors=('w',),linestyles='--',linewidths=(3,),origin=origin)
#plt.clabel(co1, fmt='%2.2f', colors='w', fontsize=10)
cbar=plt.colorbar(co)
cbar.ax.tick_params(labelsize=30)
plt.yticks(np.arange(min(r0), max(r0)+0.1, 0.1),fontsize=25)
plt.xticks(fontsize=25)
plt.title(r'$m$=%3.1f $M$=%3.1f $R$=%3.1f '% (m1,mb0,R0)+'\n'+r'$\tau$=%3.1f $\rho_{ism}$=%6.5f $ratio$=%3.0f  '%(tau,ismdensity,ratio),fontsize=30,y=1.05)
plt.xlabel(r'$\beta$',fontsize=30,labelpad=1)
plt.ylabel(r'$r_{0}$',fontsize=30,rotation = 0,labelpad=20)
plt.savefig('mod2-beta-r.eps',format='eps',  bbox_inches='tight',dpi=300)
plt.show()
