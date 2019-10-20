#3a)
from time import perf_counter
t1_start = perf_counter()

import numpy as np
import matplotlib.pyplot as plt
α = 2

def integrand(r1, r2, theta1, theta2, phi1, phi2):

    cosbeta = np.cos(theta1)*np.cos(theta2) + np.sin(theta1)*np.sin(theta2)*np.cos(phi1 - phi2)
    dr = np.sqrt(r1**2 + r2**2 - 2*r1*r2*cosbeta)

    eps = 1e-4

    if dr > eps: #accounting for the problem that arises when dr = 0
        val = np.exp(-(3/2)*α*(r1 + r2))*(1/dr)*r1**2*r2**2*np.sin(theta1)*np.sin(theta2) #-(3/2 because numpys function wants exp(-x))
    else:
        val = 0
    return val

print('ref = ', 5*np.pi**2/16**2)

alfa = 1; print('alfa = ', alfa)

N = 11
print('N = ', N)

#Integrate (r1, r2) with Gauss-Laguerre:
xr,wr = np.polynomial.laguerre.laggauss(N) #Laguerre
xt,wt = np.polynomial.legendre.leggauss(N) #legendre here because of different interval
xp,wp = np.polynomial.legendre.leggauss(N) #same as above
xt0 = xt + alfa #adjusting to [0,2]
xt1 = xt0*np.pi/(2*alfa) #[0,pi]
xp0 = xp + alfa
xp1 = xp0*np.pi/(alfa) #[0,2pi]
sum = 0
wt = wt*(np.pi/2) #weights also need adjusting
wp = wp*(np.pi)
for i in range(N):
    for j in range(N):
        for k in range(N):
            for l in range(N):
                for m in range(N):
                    for n in range(N):
                        sum += wr[i]*wr[j]*wt[k]*wt[l]*wp[m]*wp[n]*integrand(xr[i], xr[j], xt1[k], xt1[l], xp1[m], xp1[n])
print('sum = ', sum)



t1_stop = perf_counter()
print("Elapsed time during the whole program in seconds:",
                                        t1_stop-t1_start)
