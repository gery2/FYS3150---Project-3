#3a)
from time import perf_counter
t1_start = perf_counter()

import numpy as np
import matplotlib.pyplot as plt
α = 2

def integrand(x1, y1, z1, x2, y2, z2):
    dr = abs(np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2))

    r1 = np.sqrt(x1**2 + y1**2 + z1**2)
    r2 = np.sqrt(x2**2 + y2**2 + z2**2)

    eps = 1e-8

    if dr > eps: #accounting for the problem that arises when dr = 0
        val = np.exp(-2*α*(r1 + r2))*(1/dr)
    else:
        val = 0
    return val

print('ref = ', 5*np.pi**2/16**2)

alfa = 0.9613; print('alfa = ', alfa) #integration limits [-alfa, alfa]

N = 11
print('N = ', N)

#Integrate with Gauss-legendre!
x,w = np.polynomial.legendre.leggauss(N)
x = x*alfa
sum = 0
for i in range(N):
    for j in range(N):
        for k in range(N):
            for l in range(N):
                for m in range(N):
                    for n in range(N):
                        sum += w[i]*w[j]*w[k]*w[l]*w[m]*w[n]*integrand(x[i], x[j], x[k], x[l], x[m], x[n])

print('sum = ', sum)

n = 100
x = np.linspace(-alfa, alfa, n) #making the plot smoother

array = np.zeros(n)
for i in range(n):
    array[i] = integrand(x[i], 0,0,0,0,0) #just need to plot for one variable

'''
plt.plot(x, array) #plotting x vs max value integrand
plt.title('Gauss-Legendre quadrature, N = %d' % N)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
'''

print('integrand x[0] = ', integrand(x[0], 0,0,0,0,0)) #checking if this is near enough zero.
print('integrand x[-1] = ', integrand(x[-1], 0,0,0,0,0))

t1_stop = perf_counter()
print("Elapsed time during the whole program in seconds:",
                                        t1_stop-t1_start)
