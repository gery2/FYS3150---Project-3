#3d
from time import perf_counter
t1_start = perf_counter()

import numpy as np
import matplotlib.pyplot as plt

α = 2
b = 4
d = np.pi
f = np.pi*2

def func(r1, r2, theta1, theta2, phi1, phi2):

    cosbeta = np.cos(theta1)*np.cos(theta2) + np.sin(theta1)*np.sin(theta2)*np.cos(phi1 - phi2)
    dr = np.sqrt(r1**2 + r2**2 - 2*r1*r2*cosbeta)

    eps = 1e-4

    if dr > eps: #accounting for the problem that arises when dr = 0
        val = (1/dr)*r1**2*r2**2*np.sin(theta1)*np.sin(theta2)
    else:
        val = 0
    return val

#improved Monte Carlo integration:
def integrate(b, d, f):

    sum = 0
    func_array = np.zeros(N)
    jacobi = np.pi**4/4

    for i in range(N):
        X1 = np.random.exponential(1/4)
        X2 = np.random.exponential(1/4)
        Z1 = np.random.uniform(0,f)
        Y1 = np.random.uniform(0,d)
        Y2 = np.random.uniform(0,d)
        Z2 = np.random.uniform(0,f)
        func_array[i] = func(X1, X2, Y1, Y2, Z1, Z2)
        sum += func(X1, X2, Y1, Y2, Z1, Z2)

    mc = sum/N
    var = 0

    for i in range(N):
        var += (mc - func_array[i])**2

    var = var*jacobi/N
    std = np.sqrt(var)/np.sqrt(N) #variance should decrease with higher N.

    result = jacobi*sum/N

    return result, std


n = [2*10**6] #for plotting: [10**2,10**3,10**4,10**5,10**6]
result = np.zeros(len(n)); variance = np.zeros(len(n))

for i in range(len(n)):
    N = n[i]
    result[i], variance[i] = integrate(b, d, f)

print('N = ', n[-1]) #comment out when plotting
print('result = ', result)
print('variance = ', variance)
'''
#plot n,variance
import matplotlib.pyplot as plt
plt.loglog(n,variance)
plt.xlabel('log(N)')
plt.ylabel('log(Variance)')
plt.title('Improved: How the variance decreases with higher N')
plt.show()
'''



t1_stop = perf_counter()
print("Elapsed time during the whole program in seconds:",
                                        t1_stop-t1_start)

#
