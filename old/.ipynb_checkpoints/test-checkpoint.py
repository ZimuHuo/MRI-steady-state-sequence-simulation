from matplotlib import pyplot as plt
import numpy as np
from simulator.bSSFP import *


alpha = np.pi/2
size = 100
Nr = size
# T1 = 4000
# T2 = 2200
T1 = 790 
T2 = 92
M0 = 1
sample = size
samples = np.zeros([sample,1], dtype = complex)
TR = 10
TE = TR/2
M0 = 1
phi = 0
dphi = 0
betaGrad = np.linspace(-np.pi, np.pi, Nr)
for index, beta in enumerate(betaGrad):
    samples[index] = SSFP(M0 =M0 , alpha = alpha, phi = phi, dphi = dphi, beta = beta, TR= TR, TE= TE, T1 = T1, T2 = T2, Nr= Nr)

plt.figure()
plt.subplot(211)
x = np.linspace(0, np.pi, Nr)
plt.plot(x/(2*np.pi*TR),np.absolute(samples))
plt.subplot(212)
x = np.linspace(0, np.pi, Nr)
plt.plot(x/(2*np.pi*TR),np.angle(samples))
plt.show()