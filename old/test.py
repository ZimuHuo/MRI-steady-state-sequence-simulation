from matplotlib import pyplot as plt
import numpy as np
from simulator.bSSFP import *

alpha = np.pi/16
size = 1000
Nr = size
# T1 = 4000
# T2 = 2200
T1 = 790 
T2 = 92
M0 = 1
sample = size
samples = np.zeros([sample,3])
TR = 10
TE = TR/2
M0 = 1
phi = 0
dphi = 0
beta = 2
data = np.zeros([Nr,3])
data = SSFP_trace(M0 =M0 , alpha = alpha, phi = phi, dphi = dphi, beta = beta, TR= TR, TE= TE, T1 = T1, T2 = T2, Nr= Nr)
fig = plt.figure(figsize=(16, 12), dpi=80)
ax = plt.axes(projection="3d")
plt.plot(data[:,0], data[:,1], data[:,2], lw=3, c='g')[0] # For line plot
plt.xlabel("x")
plt.ylabel("y")
plt.show() 