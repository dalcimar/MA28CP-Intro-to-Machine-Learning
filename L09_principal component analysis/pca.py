import numpy as np
from matplotlib import pyplot as plt 

Y = np.array([2, 1,
3, 4,
5, 0,
7, 6,
9, 2 ])

Y = Y.reshape((5,2))
mu = Y.mean(axis=0)
Yc = Y - mu

#S = np.cov(Yc.T)
S = np.dot(Yc.T,Yc)/4

val, vet = np.linalg.eig(S)

F = np.dot(Yc, vet)

plt.figure()
plt.subplot(121, aspect='equal')
plt.plot(Y[:,0], Y[:,1], 'go')
plt.arrow(mu[0], mu[1], vet[0,0], vet[1,0], color='red')
plt.arrow(mu[0], mu[1], vet[0,1], vet[1,1], color='blue')

plt.subplot(122, aspect='equal')
plt.plot(F[:,0], F[:,1], 'ro')
