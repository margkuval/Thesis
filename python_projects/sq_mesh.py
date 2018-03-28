"""
Example 4
SIAM Workshop October 18, 2014, M. M. Sussman
example4.py
Plotting shape functions
"""
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# shape functions
phi = [lambda xi: 2.0 * (xi - 0.5) * (xi - 1.0),
       lambda xi: 4.0 * xi * (1.0 - xi),
       lambda xi: 2.0 * xi * (xi - 0.5)]

L = 5
N = 5
dx = float(L) / float(N)
Ndof = 2 * N + 1
Nplot = 100
x = np.linspace(0, L, Ndof)
xiplot = np.linspace(0, 1, Nplot)


def plotphi(k, n, hold):
    """
    plot one of the shape functions
    use "hold on" if hold is true, not if not
    """
    assert (k >= 0 and k <= N)
    xplot = (k + xiplot) * dx
    plt.hold(hold)
    plt.plot(xplot, phi[n](xiplot), label="elt%d num%d" % (k, n))


plotphi(2, 0, False)
plotphi(1, 2, True)
plotphi(4, 1, True)
plt.legend(loc='upper center')

plt.show()