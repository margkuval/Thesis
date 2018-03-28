# Square, 3DoF

from __future__ import division
import numpy as np

modE = 70000
area = 2.0

#numElem = 4
#numNodes = 4

nodes = np.array([[0,0], [1,0], [1,1], [0,1]])
elem = np.array([[0,1], [1,2], [2,3], [3,0], [0,2]])  #square

numNodes = nodes.shape[0]  #rows in max,0 eq to rows
numElem = elem.shape[0]

xx = nodes[:, 0]  #x nodes coord
yy = nodes[:, 1]  #y nodes coord

EA = modE*area
tdof = 2*numNodes  #total degrees of freedom DoF
u = np.zeros((tdof,1))  #matrix tdof by 1
F = np.zeros((tdof,1))
stiffness = np.zeros((tdof, tdof))  #matrix tdof by tdof
sigma = np.zeros((numElem, 1))
np.set_printoptions(precision = 2)

F[1] = -500.0
F[2] = -1000.0

presDof = np.array([0, 1, numNodes])
print(presDof)

for e in range(numElem):
    indice = nodes[e, :]
    elemDof = np.array([indice[0] * 2, indice[0] * 2+1, indice[1] * 2, indice[1] * 2+1])
    xa = xx[indice[1]] - xx[indice[0]]
    ya = yy[indice[1]] - yy[indice[0]]
    elemL = 1
    c = xa/elemL
    s = ya / elemL
    k1 = (EA / elemL) * np.array([[c * c, c * s, -c * c, -c * s],
                                  [c * s, s * s, -c * s, -s * s],
                                  [-c * c, -c * s, c * c, c * s],
                                  [-c * s, -s * s, c * s, s * s]])
    stiffness[np.ix_(elemDof, elemDof)] += k1

print(indice)

actDof = np.setdiff1d(np.arange(tdof), presDof)

u1 = np.linalg.solve(stiffness[np.ix_(actDof, actDof)], F[np.ix_(actDof)])
u[np.ix_(actDof)] = u1

# stresses at elements

for e in range(numElem):
    indice = nodes[e, :]
    elemDof = np.array([indice[0] * 2, indice[0] * 2 + 1, indice[1] * 2, indice[1] * 2 + 1])
    xa = xx[indice[1]] - xx[indice[0]]
    ya = yy[indice[1]] - yy[indice[0]]
    elemL = np.sqrt(xa * xa + ya * ya)
    c = xa / elemL
    s = ya / elemL
    sigma[e] = (modE / elemL) * np.dot(np.array([-c, -s, c, s]), u[np.ix_(elemDof)])

print(xa)

print(u)
print(sigma)

react = np.dot(stiffness, u)
print(react.reshape((numNodes, 2)))

