"""
Created on Thu May 08 07:07:24 2014

@author: Sukhbinder Singh

Truss FEM

"""
from __future__ import division
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

nodeCords = np.array([[0.0, 0.0], [0.0, 3000.0],
    [3000.0, 0.0], [3000.0, 3000.0],
    [6000.0, 0.0], [6000.0, 3000.0]])

elemNodes = np.array([[0, 2], [1, 2], [1, 3],
                      [0, 3], [2, 3], [2, 5], [3, 4], [3, 5], [2, 4]])
print(elemNodes)

modE = 70000
Area = 300

numElem = elemNodes.shape[0]
numNodes = nodeCords.shape[0]

xx = nodeCords[:, 0]
yy = nodeCords[:, 1]

EA = modE * Area
tdof = 2 * numNodes  # total number of degrees of freedom
disps = np.zeros((tdof, 1))
force = np.zeros((tdof, 1))
sigma = np.zeros((numElem, 1))
stiffness = np.zeros((tdof, tdof))
np.set_printoptions(precision=3)


force[3] = -50000.0
force[7] = -100000.0
force[11] = -50000.0

presDof = np.array([0, 1, numNodes])

for e in range(numElem):
    indice = elemNodes[e, :]
    elemDof = np.array([indice[0] * 2, indice[0] * 2 + 1, indice[1] * 2, indice[1] * 2 + 1])
    xa = xx[indice[1]] - xx[indice[0]]
    ya = yy[indice[1]] - yy[indice[0]]
    len_elem = np.sqrt(xa * xa + ya * ya)
    c = xa / len_elem
    s = ya / len_elem
    k1 = (EA / len_elem) * np.array([[c * c, c * s, -c * c, -c * s],
                                     [c * s, s * s, -c * s, -s * s],
                                     [-c * c, -c * s, c * c, c * s],
                                     [-c * s, -s * s, c * s, s * s]])
    stiffness[np.ix_(elemDof, elemDof)] += k1

actDof = np.setdiff1d(np.arange(tdof), presDof)

print(indice)
print(elemDof)
print(xa)
print(ya)
print(len_elem)


disp1 = np.linalg.solve(stiffness[np.ix_(actDof, actDof)], force[np.ix_(actDof)]);
disps[np.ix_(actDof)] = disp1

# stresses at elements

for e in range(numElem):
    indice = elemNodes[e, :]
    elemDof = np.array([indice[0] * 2, indice[0] * 2 + 1, indice[1] * 2, indice[1] * 2 + 1])
    xa = xx[indice[1]] - xx[indice[0]]
    ya = yy[indice[1]] - yy[indice[0]]
    len_elem = np.sqrt(pow(xa, 2) + ya * ya)
    c = xa / len_elem
    s = ya / len_elem
    sigma[e] = (modE / len_elem) * np.dot(np.array([-c, -s, c, s]), disps[np.ix_(elemDof)])

print(disps)
print(sigma)

react = np.dot(stiffness, disps)
print(react.reshape((numNodes, 2)))

matplotlib