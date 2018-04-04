from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

""""Triangle FEM"""

""""Define nodes coordinates and connections between them"""
nodes = np.array([[0, 0], [2, 0], [1, 2]])
edges = np.array([[0, 1], [1, 2], [2, 0]])

numElem = edges.shape[0]
numNodes = nodes.shape[0]

"""show all x or y coordinates in a MAT"""
xx = nodes[:, 0]
yy = nodes[:, 1]

"""material characteristics"""
modE = 40000  #MPa
area = 0.04  #0.2m x 0.2m, odhad, pri optimalizaci stanovit, ze muze menit
EA = modE*area

"""structure characteritics"""
tdof = 2*numNodes  #total # of degrees of freedom

u = np.zeros((tdof, 1))  #deflectionsMAT
F = np.zeros((tdof, 1))  #ForcesMAT
sigma = np.zeros((numElem, 1))  #mat for streses
stiffness = np.zeros((tdof, tdof))  #shape of big MAT

"""define outside forces"""
F[1] = 300 #N
F[2] = -1000  #N

presDof=np.array([0, 1])  # CO TO DELA?

for e in range(numElem):  #numElem = 3
    indice = edges[e, :]
    elemDof = np.array([indice[0]*2, indice[0]*2+1, indice[1]*2, indice[1]*2+1])
    xa = xx[indice[1]] - xx[indice[0]]  #distance calc of x coords
    ya = yy[indice[1]] - yy[indice[0]]  #same with y
    elemLen = np.sqrt(pow(xa/2, 2) + pow(ya, 2))  #length of the tilted element
                # jak prepsat tak, aby se mohl menit pri pouziti GA
    c = xa/elemLen
    s = ya/elemLen
    k1 = (EA/elemLen)* np.array([[c * c, c * s, -c * c, -c * s],
                                 [c * s, s * s, -c * s, -s * s],
                                 [-c * c, -c * s, c * c, c * s],
                                 [-c * s, -s * s, c * s, s * s]])
    stiffness[np.ix_(elemDof, elemDof)] += k1
  #jak se prepise aby dokazalo reflektovat zmeny GA?

actDof = np.setdiff1d(np.arange(tdof), presDof)  #Return the sorted, unique values in ar1 that are not in ar2.
u1 = np.linalg.solve(stiffness[np.ix_(actDof, actDof)], F[np.ix_(actDof)])
u[np.ix_(actDof)] = u1

print(u)

#jak vytisknout puvodni a novy zaroven?
#jakym smerem pusobi sily
#jsou vysledky v metrech


"""calc location of new nodes"""
newnodes = np.array(nodes.flatten()+u.flatten()).reshape(3, 2)  #original nodes + deformation, reshaping into 3x2 MAT
xxnew = newnodes[:, 0]
yynew = newnodes[:, 1]

print(newnodes)

#print both tri and tri+u
plt.plot(xx[edges], yy[edges], linestyle='-', color='black',
        markerfacecolor='black', marker='o')

plt.plot(xxnew[edges.T], yynew[edges.T], linestyle='--', color='r',
        markerfacecolor='red', marker='o')

r = [3, 6, 2]

rolled_xx = np.roll(xx, -1)
print(rolled_xx - xx)

"""Calculate l and dl (Pythagoras), rolled function moves x --> x+1 in MATs"""
l = np.sqrt((pow(np.roll(xx, -1)-(xx), 2)+ pow(np.roll(yy, -1)-(yy), 2)))  #transpose at some point...
dl = np.sqrt((pow(np.roll(xxnew, -1)-(xxnew), 2)+ pow(np.roll(yynew, -1)-(yynew), 2)))
print(l)
print(dl)




