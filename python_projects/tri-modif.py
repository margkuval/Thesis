import numpy as np
import matplotlib.pyplot as plt

"""Members characteristics"""
##when changing nodes numbering, change x and ycoord as well!!
##wonder if true... try to make a square, flip nodes :)
xcoord = np.array([0, 3, 1])
ycoord = np.array([0, 0, 5])
iEdge = np.array([0, 1, 2])  #beginning of an edge
jEdge = np.array([1, 2, 0])  #end of an edge

"Linking xcoord with i,jEdge"
xi = xcoord[np.ix_(iEdge)]
xj = xcoord[np.ix_(jEdge)]  #take jEdge #s and replace them with corresponding xcoord
yi = ycoord[np.ix_(iEdge)]
yj = ycoord[np.ix_(jEdge)]

print(xi)
print(xj)

numnode = xcoord.shape[0]  #all nodes must be used
numelem = iEdge.shape[0]  #count # of beginnings
tdof = 2*numelem  #total degrees of freedom

"Connectivity MAT computation"
ij = np.vstack([[2*iEdge, 2*iEdge+1], [2*jEdge, 2*jEdge+1]]).transpose()
print(ij)

"""Material characteristics"""
E = np.array([40000, 40000, 40000])  #modulus of elasticity for each member
A = np.array([0.01, .02, 0.02])  #area - each member

""""Global stiffness MAT"""
gStif = np.zeros((tdof, tdof))  #empty Global Stiffness MAT
print(gStif)

for p in range(numelem):
    i = iEdge[p]
    j = jEdge[p]
    n = ij[p]
    length = np.sqrt(pow((xj[p] - xi[p]), 2) + pow((yj[p] - yi[p]), 2))
    c = (xj[p] - xi[p])/length
    s = (yj[p] - yi[p])/length
    cc = c * c
    cs = c * s
    ss = s * s
    k1 = E[p]*A[p]/length * np.array([[cc, cs, -cc, -cs],
                                      [cs, ss, -cs, -ss],
                                      [-cc, -cs, cc, cs],
                                      [-cs, -ss, cs, ss]])
    print(k1)
    gStif[np.ix_(n, n)] += k1

print(gStif)


"""Forces and deflections"""
u = np.zeros((tdof, 1))  #deflectionsMAT, 1 = # of columns
F = np.zeros((tdof, 1))  #ForcesMAT

"Forces [N]"
F[4] = 100
F[2] = 60

"Fixed and active DOFs"
fixedDof = np.array([0, 1, 3])  #fixed dof
actDof = np.setdiff1d(np.arange(tdof), fixedDof) #Return sorted,unique values from tdof that are not in fixedDof

"""Solve deflections"""
u1 = np.linalg.solve(gStif[np.ix_(actDof, actDof)], F[np.ix_(actDof)])
u[np.ix_(actDof)] = u1
print(u)
