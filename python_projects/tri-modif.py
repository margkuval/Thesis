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
length = np.sqrt(pow((xj - xi), 2) + pow((yj - yi), 2))
c = (xj - xi)/length
s = (yj - yi)/length

for p in range(numelem):
    i = iEdge[p]
    j = jEdge[p]
    n = ij[p]
    cc = c[p] * c[p]
    cs = c[p] * s[p]
    ss = s[p] * s[p]
    k1 = E[p]*A[p]/length[p] * np.array([[cc, cs, -cc, -cs],
                                      [cs, ss, -cs, -ss],
                                      [-cc, -cs, cc, cs],
                                      [-cs, -ss, cs, ss]])
    print(k1)
    gStif[np.ix_(n, n)] += k1
print(gStif)

"""Forces and deflections"""
F = np.zeros((tdof, 1))  #ForcesMAT
u = np.zeros((tdof, 1))  #deflectionsMAT, 1 = # of columns

"Forces [N]"
F[4] = 100
F[2] = 60

"Fixed and active DOFs"
fixedDof = np.array([0, 1, 3])  #fixed dof
actDof = np.setdiff1d(np.arange(tdof), fixedDof)  #Return sorted,unique values from tdof that are not in fixedDof

"Solve deflections"
u1 = np.linalg.solve(gStif[np.ix_(actDof, actDof)], F[np.ix_(actDof)])  #zaznamenaji se tam i naklony prutu?
u[np.ix_(actDof)] = u1
print(u)

"""Reactions"""
#Reac = gStif*u
Reac = gStif*u

"""Inner forces"""
k = E*A/length
uxi = u[np.ix_(2*iEdge)]
uxj = u[np.ix_(2*jEdge)]
uyi = u[np.ix_(2*iEdge + 1)]
uyj = u[np.ix_(2*jEdge + 1)]

for z in range(numelem):
    Flocal = k[z]*((uxj[z] - uxi[z])*c[z] + (uyj[z] - uyi[z])*s[z])
    print(Flocal)

Flocal = k*((uxj - uxi)*c + (uyj - uyi)*s)
print(Flocal)
#calc forces in local system for each member
#graphs

sigma = Flocal/A
print(sigma)