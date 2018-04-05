import numpy as np
import matplotlib.pyplot as plt

"""Members characteristics"""
iEdge = np.array([0, 1, 2])  #beginning of an edge
jEdge = np.array([1, 2, 0])  #end of an edge
xcoord = np.array([0, 3, 1])
ycoord = np.array([0, 0, 5])

numnode = xcoord.shape[0]  #all nodes must be used
numelem = iEdge.shape[0]  #count # of beginnings
tdof = 2*numelem  #total degrees of freedom

"Connectivity MAT computation"
ij = np.vstack([[2*iEdge, 2*iEdge+1], [2*jEdge, 2*jEdge+1]]).transpose()

"""Material characteristics"""
E = np.array([40000, 40000, 40000])  #modulus of elasticity for each member
A = np.array([0.01, .02, 0.02])  #area - each member

"""Forces and deflections"""
u = np.zeros((tdof, 1))  #deflectionsMAT
F = np.zeros((tdof, 1))  #ForcesMAT

"Basic stiffness MAT"
gsMAT = np.zeros((tdof, tdof))  #Global Stiffness MAT
xi = np.ix_(iEdge, xcoord) ##rnd tryout
print(xi)

for m in range(numelem):
    i = iEdge[m]
    j = jEdge[m]
    n = ij[m]
    length = np.sqrt(pow((xj - xi)/2, 2) + pow((yj - yi), 2))
    c = (xj - xi)/length
    s = (yj - yi)/length
    k1 = E*A/length* np.array([[c * c, c * s, -c * c, -c * s],
                               [c * s, s * s, -c * s, -s * s],
                               [-c * c, -c * s, c * c, c * s],
                               [-c * s, -s * s, c * s, s * s]])
    gsMAT[np.ix_(n, n)] += k1

actDof = np.setdiff1d(np.arange(tdof), presDof)  #Return the sorted, unique values in ar1 that are not in ar2.
u1 = np.linalg.solve(gsMAT[np.ix_(actDof, actDof)], F[np.ix_(actDof)])
u[np.ix_(actDof)] = u1


##missing xi, yi, same with j
##define some Forces F as vector 6x1
