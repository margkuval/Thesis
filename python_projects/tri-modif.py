import numpy as np
import matplotlib.pyplot as plt

"""Members characteristics"""
##when changing nodes numbering, change x and ycoord as well!!
iEdge = np.array([0, 1, 2])  #beginning of an edge
jEdge = np.array([1, 2, 0])  #end of an edge
xcoord = np.array([0, 3, 1])
ycoord = np.array([0, 0, 5])

xi = iEdge[1, 2, 3]
xj = np.stack([])

numnode = xcoord.shape[0]  #all nodes must be used
numelem = iEdge.shape[0]  #count # of beginnings
tdof = 2*numelem  #total degrees of freedom

"Connectivity MAT computation"
ij = np.vstack([[2*iEdge, 2*iEdge+1], [2*jEdge, 2*jEdge+1]]).transpose()
print(ij)

"""Material characteristics"""
E = np.array([40000, 40000, 40000])  #modulus of elasticity for each member
A = np.array([0.01, .02, 0.02])  #area - each member

"""Forces and deflections"""
u = np.zeros((tdof, 1))  #deflectionsMAT
F = np.zeros((tdof, 1))  #ForcesMAT
"Fix"
u[1] = 0  #deflections in particular nodes equal to zero
u[2] = 0
u[4] = 0
"Forces [N]"
F[0] = 50
F[3] = 60

"Basic stiffness MAT"
gsMAT = np.zeros((tdof, tdof))  #Global Stiffness MAT

for m in range(numelem):
    i = iEdge[m]
    j = jEdge[m]
    n = ij[m]
    xj = np.array([3, 1, 0]).transpose()
    xi = np.array([0, 3, 1]).transpose()
    yj = np.array([0, 5, 0]).transpose()
    yi = np.array([0, 0, 5]).transpose()
    length = np.array(np.sqrt(pow((xj - xi), 2) + pow((yj - yi), 2)))
    c = (xj - xi)/length
    s = (yj - yi)/length
    k1 = E*A/length* np.array([[c * c, c * s, -c * c, -c * s],
                               [c * s, s * s, -c * s, -s * s],
                               [-c * c, -c * s, c * c, c * s],
                               [-c * s, -s * s, c * s, s * s]])
    gsMAT[np.ix_(n, n)] += k1

actDof = np.setdiff1d(np.arange(tdof), u)  #Return the sorted, unique values in ar1 that are not in ar2.
u1 = np.linalg.solve(gsMAT[np.ix_(actDof, actDof)], F[np.ix_(actDof)])
u[np.ix_(actDof)] = u1

print(u1)

#missing xi, yi, same with j

