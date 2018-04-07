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

"Basic stiffness MAT"
gsMAT = np.zeros((tdof, tdof))  #empty Global Stiffness MAT
print(gsMAT)
for mat in range(numelem):
    i = iEdge[mat]
    j = jEdge[mat]
    n = ij[mat].tran
    length = np.array(np.sqrt(pow((xj - xi), 2) + pow((yj - yi), 2)))
    c = (xj - xi)/length
    s = (yj - yi)/length
    cc = c * c
    cs = c * s
    ss = s * s
    k1 = (E*A/length) * np.array([[cc, cs, -cc, -cs],
                                  [cs, ss, -cs, -ss],
                                  [-cc, -cs, cc, cs],
                                  [-cs, -ss, cs, ss]])
    print(k1)
    gsMAT[np.ix_(n, n)] += k1

print(gsMAT)
  #Return the sorted, unique values in ar1 that are not in ar2.



"""Forces and deflections"""
u = np.zeros((tdof, 1))  #deflectionsMAT, 1 = # of columns
F = np.zeros((tdof, 1))  #ForcesMAT
"Fix"
u[0] = 0  #fixed nodes, deflections = 0
u[1] = 0
u[3] = 0
print(u)
"Forces [N]"
F[0] = 50
F[3] = 60

fixDof = np.array([0, 1, 3])  #fixed dofs

actDof = np.setdiff1d(np.arange(tdof), fixDof)  #actual Dofs w/o fixed ones
#setdiff1d #Return the sorted, unique values in ar1 that are not in ar2.
print(actDof)
u1 = np.linalg.solve(stiffness[np.ix_(actDof, actDof)], F[np.ix_(actDof)])  ##neco v tom nesedi, nevim jak s tim
u[np.ix_(actDof)] = u1

##jak je mozne, ze mam 3 sloupce? k1 by mela mit 4, gsMAT by se mela rovnat ixj