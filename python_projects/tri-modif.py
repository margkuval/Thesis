import numpy as np
import matplotlib.pyplot as plt

"""Members characteristics"""
##when changing nodes numbering, change x and ycoord as well!!
##wonder if true... try to make a square, flip nodes :)
xcoord = np.array([0., 3., 1.])
ycoord = np.array([0., 0., 5.])
iEdge = np.array([0, 1, 2])  #beginning of an edge
jEdge = np.array([1, 2, 0])  #end of an edge

"Linking xcoord with i,jEdge"
xi = xcoord[np.ix_(iEdge)]
xj = xcoord[np.ix_(jEdge)]  #take jEdge #s and replace them with corresponding xcoord
yi = ycoord[np.ix_(iEdge)]
yj = ycoord[np.ix_(jEdge)]

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
    gStif[np.ix_(n, n)] += k1

"""Forces and deflections"""
F = np.zeros((tdof, 1))  #ForcesMAT
u = np.zeros((tdof, 1))  #deflectionsMAT, 1 = # of columns

"Outside Forces [N]"
F[4] = 10
F[2] = 60

"Fixed and active DOFs"
fixedDof = np.array([0, 1, 3])  #fixed dof
actDof = np.setdiff1d(np.arange(tdof), fixedDof)  #Return sorted,unique values from tdof that are not in fixedDof

"Solve deflections"
u1 = np.linalg.solve(gStif[np.ix_(actDof, actDof)], F[np.ix_(actDof)])
u[np.ix_(actDof)] = u1
print(u)

"""Reactions"""
#Reac = gStif*u    #NOT FINISHED
F = gStif*u + F


"""Inner forces"""
k = E*A/length
uxi = u[np.ix_(2*iEdge)].transpose()
uxj = u[np.ix_(2*jEdge)].transpose()
uyi = u[np.ix_(2*iEdge + 1)].transpose()
uyj = u[np.ix_(2*jEdge + 1)].transpose()

print(uxi)

Flocal = k*((uxj - uxi)*c + (uyj - uyi)*s)
print(Flocal)

"""Stress"""
sigma = Flocal/A
print(sigma)

##graphs

xinew = xi + uxi[0]  #BUG there is a [[ in u array, if changing, need clean whole code, now solved by taking "list 1"from the MAT
xjnew = xj + uxj[0]
yinew = yi + uyi[0]
yjnew = yj + uyj[0]


"""Plot designed structure"""
"""plt.plot(xi, yi)###withoutFORfun
plt.plot(xj, yj)"""
##withForFun
for r in range(numelem):
    x = (xi[r], xj[r])
    y = (yi[r], yj[r])
    plt.plot(x, y, linestyle='-', color='black', markerfacecolor='black', marker='o', lw='0.1' )

"""Plot changed structure"""

for r in range(numelem):
    x = (xinew[r], xjnew[r])
    y = (yinew[r], yjnew[r])
    plt.plot(x, y, linestyle='-', color='black', markerfacecolor='red', marker='o', lw='0.2' )

plt.show()