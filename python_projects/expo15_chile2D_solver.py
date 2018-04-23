import numpy as np
import matplotlib.pyplot as plt
import random as rnd

"""Members characteristics x,ycoord=(m)"""

#structure is made from triangles with same side value a = 2.5m
#to define precise coordinates, hight will be used as h
a = 2.5
h = np.sqrt(pow((a), 2) - pow((a/2), 2))

xcoord = np.array([0, a/2, 0., a, 2*a, a+a/2, 2*a, a])
ycoord = np.array([2*h, h, 0., 0., 0., h, 2*h, 2*h])
iEdge = np.array([0, 1, 2, 3, 4, 5, 6, 7, 1, 7, 5, 1, 5])  #beginning of an edge
jEdge = np.array([1, 2, 3, 4, 5, 6, 7, 0, 7, 5, 1, 3, 3])  #end of an edge
print(xcoord)

"Linking x, ycoord with i,jEdge"
xi = xcoord[np.ix_(iEdge)]
xj = xcoord[np.ix_(jEdge)]  #take jEdge #s and replace them with corresponding xcoord
yi = ycoord[np.ix_(iEdge)]
yj = ycoord[np.ix_(jEdge)]

x1GA = np.zeros((len(np.unique(iEdge)), 1))

print(x1GA)

numnode = xcoord.shape[0]  #all nodes must be used
numelem = iEdge.shape[0]  #count # of beginnings
tdof = 2*numnode  #total degrees of freedom

"Connectivity MAT computation"
ij = np.vstack([[2*iEdge, 2*iEdge+1], [2*jEdge, 2*jEdge+1]]).transpose()
print(ij)

"""Material characteristics E=(kPa), A=(m2)"""
E = np.array(numelem*[40000])  #modulus of elasticity for each member
A = np.array(numelem*[0.0225])  #area - each member 0.15x0.15m

""""Global stiffness MAT"""
gStif = np.zeros((tdof, tdof))  #empty Global Stiffness MAT
length = np.sqrt(pow((xj - xi), 2) + pow((yj - yi), 2))
c = (xj - xi)/length
s = (yj - yi)/length

for p in range(numelem):
    n = ij[p]
    cc = c[p] * c[p]
    cs = c[p] * s[p]
    ss = s[p] * s[p]
    k1 = E[p]*A[p]/length[p] * np.array([[cc, cs, -cc, -cs],
                                         [cs, ss, -cs, -ss],
                                         [-cc, -cs, cc, cs],
                                         [-cs, -ss, cs, ss]])
    gStif[np.ix_(n, n)] += k1

print(gStif)
"""Forces and deflections"""
F = np.zeros((tdof, 1))  #ForcesMAT
u = np.zeros((tdof, 1))  #deflectionsMAT, 1 = # of columns

"Outside Forces [kN]"
F[2] = 15
F[13] = -10
F_numnodex2 = F.reshape(numnode, 2)

"Fixed and active DOFs"
fixedDof = np.array([0, 1, 7])  #fixed dof
actDof = np.setdiff1d(np.arange(tdof), fixedDof)  #Return sorted,unique values from tdof that are not in fixedDof

print(actDof)

"Solve deflections"
u1 = np.linalg.solve(gStif[np.ix_(actDof, actDof)], F[np.ix_(actDof)])
u[np.ix_(actDof)] = u1
print(u)

"""Inner forces"""
k = E*A/length
uxi = u[np.ix_(2*iEdge)].transpose()
uxj = u[np.ix_(2*jEdge)].transpose()
uyi = u[np.ix_(2*iEdge + 1)].transpose()
uyj = u[np.ix_(2*jEdge + 1)].transpose()

Flocal = k*((uxj - uxi)*c + (uyj - uyi)*s)  #c=cos,s=sin 
print(Flocal)

"""Stress (sigma)=(kPa)"""
stress = Flocal[0]/A
stress_normed = [i/sum(abs(stress)) for i in abs(stress)]
print(stress)

xinew = xi + uxi[0]  #BUG-there is an [[ in u array, if changing, need clean whole code, now solved by taking "list 0" from the MAT
xjnew = xj + uxj[0]
yinew = yi + uyi[0]
yjnew = yj + uyj[0]

"""Plot structure"""

"""plt.plot(xi, yi)###withoutFORfun
plt.plot(xj, yj)"""

for r in range(numelem):
    x = (xi[r], xj[r])
    y = (yi[r], yj[r])
    line = plt.plot(x,y)
    plt.setp(line, ls='-', c='black', lw='1', label='orig')

    xnew = (xinew[r], xjnew[r])
    ynew = (yinew[r], yjnew[r])
    linenew = plt.plot(xnew, ynew)
    plt.setp(linenew, ls='-', c='c' if stress[r] > 0 else 'crimson', lw=1+20*stress_normed[r], label='strain' if stress[r] > 0 else 'stress')

for r in range(numnode):
    plt.annotate(F_numnodex2[r],
                 xy=(xi[r], yi[r]), xycoords='data',
                 xytext=(np.sign(F_numnodex2[r])*-50), textcoords='offset pixels',
                 arrowprops=dict(facecolor='black', shrink=0, width=1.5, headwidth=8),
                 horizontalalignment='right', verticalalignment='bottom')
    # print("N"+str(i+1)+" = "+ str(np.round(N[i] /1000,3)) +" kN")

plt.axis('equal')
plt.xlabel('meters')
plt.ylabel('meters')
plt.title('Happy tri truss')
plt.grid(True)
plt.legend(bbox_to_anchor=[1.005,1],loc=2,borderaxespad=0)

plt.show()