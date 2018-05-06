import numpy as np
import matplotlib.pyplot as plt

"""mems characteristics x,ycoord=(m)"""

xcoord = np.array([0., 2., 3.5, 5.5, 2., 3.5])
ycoord = np.array([0., 0., 0., 0, 2.5, 2.5])
mem_begin = np.array([0, 1, 2, 0, 4, 5, 1, 2, 2])  #beginning of an edge
mem_end = np.array([1, 2, 3, 4, 5, 3, 4, 4, 5])  #end of an edge

"Linking x, ycoord with i,mem_end"
xi = xcoord[np.ix_(mem_begin)]
xj = xcoord[np.ix_(mem_end)]  #take mem_end #s and replace them with corresponding xcoord
yi = ycoord[np.ix_(mem_begin)]
yj = ycoord[np.ix_(mem_end)]

numnode = xcoord.shape[0]  #all nodes must be used
numelem = mem_begin.shape[0]  #count # of beginnings
dof_tot = 2*numnode  #total degrees of freedom

"Connectivity MAT computation"
ij = np.vstack([[2*mem_begin, 2*mem_begin+1], [2*mem_end, 2*mem_end+1]]).transpose()
print(ij)

"""Material characteristics E=(kPa), A=(m2)"""
E = np.array(numelem*[40000])  #modulus of elasticity for each mem
A = np.array(numelem*[0.0225])  #area - each mem 0.15x0.15

""""Global stiffness MAT"""
glob_stif = np.zeros((dof_tot, dof_tot))  #empty Global Stiffness MAT
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
    glob_stif[np.ix_(n, n)] += k1

print(glob_stif)
"""Forces and deflections"""
F = np.zeros((dof_tot, 1))  #ForcesMAT
u = np.zeros((dof_tot, 1))  #deflectionsMAT, 1 = # of columns

"Outside Forces [kN]"
F[3] = -5
F[8] = 10
F[9] = -15
F[11] = -15
F_numnodex2 = F.reshape(numnode, 2)

"Fixed and active DOFs"
dof_fixed = np.array([0, 1, 7])  #fixed dof
acdof_tot = np.setdiff1d(np.arange(dof_tot), dof_fixed)  #Return sorted,unique values from dof_tot that are not in dof_fixed

print(acdof_tot)

"Solve deflections"
u1 = np.linalg.solve(glob_stif[np.ix_(acdof_tot, acdof_tot)], F[np.ix_(acdof_tot)])
u[np.ix_(acdof_tot)] = u1
print(u)

"""Inner forces"""
k = E*A/length
uxi = u[np.ix_(2*mem_begin)].transpose()
uxj = u[np.ix_(2*mem_end)].transpose()
uyi = u[np.ix_(2*mem_begin + 1)].transpose()
uyj = u[np.ix_(2*mem_end + 1)].transpose()

Flocal = k*((uxj - uxi)*c + (uyj - uyi)*s)
print(Flocal)

"""Stress (sigma)=(kPa)"""
stress = Flocal[0]/A
stress_normed = [i/sum(abs(stress)) for i in abs(stress)]
print(stress)

xinew = xi + uxi[0]  #BUG-there is an [[ in u array, if changing, need clean whole code, now solved by taking "list 0"from the MAT
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
plt.title('Happy truss')
plt.grid(True)
plt.legend(bbox_to_anchor=[1.005,1],loc=2,borderaxespad=0)

plt.show()