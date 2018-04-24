import numpy as np
import matplotlib.pyplot as plt


def Stress(xcoord, ycoord, iEdge, jEdge, numelem, E, A, F):
    "Linking x, ycoord with i,jEdge"

    xi = xcoord[np.ix_(iEdge)]
    xj = xcoord[np.ix_(jEdge)]  # take jEdge #s and replace them with corresponding xcoord
    yi = ycoord[np.ix_(iEdge)]
    yj = ycoord[np.ix_(jEdge)]

    "Connectivity MAT computation"
    ij = np.vstack([[2 * iEdge, 2 * iEdge + 1], [2 * jEdge, 2 * jEdge + 1]]).transpose()

    "Other information"
    numnode = xcoord.shape[0]  # all nodes must be used
    tdof = 2 * numnode  # total degrees of freedom

    """"Global stiffness MAT"""
    gStif = np.zeros((tdof, tdof))  # empty Global Stiffness MAT
    length = np.sqrt(pow((xj - xi), 2) + pow((yj - yi), 2))  # members (edges) length
    c = (xj - xi) / length  # cos
    s = (yj - yi) / length  # sin

    for p in range(numelem):
        # takes p from the range of numelem 1 by 1 and creates multiple k1 (local) matrices
        # at the end maps k1 MATs on right places in gStiff MAT
        n = ij[p]
        cc = c[p] * c[p]
        cs = c[p] * s[p]
        ss = s[p] * s[p]
        k1 = E[p] * A[p] / length[p] * np.array([[cc, cs, -cc, -cs],
                                                 [cs, ss, -cs, -ss],
                                                 [-cc, -cs, cc, cs],
                                                 [-cs, -ss, cs, ss]])
        gStif[np.ix_(n, n)] += k1

    """Forces and deflections"""

    "Outside Forces [kN]"
    F_numnodex2 = F.reshape(numnode, 2)

    "Fixed and active DOFs"

    "Fixed nodes"
    fixedDof = np.array([0, 1, 7])

    actDof = np.setdiff1d(np.arange(tdof), fixedDof)  # Return sorted,unique values from tdof that are not in fixedDof


    "Solve deflections"
    u = np.zeros((tdof, 1))  # empty deflections MAT; 1 = # of columns
    u1 = np.linalg.solve(gStif[np.ix_(actDof, actDof)], F[np.ix_(actDof)])  # solve equation gStiff*u = F
    u[np.ix_(actDof)] = u1  # map back to the empty def MAT

    """Inner forces"""
    k = E * A / length
    uxi = u[np.ix_(2 * iEdge)].transpose()
    uxj = u[np.ix_(2 * jEdge)].transpose()
    uyi = u[np.ix_(2 * iEdge + 1)].transpose()
    uyj = u[np.ix_(2 * jEdge + 1)].transpose()

    Flocal = k * ((uxj - uxi) * c + (uyj - uyi) * s)  # c=cos,s=sin

    """Stress (sigma)=(kPa)"""
    stress = Flocal[0] / A
    stress_normed = [i / sum(abs(stress)) for i in abs(stress)]

    xinew = xi + uxi[0]  # notCLEARed-[[ in u array, now solved by taking "list 0" from the MAT
    xjnew = xj + uxj[0]
    yinew = yi + uyi[0]
    yjnew = yj + uyj[0]

    """Plot structure"""

    for r in range(numelem):
        x = (xi[r], xj[r])
        y = (yi[r], yj[r])
        line = plt.plot(x, y)
        plt.setp(line, ls='-', c='black', lw='1', label='orig')

        xnew = (xinew[r], xjnew[r])
        ynew = (yinew[r], yjnew[r])
        linenew = plt.plot(xnew, ynew)
        plt.setp(linenew, ls='-', c='c' if stress[r] > 0 else 'crimson', lw=1 + 20 * stress_normed[r],
                 label='strain' if stress[r] > 0 else 'stress')

    for r in range(numnode):
        plt.annotate(F_numnodex2[r],
                     xy=(xi[r], yi[r]), xycoords='data',
                     xytext=(np.sign(F_numnodex2[r]) * -50), textcoords='offset pixels',
                     arrowprops=dict(facecolor='black', shrink=0, width=1.5, headwidth=8),
                     horizontalalignment='right', verticalalignment='bottom')
        # print("N"+str(i+1)+" = "+ str(np.round(N[i] /1000,3)) +" kN")


    stress_max = np.round(np.max(stress), 3)  # 3 decimal nums
    return stress_max