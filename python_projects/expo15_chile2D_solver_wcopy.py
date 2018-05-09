import numpy as np
import matplotlib.pyplot as plt


def Stress(xcoord, ycoord, mem_begin, mem_end, numelem, E, A, F, dof_fixed):
    "Linking x, ycoord with i,mem_end"

    xi = xcoord[np.ix_(mem_begin)]
    xj = xcoord[np.ix_(mem_end)]  # take mem_end #s and replace them with corresponding xcoord
    yi = ycoord[np.ix_(mem_begin)]
    yj = ycoord[np.ix_(mem_end)]

    "Connectivity MAT computation"
    ij = np.vstack([[2 * mem_begin, 2 * mem_begin + 1], [2 * mem_end, 2 * mem_end + 1]]).transpose()

    "Other information"
    numnode = xcoord.shape[0]  # all nodes must be used
    dof_tot = 2 * numnode  # total degrees of freedom


    """"Global stiffness MAT"""
    glob_stif = np.zeros((dof_tot, dof_tot))  # empty Global Stiffness MAT
    length = np.sqrt(pow((xj - xi), 2) + pow((yj - yi), 2))  # mems (edges) length
    c = (xj - xi) / length  # cos
    s = (yj - yi) / length  # sin

    for p in range(numelem):
        # takes p from the range of numelem 1 by 1 and creates multiple k1 (local) matrices
        # at the end maps k1 MATs on right places in glob_stiff MAT
        n = ij[p]
        cc = c[p] * c[p]
        cs = c[p] * s[p]
        ss = s[p] * s[p]
        k1 = E[p] * A[p] / length[p] * np.array([[cc, cs, -cc, -cs],
                                                 [cs, ss, -cs, -ss],
                                                 [-cc, -cs, cc, cs],
                                                 [-cs, -ss, cs, ss]])
        glob_stif[np.ix_(n, n)] += k1

    """Forces and deflections"""

    "Outside Forces [kN]"
    F_numnodex2 = F.reshape(numnode, 2)

    "Fixed and active DOFs"
    acdof_tot = np.setdiff1d(np.arange(dof_tot), dof_fixed)  # Return sorted,unique values from dof_tot that are not in dof_fixed


    "Solve deflections"
    u = np.zeros((dof_tot, 1))  # empty deflections MAT; 1 = # of columns
    u1 = np.linalg.solve(glob_stif[np.ix_(acdof_tot, acdof_tot)], F[np.ix_(acdof_tot)])  # solve equation glob_stiff*u = F
    u[np.ix_(acdof_tot)] = u1  # map back to the empty def MAT

    """Inner forces"""
    k = E * A / length
    uxi = u[np.ix_(2 * mem_begin)].transpose()
    uxj = u[np.ix_(2 * mem_end)].transpose()
    uyi = u[np.ix_(2 * mem_begin + 1)].transpose()
    uyj = u[np.ix_(2 * mem_end + 1)].transpose()

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
        plt.setp(linenew,
                 ls='-',
                 c='c' if stress[r] > 0.5 else ('r' if stress[r] < -0.5 else 'k'),
                 lw=1 + 20 * stress_normed[r],
                 label='strain' if stress[r] > 0.5 else 'stress')

    for r in range(numnode):
        plt.annotate(F_numnodex2[r],
                     xy=(xi[r], yi[r]), xycoords='data',
                     xytext=(np.sign(F_numnodex2[r]) * -50), textcoords='offset pixels',
                     arrowprops=dict(facecolor='black', shrink=0, width=1.5, headwidth=8),
                     horizontalalignment='right', verticalalignment='bottom')
    plt.axis('equal')
    plt.grid(True)

    stress_max = np.round(np.max(stress), 3)  # 3 decimal nums
    return stress_max