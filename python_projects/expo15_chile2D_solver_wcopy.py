import numpy as np
import matplotlib.pyplot as plt


def Stress(xcoord0, ycoord0, xcoord, ycoord, iEdge, jEdge, numelem, E, A, F, fixedDof):
    "Linking x, ycoord with i,jEdge"

    xi0 = xcoord0[np.ix_(iEdge)]
    xj0 = xcoord0[np.ix_(jEdge)]  # take jEdge #s and replace them with corresponding xcoord
    yi0 = ycoord0[np.ix_(iEdge)]
    yj0 = ycoord0[np.ix_(jEdge)]

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

    "Active degrees of freedom (DOFs)"
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
    fig = plt.figure()
    for i in range(6):
        ax = fig.add_subplot(2, 3, i + 1)
        ax.set_title("Plot #%i" % i)

        for r in range(numelem):
            x0 = (xi0[r], xj0[r])
            y0 = (yi0[r], yj0[r])
            fig.axes[i].plot(x0, y0)
    plt.show()

