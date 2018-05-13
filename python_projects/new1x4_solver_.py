import numpy as np
import matplotlib.pyplot as plt


def stress(xcoord, ycoord, mem_begin, mem_end, numelem, E, A, F, dof):

    "Link x, ycoord with member begin and end"""
    xi = xcoord[np.ix_(mem_begin)]
    xj = xcoord[np.ix_(mem_end)]  # take mem_end numbers and replace them with corresponding xcoord
    yi = ycoord[np.ix_(mem_begin)]
    yj = ycoord[np.ix_(mem_end)]

    "Connectivity MAT computation"
    ij = np.vstack([[2 * mem_begin, 2 * mem_begin + 1], [2 * mem_end, 2 * mem_end + 1]]).transpose()

    "Other information"
    numnode = xcoord.shape[0]  # number of nodes, xcoord b/c all nodes must be used
    dof_tot = 2 * numnode  # total degrees of freedom

    """Global stiffness MAT"""
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
    dof_a = np.array(np.where(dof == 0))  # node where dof = 0 is an active node
    dof_active = dof_a[0]

    dof_totx2 = dof.reshape(numnode, 2)  # reshape for plotting

    "Solve deflections"
    u = np.zeros((dof_tot, 1))  # empty deflections MAT; 1 = # of columns
    u1 = np.linalg.solve(glob_stif[np.ix_(dof_active, dof_active)], F[np.ix_(dof_active)])  # solve equation glob_stif*u = F
    u[np.ix_(dof_active)] = u1  # map back to the empty def MAT

    "Deflections calculation"
    # for each node in both directions
    # important to use _end and _begin to calculate new nodes location
    k = E * A / length
    uxi = u[np.ix_(2 * mem_begin)].transpose()
    uxj = u[np.ix_(2 * mem_end)].transpose()
    uyi = u[np.ix_(2 * mem_begin + 1)].transpose()
    uyj = u[np.ix_(2 * mem_end + 1)].transpose()

    "Inner forces"
    Flocal = k * ((uxj - uxi) * c + (uyj - uyi) * s)  # c=cos,s=sin
    print(Flocal)
    """Stress (sigma)=(kPa)"""
    stress = Flocal[0] / A
    stress_normed = [i / sum(abs(stress)) for i in abs(stress)]
    xinew = xi + uxi[0]  # [[ in u array, now solved by taking "list 0" from the MAT
    xjnew = xj + uxj[0]
    yinew = yi + uyi[0]
    yjnew = yj + uyj[0]

    return stress, stress_normed, xi, xj, yi, yj, xinew, xjnew, yinew, yjnew, F_numnodex2, numnode, dof_totx2


def weight(xcoord, ycoord, mem_begin, mem_end, A):
    ro = 2500  # kg/m3, reinforced concrete  # http://www.boeingconsult.com/tafe/general/symbols&units.html

    xi = xcoord[np.ix_(mem_begin)]
    xj = xcoord[np.ix_(mem_end)]  # take mem_end #s and replace them with corresponding xcoord
    yi = ycoord[np.ix_(mem_begin)]
    yj = ycoord[np.ix_(mem_end)]

    length = np.sqrt(pow((xj - xi), 2) + pow((yj - yi), 2))  # mems (edges) length
    weight = length * A * ro
    weight_max = np.round(np.max(weight), 3)
    weight_sum = np.round(weight.sum(), 3)

    return weight_sum

