import numpy as np


def deflection(xcoord, ycoord, mem_begin, mem_end, numelem, E, A, F, dof):

    "Link x and y coordinates with member's beginning and end"""
    xi = xcoord[np.ix_(mem_begin)]
    xj = xcoord[np.ix_(mem_end)]  # take mem_end numbers and replace them with corresponding xcoord
    yi = ycoord[np.ix_(mem_begin)]
    yj = ycoord[np.ix_(mem_end)]

    "Connectivity matrix"
    ij = np.vstack([[2 * mem_begin, 2 * mem_begin + 1], [2 * mem_end, 2 * mem_end + 1]]).transpose()

    "Other information"
    numnode = xcoord.shape[0]  # number of nodes, xcoord because all nodes are used
    dof_tot = 2 * numnode  # total degrees of freedom

    """Global Stiffness Matrix"""
    glob_stif = np.zeros((dof_tot, dof_tot))  # empty Global Stiffness Matrix
    length = np.sqrt(pow((xj - xi), 2) + pow((yj - yi), 2))  # defines length of members
    c = (xj - xi) / length  # cos
    s = (yj - yi) / length  # sin

    for p in range(numelem):
        # takes p from the range of numelem 1 by 1 and creates multiple k1 (local) matrices
        # at the end maps k1 matices on right places in glob_stiff matrix
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

    "Fixed and active DOFs"
    dof_a = np.array(np.where(dof == 0))  # node where dof = 0 is an active node
    dof_active = dof_a[0]

    "Solve deflections"
    u = np.zeros((dof_tot, 1))  # empty deflections matrix; 1 = number of columns
    u1 = np.linalg.solve(glob_stif[np.ix_(dof_active, dof_active)],
                         F[np.ix_(dof_active)]) # solve equation glob_stif*u = F
    u[np.ix_(dof_active)] = u1  # map back to the empty deflection matrix

    u = np.round(u, 3)
    deflection = u

    return deflection


def stress(xcoord, ycoord, mem_begin, mem_end, E, A, F, dof, deflection):

    "Link x and y coordinates with member's beginning and end"""
    xi = xcoord[np.ix_(mem_begin)]
    xj = xcoord[np.ix_(mem_end)]
    yi = ycoord[np.ix_(mem_begin)]
    yj = ycoord[np.ix_(mem_end)]

    "Other information"
    numnode = xcoord.shape[0]

    "Reshaped outside forces and DOFs"
    F_x2 = F.reshape(numnode, 2)
    dof_x2 = dof.reshape(numnode, 2)  # reshape for plotting

    """Stress calculation"""
    "Deflections in x,y directions"
    # using mem_end and mem_begin to calculate new nodes location
    length = np.sqrt(pow((xj - xi), 2) + pow((yj - yi), 2))  # members length
    k = E * A / length

    u = deflection
    uxi = u[np.ix_(2 * mem_begin)].transpose()
    uxj = u[np.ix_(2 * mem_end)].transpose()
    uyi = u[np.ix_(2 * mem_begin + 1)].transpose()
    uyj = u[np.ix_(2 * mem_end + 1)].transpose()

    "Inner forces"
    c = (xj - xi) / length  # cos
    s = (yj - yi) / length  # sin

    Flocal = k * ((uxj - uxi) * c + (uyj - uyi) * s)

    "Stress (kPa)"
    stress = Flocal[0] / A
    stress_normed = [i / sum(abs(stress)) for i in abs(stress)]

    xinew = xi + uxi[0]  # [[ in u array, now solved by taking "list 0" from the MAT
    xjnew = xj + uxj[0]
    yinew = yi + uyi[0]
    yjnew = yj + uyj[0]

    return stress, stress_normed, xi, xj, yi, yj, xinew, xjnew, yinew, yjnew, F_x2, numnode, dof_x2, length


def weight(A, xi, xj, yi, yj, length):

    "Density of each element (kg/m3)"""
    # reinforced concrete = 2500 kg/m3, steel = 7700 kg/m3
    ro = np.array([2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 7700, 2500])

    "Weigth calculation"
    weight = length * A * ro

    return weight