# (c) Rory Clune 05/16/2011

from numpy import *
from structures import *
from SearchAlgorithms import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # -----------
    # INPUT DATA FOR A MORE COMPLEX STRUCTURE
    # -----------
    # The geometry of the structural nodes (10-bar truss)
    geometry = array([[0., 0.], [5., 0.], [10., 0.], [15., 0.], [15., 5.], [10., 5.], [5., 5.], [0., 5.]])
    # A connectivity matrix defines how nodes are connected with beams. a row contains ( beam - node - node )
    connectivity = array(
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 1], [0, 6], [6, 1], [1, 5], [6, 2], [5, 2], [2, 4],
         [5, 3]], dtype=int)
    # Restrained degrees of freedom
    fixity = array([21, 22, 0, 1])
    load = zeros((3 * geometry.shape[0]));
    load[array([4, 7, 10])] = -100
    A = 10 * ones(connectivity.shape[0])
    I = 10 * ones(connectivity.shape[0])
    s = structure(geometry, connectivity, fixity, load, A, I)
    s.show(30, 30, 100, 150, array([4, 7, 10]))

    # -----------
    # RUN OPTIMIZATION
    # -----------
    s.Objective = s.EnergyObjective
    opt_geom = geometry[1:7, 1]
    opt_geom = append(opt_geom, (s.A, s.I))


    def input_x(self, x):
        self.geometry[1:7, 1] = x[0:6]
        self.A = x[6:6 + self.A.shape[0]]
        self.I = x[6 + self.A.shape[0]:6 + 2 * self.A.shape[0]]


    s.input_x = input_x

    #    UB = array([3., 3., 3., 7., 7., 7.])
    #    LB = array([-2, -2, -2, 3.5, 3.5, 3.5])
    #    scale = array([1., 1., 1., 1., 1., 1.])
    UB = append(append(array([3, 3, 3, 7, 7, 7]), 2000. * ones_like(A)), 2000. * ones_like(I))
    LB = append(append(array([-2, -2, -2, 3.5, 3.5, 3.5]), 10. * ones_like(A)), 10. * ones_like(I))
    scale = append(append(array([1., 1., 1., 1., 1., 1.]), ones_like(A)), ones_like(I))

    opt_geom *= (1 / scale);
    UB *= (1 / scale);
    LB *= (1 / scale)
    optimizer = NelderMead(opt_geom.shape[0], s, opt_geom, UB, LB, 1e-10, 15000)
    optimizer.scale = scale
    optimizer.Run()

    f = optimizer.obj_history
    plt.plot(f);
    plt.title('Objective function value achieved');
    plt.show()
    print
    'start'
    s.show(30, 30, 100, 150, array([4, 7, 10]));
    print
    'end'