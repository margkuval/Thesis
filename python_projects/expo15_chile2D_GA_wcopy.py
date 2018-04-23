import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import expo15_chile2D_solver as slv


class Individual:
    def __init__(self):
        a = 2.5
        h = np.sqrt(pow(a, 2) - pow(a/2, 2))

        xcoord = np.array([0, a/2, 0., a, 2*a, a + a/2, 2*a, a])
        ycoord = np.array([2 * h, h, 0., 0., 0., h, 2 * h, 2 * h])

        x1GA = rnd.randrange(np.round((xcoord[1] - 2)*10), np.round((xcoord[1] + 2)*10))/10
        y1GA = rnd.randrange(np.round((ycoord[1] - 2)*10), np.round((ycoord[1] + 2)*10))/10
        # take random # from a range xcoord-2 to xcoord+2

        xcoord = np.array([0, x1GA, 0., a, 2*a, a + a/2, 2*a, a])
        ycoord = np.array([2*h, y1GA, 0., 0., 0., h, 2*h, 2*h])  # can use np.ix_?

        self._nodes = np.array([xcoord, ycoord]).transpose()
        self._u = 1
        self._fitness = 1  # zmeni to hodnotu spravneho fitnessu? u Stani je nejlepsi jednec 0, ja bych ho chtela mit jako 1
        self._probability = 1

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, new):
        self._u = new

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, new):
        self._fitness = new

    @property
    def probability(self):
        return self._probability

    @probability.setter
    def probability(self,new):
        self._probability = new


class GA:
    def __init__(self, pop):
        self._pool = list()
        self._popsize = pop

    def initial(self):
        for i in range(self._popsize):
            self._pool.append(Individual())
            print("nodes : {}".format(np.round(self._pool[i]._nodes[1, :], 3)))
        print("......................")

    def calc(self):

        iEdge = np.array([0, 1, 2, 3, 4, 5, 6, 7, 1, 7, 5, 1, 5])  # beginning of an edge
        jEdge = np.array([1, 2, 3, 4, 5, 6, 7, 0, 7, 5, 1, 3, 3])  # end of an edge

        "Connectivity MAT computation"
        ij = np.vstack([[2 * iEdge, 2 * iEdge + 1], [2 * jEdge, 2 * jEdge + 1]]).transpose()

        """Material characteristics E=(kPa), A=(m2)"""
        E = np.array(iEdge.shape[0] * [40000])  # modulus of elasticity for each member
        A = np.array(iEdge.shape[0] * [0.0225])  # area - each member 0.15x0.15m

        "Outside Forces [kN]"  # forces vector
        F = np.zeros((2*len(np.unique(iEdge)), 1))
        F[0] = 0
        F[4] = 0
        F[13] = 15

        "Fixed nodes"
        fixedDof = np.array([0, 1, 7])
        print("calculation")

        for i in range(self._popsize):
            self._pool[i]._u = slv.stress(self._pool[i]._nodes, iEdge, jEdge, ij, E, A, F, fixedDof)
            self._pool[i]._probability = 1  #Stana mel 0
            print("nodes : {}  u_max : {}".format(np.round(self._pool[i]._nodes[1, :], 3), self._pool[i]._u))
        print("......................")
        # def posuny(XZ, spoj, EA, F):

    def fitness(self):
        print("fitness")
        for i in range(self._popsize):
            self._pool[i]._fitness = self._pool[i]._u / max(self._pool, key=lambda x: x._u)._u
        self._pool.sort(key=lambda x: x._fitness)
        sum_fit = sum(map(lambda x: x._fitness, self._pool))
        ###### urceni pravdepodobnosti #####
        probab = []
        for i in range(self._popsize):
            probab.append(sum_fit / self._pool[i]._fitness)
        sum_prob = sum(probab)
        ###### zapsani pravdepodobnosti #####
        for i in range(self._popsize):
            self._pool[i]._probability = self._pool[i - 1]._probability + probab[i] / sum_prob
            print("nodes : {}  fit : {}  prob : {} ".format(np.round(self._pool[i]._nodes[1, :], 3),
                                                           np.round(self._pool[i]._fitness, 3),
                                                           np.round(self._pool[i]._probability, 3)))
        print("..............")

    def crossover(self):
        selected_pool = list()
        select_num = 6
        for i in range(select_num):
            a = np.random.uniform(0, 1)
            print(round(a, 3))
            select_ind = self._pool[0]._nodes[1, :]
            for individual in self._pool:
                if individual._probability > a:
                    select_ind = individual._nodes[1, :]
                    selected_pool.append(select_ind)
                    break
        # print(selected_pool)
        for i in range(3):
            self._pool[i]._nodes[1, :] = (selected_pool[2 * i] + selected_pool[2 * (i + 1) - 1]) / 2
        for i in range(self._popsize):
            print(self._pool[i]._nodes[1, :])
        print("___________________________________")
