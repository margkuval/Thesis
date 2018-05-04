import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import expo15_chile2D_solver_wcopy as slv


class Individual:
    def __init__(self):
        a = 2.5
        h = np.sqrt(pow(a, 2) - pow(a/2, 2))

        xcoord = np.array([0, a, a/2])  # CH
        ycoord = np.array([0, 0, h])    # CH

        x1GA = rnd.randrange(np.round((xcoord[2] - 0.5)*10), np.round((xcoord[2] + 0.5)*10))/10
        y1GA = rnd.randrange(np.round((ycoord[2] - 0.5)*10), np.round((ycoord[2] + 0.5)*10))/10
        # take random # from a range xcoord-2 to xcoord+2

        xcoord = np.array([0, a, x1GA])    # CH
        ycoord = np.array([0, 0, y1GA])  # can use np.ix_?    # CH

        self._nodes = np.array([xcoord, ycoord])
        self._stress = 0
        self._fitness = 0  # zmeni to hodnotu spravneho fitnessu? u Stani je nejlepsi jednec 0, ja bych ho chtela mit jako 1
        self._probability = 0

    @property
    def stress(self):
        return self._stress

    @stress.setter
    def stress(self, new):
        self._stress = new

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
            print("nodes : {}".format(np.round([self._pool[i]._nodes[0, 2], self._pool[i]._nodes[1, 2]], 3)))  # one or zero?
        print("......................")

    def calc(self):

        iEdge = np.array([0, 1, 2])  # beginning of an edge   # CH
        jEdge = np.array([1, 2, 0])  # end of an edge         # CH

        self._iEdge = iEdge
        self._jEdge = jEdge

        numelem = iEdge.shape[0]  # count # of beginnings

        """Material characteristics E=(kPa), A=(m2)"""
        E = np.array(iEdge.shape[0] * [40000])  # modulus of elasticity for each member
        A = np.array(iEdge.shape[0] * [0.0225])  # area - each member 0.15x0.15m

        "Outside Forces [kN]"
        F = np.zeros((2*len(np.unique(iEdge)), 1))  # forces vector
        F[2] = 10
        F[5] = 10

        "Fixed dof"
        fixedDof = np.array([0, 1, 3])

        print("calculation")

        for i in range(self._popsize):
            self._pool[i]._stress = slv.Stress(self._pool[i]._nodes[0], self._pool[i]._nodes[1], iEdge, jEdge, numelem, E, A, F, fixedDof)
            self._pool[i]._probability = 0  #Stana mel 0
            print("nodes : {}  stress_max : {}".format(np.round([self._pool[i]._nodes[0, 2], self._pool[i]._nodes[1, 2]], 3), self._pool[i]._stress))
        print("......................")

    def fitness(self):
        print("fitness")
        for i in range(self._popsize):
            self._pool[i]._fitness = self._pool[i]._stress / max(self._pool, key=lambda x: x._stress)._stress
        self._pool.sort(key=lambda x: x._fitness)
        sum_fit = sum(map(lambda x: x._fitness, self._pool))

        """Define probability"""
        probab = []
        for i in range(self._popsize):
            probab.append(sum_fit / self._pool[i]._fitness)
        sum_prob = sum(probab)
        """Probability record"""
        for i in range(self._popsize):
            self._pool[i]._probability = self._pool[i - 1]._probability + probab[i] / sum_prob
            print("nodes : {}  fit : {}  prob : {} ".format(np.round([self._pool[i]._nodes[0, 2], self._pool[i]._nodes[1, 2]], 3),
                                                            np.round(self._pool[i]._fitness, 3),
                                                            np.round(self._pool[i]._probability, 3)))
        print("..............")

    def crossover(self):
        selected_pool = list()
        select_num = 6
        for i in range(select_num):
            a = np.random.uniform(0, 1)
            print(round(a,3))
            for individual in self._pool:
                if individual._probability > a:
                    select_ind = individual._nodes[0, 2]
                    selected_pool.append(select_ind)
                    break
                if individual._probability > a:
                    select_ind1 = individual._nodes[1, 2]
                    selected_pool.append(select_ind1)
                    break
        for i in range(3):
            self._pool[i]._nodes[0, 2] = (selected_pool[2 * i]) / 2
            self._pool[i]._nodes[1, 2] = (selected_pool[2 * (i + 1) - 1]) / 2
        for i in range(self._popsize):
            print([self._pool[i]._nodes[0, 2], self._pool[i]._nodes[1, 2]])
        print("___________________________________")

    def plot(self):
        plt.title("Sense you no make")
        plt.ylabel('Algorithm result')
        plt.xlabel('Population size')
        plt.axis('equal')
        plt.grid(True)
        for i in range(1):  # from how many pools im taking the information
            o = self._pool[i]._nodes
            line = plt.plot(o)
            plt.setp(line, ls='-', c='black', lw='1', label='orig')
        plt.show()


# change nodes from [0, 2] and [1, 2] to relevant ones that are moving