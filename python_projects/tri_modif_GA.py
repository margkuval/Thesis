import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import tri_modif_solver as slv


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
        self.A = np.array(3 * [0.0225])  # area - each member 0.15x0.15m TROJUHELNIK  # CH

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
        self.iEdge = np.array([0, 1, 2])  # beginning of an edge   # CH
        self.jEdge = np.array([1, 2, 0])  # end of an edge         # CH

        self._pool = list()
        self._popsize = pop

    def initial(self):
        for i in range(self._popsize):
            self._pool.append(Individual())
            print("nodes : {}".format(np.round([self._pool[i]._nodes[0, 2], self._pool[i]._nodes[1, 2]], 3)))
        print("......................")

    def calc(self):
        numelem = self.iEdge.shape[0]  # count # of beginnings

        """Material characteristics E=(kPa), A=(m2)"""
        E = np.array(self.iEdge.shape[0] * [40000])  # modulus of elasticity for each member, now all concrete

        "Outside Forces [kN]"
        F = np.zeros((2*len(np.unique(self.iEdge)), 1))  # forces vector
        F[2] = 10
        F[5] = 10

        "Fixed dof"
        fixedDof = np.array([0, 1, 3])

        print("calculation")

        for i in range(self._popsize):
            pool = self._pool[i]
            pool._stress = slv.Stress(pool._nodes[0], pool._nodes[1], self.iEdge, self.jEdge, numelem, E, pool.A, F, fixedDof)
            pool._probability = 0  #Stana mel 0
            print("nodes : {}  stress_max : {}".format(np.round([pool._nodes[0, 2], pool._nodes[1, 2]], 3), pool._stress))
        print("...")

        for i in range(self._popsize):
            pool = self._pool[i]
            pool._weight = slv.weight(pool._nodes[0], pool._nodes[1], self.iEdge, self.jEdge, pool.A)
            pool._probability = 0  #Stana mel 0
            print("nodes : {}  weight_sum : {}".format(np.round([pool._nodes[0, 2], pool._nodes[1, 2]], 3), pool._weight))
        print("......................")

    def fitness(self):
        print("fitness")
        for i in range(self._popsize):
            self._pool[i]._fitness = self._pool[i]._stress / max(self._pool, key=lambda x: x._stress)._stress
        self._pool.sort(key=lambda x: x._fitness)  # lambda = jdi pres kazdy ind a dej mi fitness
        sum_fit = sum(map(lambda x: x._fitness, self._pool))

        """Define probability"""
        probab = []
        for i in range(self._popsize):
            probab.append(sum_fit / self._pool[i]._fitness)
        sum_prob = sum(probab)
        """Probability record"""
        for i in range(self._popsize):
            pool = self._pool[i]
            pool._probability = pool._probability + probab[i] / sum_prob
            print("nodes : {}  fit : {}  prob : {} ".format(np.round([pool._nodes[0, 2], pool._nodes[1, 2]], 3),
                                                            np.round(pool._fitness, 3),
                                                            np.round(pool._probability, 3)))
        print("..............")

    def crossover_top_3(self):
        selected_pool_x = list()
        selected_pool_y = list()
        select_num = 6  # 6x se vybere pravdepodobnost - projde se vsemi ind - pokud tri ma vetsi prob nez co se vybraly,
        #  tak se oba pridaji do selected pool
        possible_x = []
        possible_y = []
        print(len(self._pool))
        #  vybereme vsechna mozna x a y ze vsech bodu
        for individual in self._pool:
            possible_x.append(individual._nodes[0, 2])
            possible_y.append(individual._nodes[1, 2])
        #  z teech x a y vybereme nahodne 3 x a 3 y
        selected_pool_x = np.random.choice(possible_x, 3)
        selected_pool_y = np.random.choice(possible_y, 3)

        #  Do firstch tri (tedy nejhorsich) ulozime nove x a nove y
        for i in range(3):
            self._pool[i]._nodes[0, 2] = selected_pool_x[i]  # / 2
            self._pool[i]._nodes[1, 2] = selected_pool_y[i]  # / 2
        for i in range(self._popsize):
            print([self._pool[i]._nodes[0, 2], self._pool[i]._nodes[1, 2]])
        print("___________________________________")

    def _switch(self, individual_pair, axis=0):
        # prohod hodnoty mezi 2 individualy
        # axis 0 -> prohod x
        # axis 1 -> prohod y
        first = individual_pair[0]
        second = individual_pair[1]
        tmp = first._nodes[axis, 2]  # temporary
        first._nodes[axis, 2] = second._nodes[axis, 2]
        second._nodes[axis, 2] = tmp

    def crossover(self):
        # vyber individualy co se prohodi
        # TODO: zakomponovat pravdepodobnost do np.random.choice
        probs = [x._probability for x in self._pool]
        switch_x = np.random.choice(self._pool, 2, replace=False, p=probs)  #vyber dvojice na prohozeni x
        switch_y = np.random.choice(self._pool, 2, replace=False, p=probs)
        #  vybereme dvojici na prohozeni A
        switch_a =  np.random.choice(self._pool, 2, replace=False, p=probs)
        first_A = switch_a[0]
        second_A = switch_a[1]
        # prohodime A
        tmp = first_A.A
        first_A = second_A.A
        second_A = tmp

        self._switch(switch_x, 0)
        self._switch(switch_y, 1)

    def mutate(self, mutation_type):
        probs = []  #  bude pole pravdepodobnosti
        for individual in self._pool:
            probs.append(individual._probability)  # append = pridej nakonec
        # vyber kandidata na mutaci, muhehe
        mutation_candidate = np.random.choice(self._pool, 1, p=probs)[0]
        possible_coefficients = [0.9, 0.9, 0.9, 1.1, 1.2, 0.8, 0.75, 1.3, 1.2, 1.1]
        coefficient = np.random.choice(possible_coefficients, 1)
        if mutation_type == "x":
            mutation_candidate._nodes[0, 2] = mutation_candidate._nodes[0, 2] * coefficient
        if mutation_type == "y":
            mutation_candidate._nodes[1, 2] = mutation_candidate._nodes[1, 2] * coefficient
        if mutation_type == "a":
            # TODO: mutovat kazdou osu zvlast, cim vetsi napeti v ose, tim vetsi prurez
            # TODO: dat maximalni a minimalni hodnoty A, x, y
            Arnd = np.random.choice(3, 1)
            mutation_candidate.A[Arnd] = mutation_candidate.A[Arnd] * coefficient
            print(mutation_candidate.A)

    def mutate_worst(self):
        possible_coefficients = [0.9, 1.1, 1.2, 0.8, 0.75, 1.3, 1.2]
        x_coefficient = np.random.choice(possible_coefficients, 1)
        y_coefficient = np.random.choice(possible_coefficients, 1)
        choice = np.random.randint(0, 3)
        self._pool[choice]._nodes[0, 2] *= x_coefficient
        self._pool[choice]._nodes[1, 2] *= y_coefficient

    def plot(self):
        plt.title("Sense you make")
        plt.ylabel('Algorithm result')
        plt.xlabel('Population size')
        plt.axis('equal')
        plt.grid(True)
        for i in range(2):  # from how many pools im taking the information
            for j in range(1):
                for l in range(2):
                    o = (self._pool[i]._nodes[0, l], self._pool[i]._nodes[0, l+1])
                    p = (self._pool[i]._nodes[1, l], self._pool[i]._nodes[1, l+1])
                    # chybi tam provazanost s ij matici, nevi to, kde to zacina a kde konci
                    line = plt.plot(o, p)
                    plt.setp(line, ls='-', c='black', lw='1', label='orig')



# change nodes from [0, 2] and [1, 2] to relevant ones that are moving