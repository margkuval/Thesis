import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import expo15_chile2D_solver as slv


class Individual:
    def __init__(self):
        # structure is made from triangles with same side value a = 2.5m
        # to define precise coordinates, hight will be used as h
        a = 2.5
        h = np.sqrt(pow((a), 2) - pow((a / 2), 2))

        xcoord = np.array([0, a / 2, 0., a, 2 * a, a + a / 2, 2 * a, a])
        ycoord = np.array([2 * h, h, 0., 0., 0., h, 2 * h, 2 * h])
        iEdge = np.array([0, 1, 2, 3, 4, 5, 6, 7, 1, 7, 5, 1, 5])  # beginning of an edge
        jEdge = np.array([1, 2, 3, 4, 5, 6, 7, 0, 7, 5, 1, 3, 3])  # end of an edge
        print(xcoord)

        ## prehodit sem vse, co je potreba menit. Solver bych chtela zanechat jako proste programek, co je schopen pocitat, ale data se sypou sem
        x = xcoord
        y = ycoord
        a = rnd.randrange(x - 1, x + 1) / 10
        b = rnd.randrange(y - 1, y + 1) / 10  ###snazim se dostat. Chci aby nodes 1 a 5 (prozatim) si nasly svoje misto, dle zatizeni.
        #chci, aby se jejich poloha odvijela od polohy stavajici a byla v rozmezi +- 0.5 metr od stavajici

        self._nodes = np.array([[0, 0], [a, b], [5, 0]])  ##da se to udelat tak, aby to bralo nodes ze solveru, nebo je mozna potreba to sem pretahnout..
        self._u = 1
        self._fitness = 1  ##zmeni to hodnotu spravneho fitnessu? u Stani je nejlepsi jednec 0, ja bych ho chtela mit jako 1
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
            self._pool.append(individual())
            print("body : {}".format(np.round(self._pool[i]._body[1, :], 3)))
        print("......................")

    def vypocet(self):
        E = 2.1 * 10 ** 11  ##210GPa - odpovida oceli
        A = np.pi * 0.04 ** 2 - np.pi * (0.04 - 0.002) ** 2
        EA = E * A
        spoj = np.array([[0, 1], [1, 2], [0, 2]])  # pořadí napojení prutů#
        print("vypocet")
        F = 15000
        for i in range(self._popsize):
            self._pool[i]._posun = solve.posuny(self._pool[i]._body, spoj, EA, F)
            self._pool[i]._probability = 0
            print("body : {}  u_max : {}".format(np.round(self._pool[i]._body[1, :], 3), self._pool[i]._posun))
        print("......................")
        # def posuny(XZ, spoj, EA, F):

    def fitness(self):
        print("fitness")
        for i in range(self._popsize):
            self._pool[i]._fitness = self._pool[i]._posun / max(self._pool, key=lambda x: x._posun)._posun
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
            print("body : {}  fit : {}  prob : {} ".format(np.round(self._pool[i]._body[1, :], 3),
                                                           np.round(self._pool[i]._fitness, 3),
                                                           np.round(self._pool[i]._probability, 3)))
        print("..............")

    def crossover(self):
        selected_pool = list()
        select_num = 6
        for i in range(select_num):
            a = np.random.uniform(0, 1)
            print(round(a, 3))
            select_ind = self._pool[0]._body[1, :]
            for individual in self._pool:
                if individual._probability > a:
                    select_ind = individual._body[1, :]
                    selected_pool.append(select_ind)
                    break
        # print(selected_pool)
        for i in range(3):
            self._pool[i]._body[1, :] = (selected_pool[2 * i] + selected_pool[2 * (i + 1) - 1]) / 2
        for i in range(self._popsize):
            print(self._pool[i]._body[1, :])
        print("___________________________________")
