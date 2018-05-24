import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import new1x4_solver_ as slv
from matplotlib.gridspec import GridSpec
import datetime
import logger
import seaborn as sns

# CH = change if implementing on a new structure

class Individual:
    def __init__(self):
        "Structural dimentions (m)"""
        a = 2  # CH
        h = a  # triangle height  # CH

        "Original coordinates"
        xcoord = np.array([0, a, 2.5 * a, 4 * a, 5 * a, a, 2.5 * a, 4 * a])  # CH
        ycoord = np.array([0, 0, 0, 0, 0, h, h, h])  # CH

        "Choose a random number in range +- (m) from the original coordinate"
        x2GA = rnd.randrange(np.round((xcoord[2] - 0.7) * 100), np.round((xcoord[2] + 0.7) * 100)) / 100  # CH
        #y2GA = rnd.randrange(np.round((ycoord[2] - 1) * 100), np.round((ycoord[2] + 1.3) * 100)) / 100

        x1GA = rnd.randrange(np.round((xcoord[1] - 0.7) * 100), np.round((xcoord[1] + 0.7) * 100)) / 100
        #y1GA = rnd.randrange(np.round((ycoord[1] - 2) * 100), np.round((ycoord[1] + 1.3) * 100)) / 100

        x3GA = rnd.randrange(np.round((xcoord[3] - 0.7) * 100), np.round((xcoord[3] + 0.7) * 100)) / 100
        #y3GA = rnd.randrange(np.round((ycoord[3] - 1) * 100), np.round((ycoord[3] + 1.3) * 100)) / 100

        "New coordinates"
        xcoord = np.array([0, x1GA, x2GA, x3GA, 5 * a, a, 2.5 * a, 4 * a])  # CH
        ycoord = np.array([0, 0, 0, 0, 0, h, h, h])  # can use np.ix_?    # CH

        "Cross-section area (m)"
        self.A = np.random.uniform(low=0.0144, high=0.0539, size=(13,))  # area between 12x12 and 23x23cm # CH
        self.A[11] = rnd.randrange(0.0004 * 10000, 0.0064 * 10000) / 10000  # special condition for steel element # CH

        "Material characteristic E=(MPa)"
        # modulus of elasticity for each member, E_concrete = 40 000 MPa, E_steel = 210 000 MPa # CH
        self.E = np.array([40000, 40000, 40000, 40000, 40000, 40000, 40000, 40000, 40000, 40000, 40000, 210000, 40000])

        self._plot_dict = None
        self._nodes = np.array([xcoord, ycoord])
        self._deflection = 0
        self._stress = 0
        self._weight = 0
        self._fitness = 0
        self._probability = 0

    @property
    def deflection(self):
        return self._deflection

    @deflection.setter
    def deflection(self, new):
        self._deflection = new

    @property
    def stress(self):
        return self._stress

    @stress.setter
    def stress(self, new):
        self._stress = new

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, new):
        self._weight = new

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
    def probability(self, new):
        self._probability = new


class GA:
    def __init__(self, pop):
        self.mem_begin = np.array([0, 1, 2, 3, 4, 7, 6, 5, 1, 5, 6, 2, 7])  # beginning of an edge   # CH
        self.mem_end = np.array([1, 2, 3, 4, 7, 6, 5, 0, 5, 2, 2, 7, 3])    # end of an edge         # CH

        self._pool = list()
        self._popsize = pop

    def initial(self):
        for i in range(self._popsize):
            self._pool.append(Individual())
            print(
                "node_1:{} node_2:{} node_3:{}".format(
                                                 np.round([self._pool[i]._nodes[0, 1], self._pool[i]._nodes[1, 1]], 3),
                                                 np.round([self._pool[i]._nodes[0, 2], self._pool[i]._nodes[1, 2]], 3),
                                                 np.round([self._pool[i]._nodes[0, 3], self._pool[i]._nodes[1, 3]], 3)))
        print("......................")

    def calculation(self):
        numelem = len(self.mem_begin)  # count number of beginnings

        """Structural characteristics"""

        "Fixed Degrees of Freedom (DOF)"
        dof = np.zeros((2 * len(np.unique(self.mem_begin)), 1))  # dof vector  # CH
        dof[0] = 1  # 1 = fixed
        dof[1] = 1
        dof[9] = 1

        "Outside Forces [kN]"
        F = np.zeros((2 * len(np.unique(self.mem_begin)), 1))  # forces vector  # CH
        F[10] = 10
        F[11] = -15
        F[13] = -5
        F[14] = 10
        F[15] = -15

        print("calculation ")

        """Access solver"""  # deflection, stress, weight
        for i in range(self._popsize):

            "DEFLECTION"
            pool = self._pool[i]
            res = slv.deflection(pool._nodes[0], pool._nodes[1], self.mem_begin, self.mem_end, numelem,
                                              pool.E, pool.A, F, dof)
            deflection = res
            pool._deflection = deflection
            pool._probability = 0

            "STRESS"
            # globbing, to "res" save everything that slv.stress returns (tuple)
            res = slv.stress(pool._nodes[0], pool._nodes[1],
                             self.mem_begin, self.mem_end, pool.E, pool.A, F, dof, deflection)
            stress, stress_normed, xi, xj, yi, yj, xinew, xjnew, yinew, yjnew, F_x2, numnode, dof_x2, length = res
            pool._stress = stress
            pool._stress_normed = stress_normed
            pool._stress_max = np.round(np.max(pool._stress), 3)

            plot_dict = {"xi": xi, "xj": xj, "yi": yi, "yj": yj, "xinew": xinew, "xjnew": xjnew, "yinew": yinew,
                         "yjnew": yjnew, "F_x2": F_x2, "dof_x2": dof_x2,
                         "stress_normed": stress_normed, "numnode": numnode, "numelem": numelem, "A": pool.A}
            pool._plot_dict = plot_dict

            "WEIGHT"
            pool._weight = slv.weight(xi, xj, yi, yj, pool.A, length)

            print("node_1:{} node_2:{} node_3:{} |def| sum:{} |stress| sum:{} |weight| sum:{}".format(
                np.round([pool._nodes[0, 1], pool._nodes[1, 1]], 3),
                np.round([pool._nodes[0, 2], pool._nodes[1, 2]], 3),
                np.round([pool._nodes[0, 3], pool._nodes[1, 3]], 3),
                np.round(abs(pool._deflection).sum(), 3),
                np.round(abs(pool._stress).sum()),
                np.round(pool._weight.sum())))
        print("......................")

    def fitness(self):
        fitnesses = []

        "Condition to disadvantage members with stress > E"
        # if the inner force is higher than member's strength, make its fitness much worse
        for x in self._pool:
            for i in range(len(self.mem_begin)):
                for strength in self._pool[i].E:
                    if strength < abs(x._stress[i]):
                        x._stress[i] = x._stress[i] * 100

        """Rate / give fitness to each member"""
        print("fitness")  # lower fitness = better fitness

        "Conditions to find the best candidate"
        deflections = [(abs(x._deflection)).sum() for x in self._pool]  # abs sum of deflections
        stresses = [sum(abs(x._stress)).sum() for x in self._pool]  # sum of absolute stresses
        weights = [sum(x._weight).sum() for x in self._pool]  # abs sum of weights

        "Importance coefficients for stated conditions"
        deflection_coef = 0.40
        stress_coef = 0.50
        weight_coef = 0.10

        "Fill the cell based on conditions and importance coefficients"
        for deflection, stress, weight in zip(deflections, stresses, weights):
            if weight.sum() < 0:
                print("Weight is negative!")
            else:
                fitnesses.append(1/(deflection_coef * deflection +
                                 stress_coef * stress * (min(deflections)/min(stresses)) +
                                 weight_coef * weight * (min(deflections)/min(weights))))

        sum_fit = sum(fitnesses)

        "Fitness of each candidate"
        len_sf = len(self._pool)
        for i in range(len_sf):
            self._pool[i]._fitness = fitnesses[i]

            "Probability record"
            # member with lower fit (= better) has a higher probability to be chosen for the crossover
            self._pool[i]._probability = fitnesses[i] / sum_fit

        "Sort members based on probability"
        # sorting in Python is ascending (if "-x._fitness", would be descending)
        self._pool.sort(key=lambda x: -x._fitness)

        "Print results"
        for i in range(self._popsize):
            pool = self._pool[i]
            print("node_1:{} node_2:{} node_3:{} fit:{}  prob:{} "
                  "|def(mm)| sum:{} |stress(kPa)| sum:{} |weight(t)| sum:{}".format(
                np.round([pool._nodes[0, 1], pool._nodes[1, 1]], 3),
                np.round([pool._nodes[0, 2], pool._nodes[1, 2]], 3),
                np.round([pool._nodes[0, 3], pool._nodes[1, 3]], 3),
                np.round(pool._fitness, 3),
                np.round(pool._probability, 3),
                np.round(abs(pool._deflection).sum(), 3),
                np.round(abs(pool._stress).sum()),
                np.round(pool._weight.sum())))
        print("..............")

    def get_fit(self):
        "Best fitness"
        best_obj= max(self._pool, key=lambda x: x._fitness)
        return np.round(best_obj.fitness, 3)

    def _switch1(self, individual_pair, axis=0):
        # set switch values between 2 individuals (node 1)
        first = individual_pair[0]
        second = individual_pair[1]
        tmp = first._nodes[axis, 1]  # = temporary
        first._nodes[axis, 1] = second._nodes[axis, 1]
        second._nodes[axis, 1] = tmp

    def _switch2(self, individual_pair, axis=0):
        # set switch values between 2 individuals (node 2)
        first = individual_pair[0]
        second = individual_pair[1]
        tmp = first._nodes[axis, 2]  # temporary
        first._nodes[axis, 2] = second._nodes[axis, 2]
        second._nodes[axis, 2] = tmp

    def _switch3(self, individual_pair, axis=0):
        # set switch values between 2 individuals (node 3)
        first = individual_pair[0]
        second = individual_pair[1]
        tmp = first._nodes[axis, 3]  # temporary
        first._nodes[axis, 3] = second._nodes[axis, 3]
        second._nodes[axis, 3] = tmp

    def _switch_A(self, individual_pair, axis=0):
        # set switch values between 2 individuals (node 3)
        first = individual_pair[0]
        second = individual_pair[1]
        tmp = first._nodes[axis, 0]  # temporary
        first._nodes[axis, 0] = second._nodes[axis, 0]
        second._nodes[axis, 0] = tmp

    def crossover(self):
        probs = np.array([x._probability for x in self._pool])

        "Nodes crossover"
        # choose 2 individuals that will crossover
        switch_x0 = np.random.choice(self._pool, 2, replace=False, p=probs)
        switch_x1 = np.random.choice(self._pool, 2, replace=False, p=probs)
        switch_x2 = np.random.choice(self._pool, 2, replace=False, p=probs)
        #switch_y = np.random.choice(self._pool, 2, replace=False, p=probs)

        best = max(self._pool, key=lambda x: x._probability)

        self._switch1(switch_x0, 0)
        self._switch2(switch_x1, 0)
        self._switch3(switch_x2, 0)
        #self._switch2(switch_y, 1)

        "Areas Crossover"
        switch_a = np.random.choice(self._pool, 2, replace=False, p=probs)
        self._switch_A(switch_a, 0)

        "Select the best member"
        for i in range(self._popsize):
            all = self._pool[i]
            best = np.argmax(all)

        "Best member stays in population"
        for i in range(0, 1):
            if (switch_x0[i]) or (switch_x1[i]) or (switch_x2[i]) or switch_a[i] == best:
                self._pool[-1] = best
                #self._pool[-1].probability = (x._probability for x in self._pool[-1])
                self._pool[-1].probability = 1 - sum(x._probability for x in self._pool[:(len(self._pool))])

    def mutation(self, mutation_type):
        probs = [x._probability for x in self._pool]

        fits = np.array([x._fitness for x in self._pool])
        sum_f = sum(x._fitness for x in self._pool)

        if sum(probs) !=1:
            probs = abs(np.array(fits))/sum_f

        "Pick a mutation candidate"
        mutation_candidate = np.random.choice(self._pool, 1, p=probs)[0]

        possible_coefficients = [0.9, 0.9, 0.9, 1.1, 1.1, 1.1, 0.8, 0.85, 0.75, 1.3, 1.2, 1.2]
        coef = np.random.choice(possible_coefficients, 1)

        "Mutate"
        for i in range(rnd.randrange(1, 2, 3)):  # CH
            if mutation_type == "x":
                mutation_candidate._nodes[0, i] = mutation_candidate._nodes[0, i] * coef
            if mutation_type == "y":
                mutation_candidate._nodes[1, i] = mutation_candidate._nodes[1, i] * coef

        if mutation_type == "a":
            for i in range(rnd.randrange(len(self.mem_begin))):
                cur_candidate = self._pool[i]
                se = np.argmin(self._pool[i]._stress)
                if cur_candidate.A[se] > 0.0001:
                    continue
                cur_candidate.A[se] = cur_candidate.A[se] * coef
                break

        "Best member stays in population"
        for i in range(self._popsize):
            best = max(self._pool, key=lambda x: x._probability)
            if mutation_candidate == best:
                self._pool[-1] = best
                self._pool[-1].probability = 1 - sum(x._probability for x in self._pool[:(len(self._pool))])

    def plot_stress(self):
        num_to_plot = 4

        gs = GridSpec(1, 4)
        gs.update(left=0.05, right=0.95, wspace=0.2)
        # fig, ax = plt.subplots(figsize=(10, 3), sharey='col')
        fig = plt.figure(figsize=(18, 5))
        fig.suptitle("Best members in generation - stress")

        # TODO: naming Generation xx - based on the iteration

        for index in range(num_to_plot):
            # take num_to_plot best candidates, load data from saved dict
            pool = self._pool[index]
            plot_dict = pool._plot_dict
            stress = pool._stress
            xi = plot_dict['xi']
            xj = plot_dict['xj']
            yi = plot_dict['yi']
            yj = plot_dict['yj']
            xinew = plot_dict['xinew']
            xjnew = plot_dict['xjnew']
            yinew = plot_dict['yinew']
            yjnew = plot_dict['yjnew']
            stress_normed = plot_dict['stress_normed']
            F_x2 = plot_dict['F_x2']
            dof_x2 = plot_dict['dof_x2']
            numnode = plot_dict['numnode']
            numelem = plot_dict['numelem']

            ax = fig.add_subplot(gs[0, index], aspect="equal")

            ax.grid(True)
            ax.set_xlim(-1, 11)  # CH
            ax.set_ylim(-2.5, 5)  # CH
            # ax.axis('equal') solved by adding equal to "ax = "
            ax.set_title("Candidate {}".format(index + 1))

            for r in range(numelem):
                x = (xi[r], xj[r])
                y = (yi[r], yj[r])

                line = ax.plot(x, y)
                plt.setp(line, ls='-', c='black', lw='1', label='orig')

                xnew = (xinew[r], xjnew[r])
                ynew = (yinew[r], yjnew[r])

                linenew = ax.plot(xnew, ynew)

                plt.setp(linenew, ls='-',
                         c='c' if stress[r] > 0.000001 else ('red' if stress[r] < -0.000001 else 'black'),
                         lw=(1 + 20 * stress_normed[r]), label='strain' if stress[r] > 0 else 'stress')
                ax.plot()

            "Annotate outside forces"
            for r in range(numnode):
                plt.annotate(F_x2[r],
                             xy=(xi[r], yi[r]), xycoords='data', xytext=np.sign(F_x2[r]) * -35,
                             textcoords='offset pixels',
                             arrowprops=dict(facecolor='black', shrink=0, width=1.3, headwidth=5),
                             horizontalalignment='right', verticalalignment='bottom')

            "Annotate fixed DOFs"
            for r in range(numnode):
                if np.array_equal(dof_x2[r], np.array([0, 1])):
                    plt.plot([xi[r]], [yi[r] - 0.2], 'o', c='k', markersize=8)
                if np.array_equal(dof_x2[r], np.array([1, 0])):
                    plt.plot([xi[r] - 0.2], [yi[r]], 'o', c='k', markersize=8)
                if np.array_equal(dof_x2[r], np.array([1, 1])):
                    plt.plot([xi[r]], [yi[r] - 0.2], '^', c='k', markersize=8)

        plt.savefig(datetime.datetime.now().strftime('stress_%Y%m%d_%H%M%S_') + ".png", DPI=1200)

        #plt.show()

    def plot_A(self):
        # ziskej hodnoty z dictionary
        num_to_plot = 4

        gs = GridSpec(1, 4)
        gs.update(left=0.05, right=0.95, wspace=0.2)
        # fig, ax = plt.subplots(figsize=(10, 3), sharey='col')
        fig = plt.figure(figsize=(18, 5))
        fig.suptitle("Best members in generation - cross section")

        # TODO: naming Generation xx - based on the iteration
        for index in range(num_to_plot):
            # take num_to_plot best candidates, load data from saved dict
            pool = self._pool[index]
            plot_dict = pool._plot_dict
            stress = pool._stress
            xi = plot_dict['xi']
            xj = plot_dict['xj']
            yi = plot_dict['yi']
            yj = plot_dict['yj']
            xinew = plot_dict['xinew']
            xjnew = plot_dict['xjnew']
            yinew = plot_dict['yinew']
            yjnew = plot_dict['yjnew']
            stress_normed = plot_dict['stress_normed']
            F_x2 = plot_dict['F_x2']
            dof_x2 = plot_dict['dof_x2']
            numnode = plot_dict['numnode']
            numelem = plot_dict['numelem']

            ax = fig.add_subplot(gs[0, index], aspect="equal")

            ax.grid(True)
            ax.set_xlim(-1, 11)  # CH
            ax.set_ylim(-2.5, 5)  # CH
            # ax.axis('equal') solved by adding equal to "ax = "
            ax.set_title("Candidate {}".format(index + 1))

            for r in range(numelem):
                x = (xi[r], xj[r])
                y = (yi[r], yj[r])

                line = ax.plot(x, y)
                plt.setp(line, ls='-', c='black', lw='1', label='orig')

                xnew = (xinew[r], xjnew[r])
                ynew = (yinew[r], yjnew[r])

                linenewA = ax.plot(xnew, ynew)

                plt.setp(linenewA, ls='-', c='green', lw=(1 + 70 * pool.A[r]))
            ax.plot()

            "Annotate outside forces"
            for r in range(numnode):
                plt.annotate(F_x2[r],
                             xy=(xi[r], yi[r]), xycoords='data', xytext=np.sign(F_x2[r]) * -35,
                             textcoords='offset pixels',
                             arrowprops=dict(facecolor='black', shrink=0, width=1.5, headwidth=8),
                             horizontalalignment='right', verticalalignment='bottom')

            "Annotate fixed DOFs"
            for r in range(numnode):
                if np.array_equal(dof_x2[r], np.array([0, 1])):
                    plt.plot([xi[r]], [yi[r] - 0.2], 'o', c='k', markersize=8)
                if np.array_equal(dof_x2[r], np.array([1, 0])):
                    plt.plot([xi[r] - 0.2], [yi[r]], 'o', c='k', markersize=8)
                if np.array_equal(dof_x2[r], np.array([1, 1])):
                    plt.plot([xi[r]], [yi[r] - 0.2], '^', c='k', markersize=8)

        # can use Textbox if needed
        # plt.subplots(1, 2,sharex=True, sharey=True)
        plt.savefig(datetime.datetime.now().strftime('cross section_%Y%m%d_%H%M%S_') + ".png", DPI=1200)

        #plt.show()