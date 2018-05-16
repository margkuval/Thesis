import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import new1x4_solver_ as slv
from matplotlib.gridspec import GridSpec
import datetime

# CH = change if implementing on a new structure

class Individual:
    def __init__(self):

        "Structural dimentions"""
        a = 2  #CH
        h = a  # triangle height  # CH

        "Original coordinates"
        xcoord = np.array([0, a, 2.5*a, 4*a, 5*a, a, 2.5*a, 4*a])  # CH
        ycoord = np.array([0, 0, 0, 0, 0, h, h, h])    # CH

        "Take a random number in range +-0.5m from the original coordinate"
        x2GA = rnd.randrange(np.round((xcoord[2] - 0.7)*10), np.round((xcoord[2] + 0.7)*10))/10
        y2GA = rnd.randrange(np.round((ycoord[2] - 1)*10), np.round((ycoord[2] + 1.3)*10))/10

        x1GA = rnd.randrange(np.round((xcoord[1] - 0.7) * 10), np.round((xcoord[1] + 0.7) * 10)) / 10
        y1GA = rnd.randrange(np.round((ycoord[1] - 2) * 10), np.round((ycoord[1] + 1.3) * 10)) / 10

        x3GA = xcoord[4] - x1GA
        y3GA = y1GA

        "New coordinates"
        xcoord = np.array([0, x1GA, x2GA, x3GA, 5*a, a, 2.5*a, 4*a])     # CH
        ycoord = np.array([0, y1GA, y2GA, y3GA, 0, h, h, h])    # can use np.ix_?    # CH
        self.A = np.random.uniform(low=0.0144, high=0.0539, size=(13,))   # area between 12x12 and 23x23cm # CH
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
    def probability(self,new):
        self._probability = new


class GA:
    def __init__(self, pop):
        self.mem_begin = np.array([0, 1, 2, 3, 4, 7, 6, 5, 1, 5, 6, 2, 7])  # beginning of an edge   # CH
        self.mem_end =   np.array([1, 2, 3, 4, 7, 6, 5, 0, 5, 2, 2, 7, 3])  # end of an edge         # CH

        self._pool = list()
        self._popsize = pop

    def initial(self):
        for i in range(self._popsize):
            self._pool.append(Individual())
            print("node_1 : {} node_2 : {}".format(np.round([self._pool[i]._nodes[0, 1], self._pool[i]._nodes[1, 1]], 3),
                                                   np.round([self._pool[i]._nodes[0, 2], self._pool[i]._nodes[1, 2]], 3)))
        print("......................")

    def calc(self):
        numelem = self.mem_begin.shape[0]  # count number of beginnings

        """Structural characteristics"""

        "Material characteristics E=(kPa)"
        E = np.array(self.mem_begin.shape[0] * [40000])  # modulus of elasticity for each member, now all concrete

        "Fixed Degrees of Freedom (DOF)"
        dof = np.zeros((2*len(np.unique(self.mem_begin)), 1))  # dof vector  # CH
        dof[0] = 1
        dof[1] = 1
        dof[9] = 1

        "Outside Forces [kN]"
        F = np.zeros((2*len(np.unique(self.mem_begin)), 1))  # forces vector  # CH
        F[11] = -15
        F[13] = -5
        F[15] = -15

        print("calculation ")

        "Access solver"  # inner forces, stress, weight

        # DEFLECTION
        for i in range(self._popsize):
            pool = self._pool[i]
            pool._deflection = slv.deflection(pool._nodes[0], pool._nodes[1], self.mem_begin, self.mem_end, numelem,
                                              E, pool.A, F, dof)
            pool._probability = 0
            print("node_1: {} node_2 : {} abs_deflection_sum : {}".format(np.round([pool._nodes[0, 1], pool._nodes[1, 1]], 3),
                                                                          np.round([pool._nodes[0, 2], pool._nodes[1, 2]], 3),
                                                                          np.round(abs(pool._deflection).sum(), 3)))

        # STRESS
        for i in range(self._popsize):
            pool = self._pool[i]
            # globbing, to "res" save everything that slv.stress returns (tuple of 11)
            res = slv.stress(pool._nodes[0], pool._nodes[1], self.mem_begin, self.mem_end, numelem, E, pool.A, F, dof)
            stress, stress_normed, xi, xj, yi, yj, xinew, xjnew, yinew, yjnew, F_numnodex2, numnode, dof_totx2 = res
            pool._stress = stress
            pool._stress_normed = stress_normed

            plot_dict = {"xi":xi, "xj":xj, "yi":yi, "yj":yj, "xinew": xinew, "xjnew":xjnew, "yinew" : yinew, "yjnew":yjnew,
                         "F_numnodex2": F_numnodex2, "dof_totx2": dof_totx2, "stress_normed": stress_normed, "numnode": numnode,
                         "numelem": numelem, "A": pool.A}
            pool._plot_dict = plot_dict

            pool._stress_max = np.round(np.max(pool._stress), 3)
            pool._probability = 0
            #print(pool._stress)
            print("nodes : {}  stress_max : {}".format(np.round([pool._nodes[0, 1], pool._nodes[1, 1]],3),
                                                       np.round([pool._nodes[0, 2], pool._nodes[1, 2]],3),
                                                       np.round(pool._stress_max), 3))

        print("...")
        # WEIGHT
        for i in range(self._popsize):
            pool = self._pool[i]
            pool._weight = slv.weight(pool._nodes[0], pool._nodes[1], self.mem_begin, self.mem_end, pool.A)
            pool._probability = 0
            print("node_1 : {} node_2 : {}  weight_sum : {}".format(np.round([pool._nodes[0, 1], pool._nodes[1, 1]],3),
                                                                    np.round([pool._nodes[0, 2], pool._nodes[1, 2]],3),
                                                                    np.round(pool._weight.sum(), 3)))
        print("......................")

    def fitness(self):
        print("fitness")

        # list comprehension, create a list that has following char. Takes values one by one from self._pool
       #deflectionn = [[(abs(x._deflection[0, 1])/abs(sum(x._deflection))).sum() for x in self._pool]] I thought that defl should matter only at top nodes
        deflections = [(abs(x._deflection)/abs(sum(x._deflection))).sum() for x in self._pool]
        stresses = [(abs(x._stress)/abs(sum(x._stress))).sum() for x in self._pool]
        #stresses_tension = [sum((x._stress) for x in self._pool if x._stress < 0)]
        weights = [(abs(x._weight)/abs(sum(x._weight))).sum() for x in self._pool]

        #print(deflections)
        #print(stresses)
        #print(weights)
        # coef based on importance
        deflection_coef = 0.5
        stress_coef = 0.3
        weight_coef = 0.2
        #stresses_tension_coef = 11

        #d = [(1 -(i / sum(abs(self._pool[i]._deflection))) for i in abs(self._popsize))]
        #d_n = sum(d)
        #d_n_coef = deflection_coef * d_n


        #deflections_normed = [((deflection_coef * (sum(1- i / sum(abs(x._deflection)))) for i in abs(x._deflection))) for x in self._pool]
        #print(deflections_normed)
        #print(deflections)
        #stresses_normed = [(stress_coef * (1 - i / sum(abs(x._stress))) for i in abs(x._stress)) for x in self._pool]
        #weights_normed = [(weight_coef * (1 - i / sum(x._weight)) for i in x._weight) for x in self._pool]

        #ttt = deflections_normed + stresses_normed + weights_normed
        #print(sum(ttt))

        fitnesses = []

        #for deflection, stress, weight in zip(deflections_normed, stresses_normed, weights_normed):
         #   fitnesses.append(sum(sum(deflections_normed), sum(stresses_normed), sum(weights_normed)))

        # 2 variables, need to connect them together\

        for deflection, stress, weight in zip(deflections, stresses, weights):
            if weight.sum() < 0:
                print("Weight is negative!")
            else:
                fitnesses.append(deflection_coef * 20 * deflection + stress_coef * stress + weight_coef * weight)

        sum_fit = sum(fitnesses)

        "Fitness for each candidate"
        for i in range(len(self._pool)):
            self._pool[i]._fitness = fitnesses[i]
            self._pool[i]._probability =  fitnesses[i]/sum_fit
        # sort in py is ascending, so "-" is needed
        self._pool.sort(key=lambda x: x._fitness)  # lambda = go through each individual and give fitness

        """Define/create probability"""
        # create empty cell, i-times add a value at the end
        # higher individual fitness -> higher probab (a member will be chosen for a mutation with higher probability)
        # TODO: change either: better fitness - lower mutation probab or higher crossover probab
        """Probability record"""
        for i in range(self._popsize):
            pool = self._pool[i]
            print("node_1 : {} node_2 : {}  fit : {}  prob : {} area : {} ".format(np.round([pool._nodes[0, 1], pool._nodes[1, 1]], 3),
                                                                             np.round([pool._nodes[0, 2], pool._nodes[1, 2]], 3),
                                                                             np.round(pool._fitness, 3),
                                                                             np.round(pool._probability, 3),
                                                                             np.round(pool.A, 3)))
        print("..............")

    ### NOT USED
    def crossover_top_3(self):
        selected_pool_x = list()
        selected_pool_y = list()
        select_num = 6  # 6x se vybere pravdepodobnost - projde se vsemi ind - pokud tri ma vetsi prob nez co se vybraly,
        #  tak se oba pridaji do selected pool
        possible_x = []
        possible_y = []
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
    ###

    def _switch1(self, individual_pair, axis=0):
        # switch values between 2 individuals
        # axis 0 -> switch x
        # axis 1 -> switch y
        # todo: pomoc s komentovanim teto casti kodu

        first = individual_pair[0]
        second = individual_pair[1]
        tmp = first._nodes[axis, 1]  # temporary
        first._nodes[axis, 1] = second._nodes[axis, 1]
        second._nodes[axis, 1] = tmp

    def _switch2(self, individual_pair, axis=0):
        # switch values between 2 individuals
        # axis 0 -> switch x
        # axis 1 -> switch y
        # todo: pomoc s komentovanim teto casti kodu

        first = individual_pair[0]
        second = individual_pair[1]
        tmp = first._nodes[axis, 2]  # temporary
        first._nodes[axis, 2] = second._nodes[axis, 2]
        second._nodes[axis, 2] = tmp

    def crossover1(self):
        # choose 2 individuals that will switch
        probs = [x._probability for x in self._pool]
        switch_x = np.random.choice(self._pool, 2, replace=False, p=probs)
        switch_y = np.random.choice(self._pool, 2, replace=False, p=probs)

        self._switch1(switch_x, 0)
        self._switch1(switch_y, 1)

        for i in range(self._popsize):
            self._pool[i]._nodes[0, 3] = self._pool[i]._nodes[0, 4] - self._pool[i]._nodes[0, 1]
            self._pool[i]._nodes[1, 3] = self._pool[i]._nodes[1, 1]
            continue

    def crossover2(self):
        # choose 2 individuals that will switch
        probs = [x._probability for x in self._pool]
        switch_x = np.random.choice(self._pool, 2, replace=False, p=probs)
        switch_y = np.random.choice(self._pool, 2, replace=False, p=probs)

        print(switch_x)
        print(switch_y)

        self._switch2(switch_x, 0)
        self._switch2(switch_y, 1)

        for i in range(self._popsize):
            self._pool[i]._nodes[0, 3] = self._pool[i]._nodes[0, 4] - self._pool[i]._nodes[0, 1]
            self._pool[i]._nodes[1, 3] = self._pool[i]._nodes[1, 1]
            continue

        "Areas Crossover"
        # matrix with one column only #switches 3...idk
        switch_a =  np.random.choice(self._pool, 2, replace=False, p=probs)
        first_A = switch_a[0]
        second_A = switch_a[1]
        tmp = first_A.A
        first_A = second_A.A
        second_A = tmp

    def mutate1(self, mutation_type):
        # create empty cell for probability
        probs = []
        for individual in self._pool:
            probs.append(individual._probability)  # append = add to the end

        # pick a mutation candidate
        # todo: co znamena p=probs v dalsim radku?
        mutation_candidate = np.random.choice(self._pool, 1, p=probs)[0]
        possible_coefficients = [0.9, 0.9, 0.9, 1.1, 1.2, 0.8, 0.75, 1.3, 1.2, 1.1]
        coef = np.random.choice(possible_coefficients, 1)

        if mutation_type == "x":
            mutation_candidate._nodes[0, 1] = mutation_candidate._nodes[0, 1] * coef
        if mutation_type == "y":
            mutation_candidate._nodes[1, 1] = mutation_candidate._nodes[1, 1] * coef
        if mutation_type == "a":
            # TODO: mutovat kazdou osu zvlast, cim vetsi napeti v ose, tim vetsi prurez
            # TODO: dat maximalni a minimalni hodnoty A, x, y
            for i in range(self._popsize):
                cur_candidate = self._pool[i]
                se = np.argmin(self._pool[i]._stress)
                if cur_candidate.A[se] > 0.01:
                    continue
                cur_candidate.A[se] = cur_candidate.A[se] * coef

        for i in range(self._popsize):
            self._pool[i]._nodes[0, 3] = self._pool[i]._nodes[0, 4] - self._pool[i]._nodes[0, 1]
            self._pool[i]._nodes[1, 3] = self._pool[i]._nodes[1, 1]
            continue

               # print(cur_candidate.A)

    def mutate2(self, mutation_type):
        # create empty cell for probability
        probs = []
        for individual in self._pool:
            probs.append(individual._probability)  # append = add to the end

        # pick a mutation candidate
        # todo: co znamena ta nula v dalsim radku?
        mutation_candidate = np.random.choice(self._pool, 1, p=probs)[0]
        possible_coefficients = [0.9, 0.9, 0.9, 1.1, 1.2, 0.8, 0.75, 1.3, 1.2, 1.1]
        coef = np.random.choice(possible_coefficients, 1)

        if mutation_type == "x":
            mutation_candidate._nodes[0, 2] = mutation_candidate._nodes[0, 2] * coef
        if mutation_type == "y":
            mutation_candidate._nodes[1, 2] = mutation_candidate._nodes[1, 2] * coef
        if mutation_type == "a":
            # TODO: mutovat kazdou osu zvlast, cim vetsi napeti v ose, tim vetsi prurez
            # TODO: dat maximalni a minimalni hodnoty A, x, y
            for i in range(self._popsize):
                cur_candidate = self._pool[i]
                se = np.argmin(self._pool[i]._stress)
                if cur_candidate.A[se] > 0.01:
                    continue
                cur_candidate.A[se] = cur_candidate.A[se] * coef

        # print(cur_candidate.A)

        for i in range(self._popsize):
            self._pool[i]._nodes[0, 3] = self._pool[i]._nodes[0, 4] - self._pool[i]._nodes[0, 1]
            self._pool[i]._nodes[1, 3] = self._pool[i]._nodes[1, 1]
            continue

    def mutate_worst2(self):
        possible_coefficients = [0.9, 1.1, 1.2, 0.8, 0.75, 1.3, 1.2]
        # choose one from possible coof
        x_coefficient = np.random.choice(possible_coefficients, 1)
        y_coefficient = np.random.choice(possible_coefficients, 1)
        choice = np.random.randint(0, 3)
        # take a member and multiply it by a coef - change previous value for a new one
        # same as self._pool[choice]._nodes[0, 2] = self._pool[choice]._nodes[0, 2] * x_coefficient
        self._pool[choice]._nodes[0, 2] *= x_coefficient
        self._pool[choice]._nodes[1, 2] *= y_coefficient

        for i in range(self._popsize):
            self._pool[i]._nodes[0, 3] = self._pool[i]._nodes[0, 4] - self._pool[i]._nodes[0, 1]
            self._pool[i]._nodes[1, 3] = self._pool[i]._nodes[1, 1]
            continue


    def plot_stress(self):
        num_to_plot = 4

        gs = GridSpec(1, 4)
        gs.update(left=0.05, right=0.95, wspace=0.2)
        # fig, ax = plt.subplots(figsize=(10, 3), sharey='col')
        fig = plt.figure(figsize=(18,5))
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
            F_numnodex2 = plot_dict['F_numnodex2']
            dof_totx2 = plot_dict['dof_totx2']
            numnode = plot_dict['numnode']
            numelem = plot_dict['numelem']

            ax = fig.add_subplot(gs[0, index], aspect="equal")

            ax.grid(True)
            ax.set_xlim(-1, 11)    # CH
            ax.set_ylim(-2.5, 5)   # CH
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


                plt.setp(linenew, ls='-', c='c' if stress[r] > 0.000001 else ('red' if stress[r] < -0.000001 else 'black'),
                     lw=(1 + 20 * stress_normed[r]), label='strain' if stress[r] > 0 else 'stress')
                ax.plot()

            "Annotate outside forces"
            for r in range(numnode):
                plt.annotate(F_numnodex2[r],
                             xy=(xi[r], yi[r]), xycoords='data', xytext=np.sign(F_numnodex2[r]) * -35,
                             textcoords='offset pixels',
                             arrowprops=dict(facecolor='black', shrink=0, width=1.3, headwidth=5),
                             horizontalalignment='right', verticalalignment='bottom')

            "Annotate fixed DOFs"
            for r in range(numnode):
                if np.array_equal(dof_totx2[r], np.array([0, 1])):
                    plt.plot([xi[r]], [yi[r] - 0.2], 'o', c='k', markersize=8)
                if np.array_equal(dof_totx2[r], np.array([1, 0])):
                    plt.plot([xi[r] - 0.2], [yi[r]], 'o', c='k', markersize=8)
                if np.array_equal(dof_totx2[r], np.array([1, 1])):
                    plt.plot([xi[r]], [yi[r] - 0.2], '^', c='k', markersize=8)

        plt.savefig(datetime.datetime.now().strftime('stress_%Y%m%d_%H%M%S_') + ".png", DPI = 500)

        plt.show()

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
            F_numnodex2 = plot_dict['F_numnodex2']
            dof_totx2 = plot_dict['dof_totx2']
            numnode = plot_dict['numnode']
            numelem = plot_dict['numelem']

            ax = fig.add_subplot(gs[0, index], aspect="equal")

            ax.grid(True)
            ax.set_xlim(-1, 11)   # CH
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
                plt.annotate(F_numnodex2[r],
                             xy=(xi[r], yi[r]), xycoords='data', xytext=np.sign(F_numnodex2[r]) * -35,
                             textcoords='offset pixels',
                             arrowprops=dict(facecolor='black', shrink=0, width=1.5, headwidth=8),
                             horizontalalignment='right', verticalalignment='bottom')

            "Annotate fixed DOFs"
            for r in range(numnode):
                if np.array_equal(dof_totx2[r], np.array([0, 1])):
                    plt.plot([xi[r]], [yi[r] - 0.2], 'o', c='k', markersize=8)
                if np.array_equal(dof_totx2[r], np.array([1, 0])):
                    plt.plot([xi[r] - 0.2], [yi[r]], 'o', c='k', markersize=8)
                if np.array_equal(dof_totx2[r], np.array([1, 1])):
                    plt.plot([xi[r]], [yi[r] - 0.2], '^', c='k', markersize=8)

        # can use Textbox if needed
        # plt.subplots(1, 2,sharex=True, sharey=True)
        plt.savefig(datetime.datetime.now().strftime('cross section_%Y%m%d_%H%M%S_') + ".png", DPI = 500)

        plt.show()