"""
    @Author: Margarita Kuvaldina
    @https://github.com/margkuval
    @date: May 2018
"""

import fc_1x4_GA as GA
import plots_univ as plt_uni
import numpy as np

"""BRAIN FOR GENETIC ALGORITHM 1x4 TRUSS"""

"Initial values - population, number of iterations, mutation 1, mutation 2, iteration of plotting"
inp_task_1 = np.array([40, 50, 10, 8, 25])

population_1 = inp_task_1[0]  # population size
num_cycles_1 = inp_task_1[1]  # number of computation cycles
mut_x_1 = inp_task_1[2]
mut_yA_1 = inp_task_1[3]
plt_s_A_1 = inp_task_1[4]

inp_task_2 = inp_task_1  #np.array([20, 20, 10, 9, 25])

population_2 = inp_task_2[0]  # population size
num_cycles_2 = inp_task_2[1]  # number of computation cycles
mut_x_2 = inp_task_2[2]
mut_yA_2 = inp_task_2[3]
plt_s_A_2 = inp_task_2[4]

inp_task_3 = inp_task_1  #np.array([20, 20, 10, 9, 25])

population_3 = inp_task_3[0]  # population size
num_cycles_3 = inp_task_3[1]  # number of computation cycles
mut_x_3 = inp_task_3[2]
mut_yA_3 = inp_task_3[3]
plt_s_A_3 = inp_task_3[4]

"Task number 1"
task = GA.GA(population_1)

list_iter = []
list_fit = []
list_weight = []
list_stress = []
list_stress_positive = []
list_stress_negative = []
list_defl = []

task.initial()
print("New task 1")
for i in range(num_cycles_1):
    task.calculation()
    task.fitness()
    if i % plt_s_A_1 == 0:
        task.plot_stress()
        task.plot_A()
    task.crossover()
    if i % mut_x_1 == 0:
        task.mutation(mutation_type="x")
    if i % mut_yA_1== 0:
        task.mutation(mutation_type="y")
        task.mutation(mutation_type="a")

    list_iter.append(i)

    task.get_best_fit()
    task.get_best_weight()
    task.get_best_stress()
    task.get_best_stress_positive()
    task.get_best_stress_negative()
    task.get_best_defl()

    list_fit.append(task.get_best_fit())
    list_weight.append(task.get_best_weight())
    list_stress.append(task.get_best_stress())
    list_stress_positive.append(task.get_best_stress_positive())
    list_stress_negative.append(task.get_best_stress_negative())
    list_defl.append(task.get_best_defl())

plt_best_1 = plt_uni.plot_best_1(list_iter, list_fit,list_stress_positive, list_stress_negative, list_weight, list_defl)

"Task number 2"
task_2 = GA.GA(population_2)  # population size

list_iter_2 = []
list_fit_2 = []
list_weight_2 = []
list_stress_2 = []
list_stress_positive_2 = []
list_stress_negative_2 = []
list_defl_2 = []

task_2.initial()
print("New task 2")
for r in range(num_cycles_2):  # number of computation cycles
    task_2.calculation()
    task_2.fitness()
    if r % plt_s_A_2 == 0:
        task_2.plot_stress()
        task_2.plot_A()
    task_2.crossover()
    if r % mut_x_2 == 0:
        task_2.mutation(mutation_type="x")
    if r % mut_yA_2 == 0:
        task_2.mutation(mutation_type="y")
        task_2.mutation(mutation_type="a")

    list_iter_2.append(r)

    task_2.get_best_fit()
    task_2.get_best_weight()
    task_2.get_best_stress()
    task_2.get_best_stress_positive()
    task_2.get_best_stress_negative()
    task_2.get_best_defl()

    list_fit_2.append(task_2.get_best_fit())
    list_weight_2.append(task_2.get_best_weight())
    list_stress_2.append(task_2.get_best_stress())
    list_stress_positive_2.append(task_2.get_best_stress_positive())
    list_stress_negative_2.append(task_2.get_best_stress_negative())
    list_defl_2.append(task_2.get_best_defl())

plt_uni.plot_best_2(list_iter_2, list_fit_2, list_stress_positive_2, list_stress_negative_2, list_weight_2, list_defl_2)

"Task number 3"
task_3 = GA.GA(population_3)  # population size

list_iter_3 = []
list_fit_3 = []
list_weight_3 = []
list_stress_3 = []
list_stress_positive_3 = []
list_stress_negative_3 = []
list_defl_3 = []

task_3.initial()
print("New task 3")
for k in range(num_cycles_3):  # number of computation cycles
    task_3.calculation()
    task_3.fitness()
    if k % plt_s_A_3 == 0:
        task_3.plot_stress()
        task_3.plot_A()
    task_3.crossover()
    if k % mut_x_3 == 0:
        task_3.mutation(mutation_type="x")
    if k % mut_yA_3 == 0:
        task_3.mutation(mutation_type="y")
        task_3.mutation(mutation_type="a")

    list_iter_3.append(k)

    task_3.get_best_fit()
    task_3.get_best_weight()
    task_3.get_best_stress()
    task_3.get_best_stress_positive()
    task_3.get_best_stress_negative()
    task_3.get_best_defl()

    list_fit_3.append(task_3.get_best_fit())
    list_weight_3.append(task_3.get_best_weight())
    list_stress_3.append(task_3.get_best_stress())
    list_stress_positive_3.append(task_3.get_best_stress_positive())
    list_stress_negative_3.append(task_3.get_best_stress_negative())
    list_defl_3.append(task_3.get_best_defl())

plt_uni.plot_best_3(list_iter_3, list_fit_3, list_stress_positive_3, list_stress_negative_3, list_weight_3, list_defl_3)

plt_fits_3 = plt_uni.plot_fits_3(list_iter, list_iter_2, list_iter_3,
                list_fit, list_fit_2, list_fit_3,
                population_1, population_2, population_3,
                mut_x_1, mut_x_2, mut_x_3, mut_yA_1, mut_yA_2, mut_yA_3)