import fc_1x4_GA as GA
import plots_univ as plt_uni
import matplotlib.pyplot as plt
import numpy as np
import datetime

task = GA.GA(50)  # population size

list_iter = []
list_fit = []
list_weight = []
list_stress = []
list_defl = []

task.initial()
for i in range(10):  # number of computation cycles
    task.calculation()
    task.fitness()
    # if i % 20 == 0:
    # task.plot_stress()
    # task.plot_A()
    task.crossover()
    if i % 10 == 0:
        task.mutation(mutation_type="x")
    if i % 20 == 0:
        task.mutation(mutation_type="y")
        task.mutation(mutation_type="a")

    list_iter.append(i)

    task.get_best_fit()
    task.get_best_weight()
    task.get_best_stress()
    task.get_best_defl()

    list_fit.append(task.get_best_fit())
    list_weight.append(task.get_best_weight())
    list_stress.append(task.get_best_stress())
    list_defl.append(task.get_best_defl())

plt_best = plt_uni.plot_best(list_iter, list_fit, list_stress, list_weight, list_defl)

plt_mean = plt_uni.plot_best(list_iter, list_fit, list_stress, list_weight, list_defl)


task_2 = GA.GA(25)  # population size

list_iter_2 = []
list_fit_2 = []
list_weight_2 = []
list_stress_2 = []
list_defl_2 = []

task_2.initial()
for r in range(15):  # number of computation cycles
    task_2.calculation()
    task_2.fitness()
    # if i % 20 == 0:
    # task_2.plot_stress()
    # task_2.plot_A()
    task_2.crossover()
    if i % 10 == 0:
        task.mutation(mutation_type="x")
    if i % 20 == 0:
        task.mutation(mutation_type="y")
        task.mutation(mutation_type="a")

    list_iter_2.append(r)

    task_2.get_best_fit()
    task_2.get_best_weight()
    task_2.get_best_stress()
    task_2.get_best_defl()

    list_fit_2.append(task_2.get_best_fit())
    list_weight_2.append(task_2.get_best_weight())
    list_stress_2.append(task_2.get_best_stress())
    list_defl_2.append(task_2.get_best_defl())


#plt_fit1_2 = plt_uni.plot_fits(list_iter, list_iter_2, list_fit, list_fit_2)
plt_fits_2 = plt_uni.plot_fits_2(list_iter, list_iter_2, list_fit, list_fit_2)

plt.show()


"""plot(x, y, ylim, cex.points = 0.7,
     col = c("green3", "dodgerblue3",  adjustcolor("green3", alpha.f = 0.1)),
     pch = c(16, 1), lty = c(1,2), legend = TRUE, grid = graphics:::grid, ...)"""