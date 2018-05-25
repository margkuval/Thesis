import fc_1x4_GA as GA
import plots_univ as plt_uni
import matplotlib.pyplot as plt

population_1 = 30
num_cycles_1 = 10
mut_1_1 = 10
mut_1_2 = 20

population_2 = 15
num_cycles_2 = 10
mut_2_1 = 10
mut_2_2 = 20

population_3 = 15
num_cycles_3 = 10
mut_3_1 = 10
mut_3_2 = 20

"Task number 1"
task = GA.GA(population_1)  # population size

list_iter = []
list_fit = []
list_weight = []
list_stress = []
list_defl = []

task.initial()
for i in range(num_cycles_1):  # number of computation cycles
    task.calculation()
    task.fitness()
    # if i % 20 == 0:
    # task.plot_stress()
    # task.plot_A()
    task.crossover()
    if i % mut_1_1 == 0:
        task.mutation(mutation_type="x")
    if i % mut_1_2 == 0:
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


"Task number 2"
task_2 = GA.GA(population_2)  # population size

list_iter_2 = []
list_fit_2 = []
list_weight_2 = []
list_stress_2 = []
list_defl_2 = []

task_2.initial()
for r in range(num_cycles_2):  # number of computation cycles
    task_2.calculation()
    task_2.fitness()
    # if i % 20 == 0:
    # task_2.plot_stress()
    # task_2.plot_A()
    task_2.crossover()
    if i % mut_2_1 == 0:
        task.mutation(mutation_type="x")
    if i % mut_2_2 == 0:
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

#plt_best = plt_uni.plot_best(list_iter, list_fit, list_stress, list_weight, list_defl)

"Task number 3"
task_3 = GA.GA(population_3)  # population size

list_iter_3 = []
list_fit_3 = []
list_weight_3 = []
list_stress_3 = []
list_defl_3 = []

task_3.initial()
for r in range(num_cycles_3):  # number of computation cycles
    task_3.calculation()
    task_3.fitness()
    # if i % 20 == 0:
    # task_3.plot_stress()
    # task_3.plot_A()
    task_3.crossover()
    if i % mut_3_1 == 0:
        task.mutation(mutation_type="x")
    if i % mut_3_2 == 0:
        task.mutation(mutation_type="y")
        task.mutation(mutation_type="a")

    list_iter_3.append(r)

    task_3.get_best_fit()
    task_3.get_best_weight()
    task_3.get_best_stress()
    task_3.get_best_defl()

    list_fit_3.append(task_3.get_best_fit())
    list_weight_3.append(task_3.get_best_weight())
    list_stress_3.append(task_3.get_best_stress())
    list_defl_3.append(task_3.get_best_defl())

#plt_best = plt_uni.plot_best(list_iter, list_fit, list_stress, list_weight, list_defl)
plt_fits_3 = plt_uni.plot_fits_3(list_iter, list_iter_2, list_iter_3,
                                 list_fit, list_fit_2, list_fit_3,
                                 population_1, population_2, population_3)

plt.show()


"""plot(x, y, ylim, cex.points = 0.7,
     col = c("green3", "dodgerblue3",  adjustcolor("green3", alpha.f = 0.1)),
     pch = c(16, 1), lty = c(1,2), legend = TRUE, grid = graphics:::grid, ...)"""