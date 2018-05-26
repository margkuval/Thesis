import fc_tri_truss_2x2_GA as GA
import plots_univ as plt_uni
import matplotlib.pyplot as plt

population_1 = 30  # population size
num_cycles_1 = 20 # number of computation cycles
mut_x_1 = 8
mut_yA_1 = 7
plt_s_A_1 = 20

population_2 = 30
num_cycles_2 = 20
mut_x_2 = 8
mut_yA_2 = 7
plt_s_A_2 = 20

population_3 = 30
num_cycles_3 = 20
mut_x_3 = 8
mut_yA_3 = 7
plt_s_A_3 = 20

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



"""plot(x, y, ylim, cex.points = 0.7,
     col = c("green3", "dodgerblue3",  adjustcolor("green3", alpha.f = 0.1)),
     pch = c(16, 1), lty = c(1,2), legend = TRUE, grid = graphics:::grid, ...)"""