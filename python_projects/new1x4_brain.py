import new1x4_GA as GA
import matplotlib as plt
import numpy as np

task = GA.GA(20)  # population size
list = []

task.initial()
for i in range(92):  # number of computation cycles
    task.calculation()
    task.fitness()
    if i % 20 == 0:
        task.plot_stress()
        task.plot_A()
    task.crossover()
    if i % 190 == 0:
        task.mutation(mutation_type="x")

    if i % 590 == 0:
        task.mutation(mutation_type="y")
        task.mutation(mutation_type="a")

    task.get_fit()
    list.append((i, task.get_fit()))
    print(list)


"""def plot_fit(self):
    res = GA.GA.fitness(self)
    fitnesses = res

    max_fit = np.argmax(fitnesses)
    print(max_fit)


plot(x, y, ylim, cex.points = 0.7,
     col = c("green3", "dodgerblue3",  adjustcolor("green3", alpha.f = 0.1)),
     pch = c(16, 1), lty = c(1,2), legend = TRUE, grid = graphics:::grid, ...)"""