import new1x4_GA as GA
import matplotlib.pyplot as plt
import numpy as np

task = GA.GA(20)  # population size
list_fit = []
list_weight = []

task.initial()
for i in range(30):  # number of computation cycles
    task.calculation()
    task.fitness()
    #if i % 20 == 0:
        #task.plot_stress()
        #task.plot_A()
    task.crossover()
    if i % 10 == 0:
        task.mutation(mutation_type="x")
    if i % 20 == 0:
        task.mutation(mutation_type="y")
        task.mutation(mutation_type="a")

    task.get_fit()
    list_fit.append((i, task.get_fit()))
    task.get_weight()
    list_weight.append((i, task.get_weight()))
    print(list_fit)


fig = plt.figure()
list_fit = np.array(list_fit).transpose()
x_fit = list_fit[[0]]
y_fit = list_fit[[1]]

ax1 = plt.subplot(121)
plt.scatter(x_fit, y_fit, c='k', alpha=0.5, edgecolors='grey',label='best fitness ever')
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.grid(True)
plt.legend()

list_weight = np.array(list_weight).transpose()
x_weight = list_weight[[0]]
y_weight = list_weight[[1]]

ax2 = plt.subplot(122, sharex = ax1)
plt.scatter(x_weight, y_weight, c='r', alpha=0.3, edgecolors='r',label='Weight evolution')
plt.xlabel('Iterations')
plt.ylabel('Construction weight')
plt.grid(True)
plt.legend()

plt.show()


"""def plot_fit(self):
    res = GA.GA.fitness(self)
    fitnesses = res

    max_fit = np.argmax(fitnesses)
    print(max_fit)


plot(x, y, ylim, cex.points = 0.7,
     col = c("green3", "dodgerblue3",  adjustcolor("green3", alpha.f = 0.1)),
     pch = c(16, 1), lty = c(1,2), legend = TRUE, grid = graphics:::grid, ...)"""