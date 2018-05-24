import new1x4_GA as GA
import matplotlib.pyplot as plt
import numpy as np

task = GA.GA(20)  # population size
iter_and_fit = []

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
    iter_and_fit.append((i, task.get_fit()))
    print(iter_and_fit)


fig = plt.figure()
iter_and_fit = np.array(iter_and_fit).transpose()

x = iter_and_fit[[0]]
print(x)
y = iter_and_fit[[1]]
plt.scatter(x, y, c='k', alpha=0.3, edgecolors='grey',label='best fitness ever')
plt.xlabel('Iterations')
plt.ylabel('Fitness')
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