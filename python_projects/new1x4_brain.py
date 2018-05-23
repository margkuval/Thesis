import new1x4_GA as GA
import matplotlib as plt

task = GA.GA(13)  # population size
task.initial()
for i in range(80):  # num of computation cycles
    task.calc()
    task.fitness()
    if i % 20 == 0:
        task.plot_stress()
        task.plot_A()
    task.crossover()
    if i % 10 == 0:
        task.mutation(mutation_type="y")
    if i % 4 == 0:
        task.mutation(mutation_type="x")
        task.mutation(mutation_type="a")