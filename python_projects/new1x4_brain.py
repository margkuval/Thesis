import new1x4_GA as GA
import matplotlib as plt

task = GA.GA(100)  # population size
task.initial()
for i in range(810):  # number of computation cycles
    task.calculation()
    task.fitness()
    if i % 200 == 0:
        task.plot_stress()
        task.plot_A()
    task.crossover()
    if i % 100 == 0:
        task.mutation(mutation_type="x")
    if i % 50 == 0:
        task.mutation(mutation_type="y")
        task.mutation(mutation_type="a")