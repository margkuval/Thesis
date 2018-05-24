import new1x4_GA as GA
import matplotlib as plt

task = GA.GA(40)  # population size
task.initial()
for i in range(11):  # number of computation cycles
    task.calculation()
    task.fitness()
    if i % 20 == 0:
        task.plot_stress()
        task.plot_A()
    task.crossover()
    if i % 10 == 0:
        task.mutation(mutation_type="x")
    if i % 5 == 0:
        task.mutation(mutation_type="y")
        task.mutation(mutation_type="a")

print(GA.fitness)