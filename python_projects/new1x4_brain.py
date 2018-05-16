import new1x4_GA as GA
import matplotlib as plt

task = GA.GA(43)  # num of different results
task.initial()
for i in range(58):  # num of cycles
    task.calc()
    task.fitness()
    if i % 10 == 0:
        task.plot_stress()
        task.plot_A()
    task.crossover1()
    task.crossover2()
    if i % 12 == 0: #todo: start from 10th? (always starts at 0)
        task.mutate1(mutation_type="x")
        task.mutate2(mutation_type="y")
    if i % 14 == 0:
        task.mutate1(mutation_type="y")
        task.mutate2(mutation_type="x")
        task.mutate2(mutation_type="a")
        task.mutate_worst2()
