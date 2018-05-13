import new1x4_GA as GA

task = GA.GA(100)  # num of different results
task.initial()
for i in range(200):  # num of cycles
    task.calc()
    task.fitness()
    if i % 100 == 0:
        task.plot()
    task.crossover()
    task.mutate(mutation_type="x")
    task.mutate(mutation_type="a")
    task.mutate(mutation_type="y")
    task.mutate_worst()
