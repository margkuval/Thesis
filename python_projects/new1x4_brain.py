import new1x4_GA as GA

task = GA.GA(100)  # num of different results
task.initial()
for i in range(201):  # num of cycles
    task.calc()
    task.fitness()
    if i % 100 == 0:
        task.plot()
    task.crossover()
    if i % 30 == 0:
        task.mutate(mutation_type="x")
        task.mutate(mutation_type="a")
    if i % 40 == 0:
        task.mutate(mutation_type="y")
        task.mutate_worst()
