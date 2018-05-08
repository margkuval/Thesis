import tri_modif_GA as triGA

task = triGA.GA(6)  # num of different results
task.initial()
for i in range(700):  # num of cycles
    task.calc()
    if i % 100 == 0:
        task.plot()
    task.fitness()
    task.crossover()
    task.mutate(mutation_type="x")
    task.mutate(mutation_type="a")
    task.mutate(mutation_type="y")
    task.mutate_worst()
