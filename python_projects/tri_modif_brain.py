import tri_modif_GA as triGA

task = triGA.GA(50)  # num of different results
task.initial()
for i in range(20):  # num of cycles
    task.calc()

    task.fitness()
    if i % 10 == 0:
        task.plot()
    task.crossover()
    task.mutate(mutation_type="x")
    task.mutate(mutation_type="a")
    task.mutate(mutation_type="y")
    task.mutate_worst()
