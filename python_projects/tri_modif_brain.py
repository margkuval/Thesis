import tri_modif_GA as triGA

task = triGA.GA(60)  # num of different results
task.initial()
for i in range(100):  # num of cycles
    task.calc()
    task.fitness()
    task.crossover()
    task.mutate()
    task.plot()