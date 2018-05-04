import tri_modif_GA as triGA

task = triGA.GA(6)  # num of different results
task.initial()
for i in range(1):  # num of cycles
    task.calc()
    task.fitness()
    task.crossover()
    task.plot()