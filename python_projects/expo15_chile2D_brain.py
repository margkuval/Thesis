import expo15_chile2D_GA_wcopy as trussGA

task = trussGA.GA(6)
task.initial()
for i in range(1):
    task.calc()
    task.fitness()
    task.crossover()
    task.plot()