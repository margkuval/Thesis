import expo15_chile2D_GA_wcopy as trussGA

task = trussGA.GA(10)
task.initial()
for i in range(3):
    task.calc()
    task.fitness()
    task.crossover()
    task.plot()