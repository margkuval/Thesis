import expo15_chile2D_GA_wcopy as trussGA

task = trussGA.GA(6)
task.initial()
for i in range(1):
    task.calc()
    task.fitness()
    task.crossover()
    task.plot()


    # pravdepodobnost - moc skace (udelal if - 9999). Mam zaridit, aby ty vahy nikdy nebyly zaporne
    # mutace A - na cem na zaviset mutace A