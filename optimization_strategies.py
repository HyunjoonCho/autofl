import math
import operator
import random
import itertools

import numpy as np
from sklearn.linear_model import LinearRegression
from deap import base, creator, tools

def grid_search(evaluator, size):
    max_weight = size # okay, four models for now
    granularity = 10
    step = max_weight / granularity 

    best_weight = []
    best_accs = []
    for combination in itertools.product(range(granularity + 1), repeat=size):
        if sum(combination) == granularity:
            weight = [x * step for x in combination]
            accs = evaluator(weight)
            if accs > best_accs:
                best_accs = accs
                best_weight = weight

    print(f'{best_weight} achieved accuracies of {best_accs}')
    return best_weight

def linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return list(model.coef_)

def create_stats_and_logbook():
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    return stats, logbook

def ga(evaluator, size):
    POPSIZE, NUMGEN, CXPB, MUTPB = 20, 20, 0.5, 0.3

    creator.create("WeightedFitness", base.Fitness, weights=(1.0, 0.1, 0.01))
    creator.create("Individual", list, fitness=creator.WeightedFitness) # or np.ndarray?

    toolbox = base.Toolbox()
    toolbox.register("SingleWeight", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.SingleWeight, size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluator)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=POPSIZE)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    stats, logbook = create_stats_and_logbook()

    best = None

    for g in range(NUMGEN):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        pop[:] = offspring
        best = tools.selBest(pop, 1)[0] 
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        print(logbook.stream)

    log = f'Population: {POPSIZE}\tGenerations: {NUMGEN}\tCrossover Prob: {CXPB}\tMutation Prob: {MUTPB}\n' + str(logbook)
    return best, log

def pso(evaluator, size):
    POPSIZE, NUMGEN = 10, 100

    creator.create("WeightedFitness", base.Fitness, weights=(1, 0.1, 0.01))
    creator.create("Particle", list, fitness=creator.WeightedFitness, speed=list, smin=None, smax=None, best=None)

    def generate(size, pmin, pmax, smin, smax):
        part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size)) 
        part.speed = [random.uniform(smin, smax) for _ in range(size)]
        part.smin = smin
        part.smax = smax
        return part

    def updateParticle(part, best, phi1, phi2):
        u1 = (random.uniform(0, phi1) for _ in range(len(part)))
        u2 = (random.uniform(0, phi2) for _ in range(len(part)))
        v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
        v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
        part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
        for i, speed in enumerate(part.speed):
            if abs(speed) < part.smin:
                part.speed[i] = math.copysign(part.smin, speed)
            elif abs(speed) > part.smax:
                part.speed[i] = math.copysign(part.smax, speed)
        part[:] = list(map(operator.add, part, part.speed))

    toolbox = base.Toolbox()
    toolbox.register("particle", generate, size=size, pmin=0, pmax=1, smin=-0.5, smax=0.5)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("update", updateParticle, phi1=0.3, phi2=0.5)
    toolbox.register("evaluate", evaluator)

    pop = toolbox.population(n=POPSIZE)
    stats, logbook = create_stats_and_logbook()
    best = None

    for g in range(NUMGEN):
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)

        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        print(logbook.stream)

    log = f'Particles: {POPSIZE}\tGenerations: {NUMGEN}\n' + str(logbook)
    return best, log

def de(evaluator, size):
    POPSIZE, NUMGEN, CXPB, DW = 30, 30, 0.9, 0.8

    creator.create("WeightedFitness", base.Fitness, weights=(1.0, 0.1, 0.01))
    creator.create("Agent", list, fitness=creator.WeightedFitness)

    toolbox = base.Toolbox()
    toolbox.register("SingleWeight", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Agent, toolbox.SingleWeight, size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluator)

    def updateAgent(agent, pop, size):
        original_agent = toolbox.clone(agent)
        a, b, c = random.sample(pop, 3)
        R = random.randint(1, size) - 1
        for i in range(size):
            if i == R or random.random() < CXPB:
                agent[i] = a[i] + DW * (b[i] - c[i])
        new_fitness = toolbox.evaluate(agent)
        if list(agent.fitness.values) > new_fitness:
            agent[:] = original_agent[:]
        else:
            agent.fitness.values = new_fitness

    toolbox.register("update", updateAgent, size=size)

    pop = toolbox.population(n=POPSIZE)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    stats, logbook = create_stats_and_logbook()
    best = None

    for g in range(NUMGEN):
        for agent in pop:
            toolbox.update(agent, pop)
        pop = sorted(pop, key=lambda ind: ind.fitness.values, reverse=True)
        best = tools.selBest(pop, 1)[0] 
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        print(logbook.stream)

    log = f'Population: {POPSIZE}\tGenerations: {NUMGEN}\tCrossover Prob: {CXPB}\tDifferential Weight: {DW}\n' + str(logbook)
    return best, log