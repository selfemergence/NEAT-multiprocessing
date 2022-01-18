import logging

import sys
import os
p = os.path.abspath('../..')
sys.path.insert(1, p)
import population as pop
import experiments.cartpole.config as c
from visualize import draw_net
from tqdm import tqdm


num_of_solutions = 0

avg_num_hidden_nodes = 0
min_hidden_nodes = 10
max_hidden_nodes = 0
found_minimal_solution = 0

avg_num_generations = 0
min_num_generations = 100

neat = pop.Population(c.Config)
#solution, generation = neat.run()
        
# save solution
import pickle
# save best weights for future uses
file = open("./results/cartpole-evolve.plt",'rb')
best_genome = pickle.load(file)
print("Expected Fitness ", best_genome.fitness)
file.close()

config = c.Config
fitness = config.fitness_fn(config, genome=best_genome, render_test=True)
print("test fitness ", fitness)


