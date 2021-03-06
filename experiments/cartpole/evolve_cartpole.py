# relative import from another directory
import os
import sys
p = os.path.abspath('../..')
sys.path.insert(1, p)
import population as pop
import experiments.cartpole.config as c
from visualize import draw_net
#from tqdm import tqdm
# setting Seeds
import torch
torch.manual_seed(c.Config.SEED)
import numpy as np
np.random.seed(c.Config.SEED)
import random
random.seed(c.Config.SEED)


neat = pop.Population(c.Config)
bests = neat.run()

solution = max(bests)

draw_net(solution, view=True, filename='./results/solution' , show_disabled=True)
        
# save solution
import pickle
# save best weights for future uses
with open("./results/cartpole-bests.plt","wb") as f:
    pickle.dump(bests, f)



