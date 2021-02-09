# So I didn't have to start from scratch, I started with this person's code from here:
# https://github.com/rajeshrinet/compPhy/blob/master/notebooks/2014/IsingModel.ipynb

import numpy as np
import os
import networkx as nx
from itertools import repeat
import pickle
from multiprocessing import Pool, freeze_support
# --
from functions import IsingModel


# ============= Step 0: define the ising model parameters ==============

temps = list(np.linspace(2, 3, num=8))
nsteps = 1000
min_size = 3
max_size = 3
trials = 100

# make pickle_jar
if not os.path.isdir('pickle_jar'):
    os.mkdir('pickle_jar')


# ============= Step 1: run the ising model at some temperature for various sizes ==============

def run_statespaces(temp, nsteps):

    """
    Runs the Ising model and saves the trajectory for each state space, and also the resulting state space topology.
    Pickles the result to pickle_jar
    :param temp: temperature for the ising model
    :param nsteps: number of steps to run it
    :return: Nothing, just saves to file
    """

    for size in range(min_size, max_size+1):

        # initiate the class
        ising_model = IsingModel()

        print('size: ' + str(size))

        for trial in range(trials):

            print(trial)

            # run the model and get the state space evolutions for all state spaces
            data = ising_model.simulate(size, temp, nsteps, show_plots=False)

            # for each state trajectory, make graph of transitions between states
            graphs = {}

            for statespace in data:

                trajectory = data[statespace]

                # get the list of edges
                edges = []
                for e in range(nsteps-1):
                    edge = (str(trajectory[e]), str(trajectory[e+1]))
                    edges.append(edge)

                # turn into a graph
                g = nx.MultiDiGraph()
                g.add_edges_from(edges)

                # save to dict
                # graphs[state space coord string] = [graph, list of states in trajectory]
                graphs[statespace] = [g, list(trajectory.values())]

            # save to pickle file
            with open('pickle_jar/graphs_temp_%0.2f_size_%d_trial_%d.p' % (temp, size, trial), 'wb') as handle:
                pickle.dump(graphs, handle)


if __name__ == '__main__':

    freeze_support()

    # parallelize to make this faster
    with Pool() as pool:
        pool.starmap(run_statespaces, zip(temps, repeat(nsteps)))

    quit()
