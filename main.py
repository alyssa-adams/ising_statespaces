# So I didn't have to start from scratch, I started with this person's code from here:
# https://github.com/rajeshrinet/compPhy/blob/master/notebooks/2014/IsingModel.ipynb

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
min_size = 4
max_size = 5
trials = 100

# make pickle_jar
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

"""
# ============= Step 2: Look at the networks and the trajectories ==============

# analyze the networks
def process_graphs():

    results = {}

    for size in range(min_size, max_size+1):

        results[size] = {}

        # load in these pickle files
        with open('pickle_jar/graphs_%d.p' % size, 'rb') as handle:

            networks = pickle.load(handle)

            # for each toplogy, get the fraction of nodes that have more than one out degree
            # also count the total number of bit flips for the whole trajectory
            for network in networks:

                # get the topology
                graph = networks[network][0]

                # change multidigraph to digraph (remove self-loops and parallel edges)
                graph = nx.DiGraph(graph)
                graph.remove_edges_from(nx.selfloop_edges(graph))

                # for each node, count out degrees
                out_degrees = [x[1] for x in list(graph.out_degree)]

                # get the fraction of nodes that are greater than 1 (1=1 of 0 out degree, 0= more than 1 out degree)
                out_degrees = [1 if x <= 1 else 0 for x in out_degrees]

                # get fraction
                deterministic_fraction = sum(out_degrees)/len(out_degrees)  # 0 is all deterministic

                # get the trajectory
                trajectory = networks[network][1]

                # count number of total bits flipped for the trajectory
                bits_flipped = 0
                trajectory = list(zip(trajectory, trajectory[1:]))
                for step in trajectory:
                    for i, bit in enumerate(step[0]):
                        if bit != step[1][i]:
                            bits_flipped += 1

                results[size][network] = (deterministic_fraction, bits_flipped)

    return results

results = process_graphs()


# ============= Step 3: Plot results ==============

# Figure 1: distribution of determinism over state spaces

fractions = []
for size in results:
    for state_space in results[size]:
        frac = results[size][state_space][0]
        fractions.append(frac)

fractions.sort()

plt.plot(fractions)
plt.show()  # TODO: each line is a different size


# Figure 1.5: distribution of bits flipped over state spaces

bits_flipped = []
for size in results:
    for state_space in results[size]:
        bits = results[size][state_space][1]
        bits_flipped.append(bits)

bits_flipped.sort()

plt.plot(bits_flipped)
plt.show()  # TODO: each line is a different size


# Figure 2: Determinism vs bits flipped

fractions = []
bits_flipped = []
for size in results:
    for state_space in results[size]:
        frac = results[size][state_space][0]
        fractions.append(frac)
        bits = results[size][state_space][1]
        bits_flipped.append(bits)

plt.scatter(fractions, bits_flipped)
plt.show()




# make plots
for size in range(min_size, max_size+1):

    # load in these pickle files
    with open('pickle_jar/graphs_%d.p' % size, 'rb') as handle:
        data = pickle.load(handle)

    # data[state space coord string] = [graph, list of states in trajectory]

    # binarize the trajectories: 0 = no change, 1 = change in state
    # These are the bit flips in each state space
    # (Need to get the energies for these too)

    ranked_statespaces = list(data.keys())
    ranked_statespaces = sorted(ranked_statespaces)
    ranked_statespaces.sort(key=len)

    bitflips = []

    for statespace in ranked_statespaces:

        state_traj = data[statespace][1]
        t0 = state_traj[:nsteps-1]
        t1 = state_traj[1:]
        xor_state_changes = [0 if step[0] == step[1] else 1 for step in list(zip(t0, t1))]
        bitflips.append(xor_state_changes)

    plt.figure(figsize=(50, 25))
    ax = sns.heatmap(bitflips, cbar=False)
    plt.show()
    # save this as figure 1, all bit flips in all state spaces in an ising model

"""
# ----------------------------------------------------------------------
#  This makes plots for all the different temperature points
# ----------------------------------------------------------------------

# nt = 32         #  number of discrete temperature points
# size = 6         #  size of the lattice, N x N
# eqSteps = 256       #  number of MC sweeps for equilibration
# mcSteps = 256       #  number of MC sweeps for calculation
#
# T = np.linspace(1.53, 3.28, nt)
# E, M, C, X = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
# n1, n2 = 1.0 / (mcSteps * size * size), 1.0 / (mcSteps * mcSteps * size * size)
# # divide by number of samples, and by system size to get intensive values


# # loop over all temp points
# for tt in range(nt):
#
#     E1 = M1 = E2 = M2 = 0
#
#     # initial config of the system
#     config = ising_model.initialstate(N)
#
#     # temp at this time step
#     iT = 1.0 / T[tt]
#
#     # move ising_model ahead by each time step
#     for i in range(eqSteps):  # equilibrate
#         config = ising_model.mcmove(config, iT, N)  # Monte Carlo moves
#
#     for i in range(mcSteps):
#         config = ising_model.mcmove(config, iT, N)
#         Ene = ising_model.calcEnergy(config, N)  # calculate the energy
#         Mag = ising_model.calcMag(config)  # calculate the magnetisation
#
#         E1 = E1 + Ene
#         M1 = M1 + Mag
#         M2 = M2 + Mag * Mag
#         E2 = E2 + Ene * Ene
#
#     E[tt] = n1 * E1
#     M[tt] = n1 * M1
#     C[tt] = (n1 * E2 - n2 * E1 * E1) * iT * iT
#     X[tt] = (n1 * M2 - n2 * M1 * M1) * iT
#
#
# f = plt.figure(figsize=(18, 10))  # plot the calculated values
#
# sp = f.add_subplot(2, 2, 1)
# plt.scatter(T, E, s=50, marker='o', color='IndianRed')
# plt.xlabel("Temperature (T)", fontsize=20)
# plt.ylabel("Energy ", fontsize=20)
# plt.axis('tight')
#
# sp = f.add_subplot(2, 2, 2)
# plt.scatter(T, abs(M), s=50, marker='o', color='RoyalBlue')
# plt.xlabel("Temperature (T)", fontsize=20)
# plt.ylabel("Magnetization ", fontsize=20)
# plt.axis('tight')
#
# sp = f.add_subplot(2, 2, 3)
# plt.scatter(T, C, s=50, marker='o', color='IndianRed')
# plt.xlabel("Temperature (T)", fontsize=20)
# plt.ylabel("Specific Heat ", fontsize=20)
# plt.axis('tight')
#
# sp = f.add_subplot(2, 2, 4)
# plt.scatter(T, X, s=50, marker='o', color='RoyalBlue')
# plt.xlabel("Temperature (T)", fontsize=20)
# plt.ylabel("Susceptibility", fontsize=20)
# plt.axis('tight')
#
# plt.show()
