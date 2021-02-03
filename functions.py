import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt



# ----------------------------------------------------------------------
##  BLOCK OF FUNCTIONS USED IN THE MAIN CODE
# ----------------------------------------------------------------------

class IsingModel():

    def initialstate(self, size):

        '''generates a random spin configuration for initial condition'''

        state = 2 * np.random.randint(2, size=(size, size)) - 1

        return state

    def mcmove(self, config, size, beta):

        '''Monte Carlo move using Metropolis algorithm '''

        # randomly hop through the cells, visit each one once
        randrows = list(range(size))
        np.random.shuffle(randrows)
        randcols = list(range(size))
        np.random.shuffle(randrows)

        for m in randrows:
            for n in randcols:

                # get cell location
                s = config[m, n]

                # calculate the energy cost of trying to flip it
                nb = config[(m + 1) % size, n] + config[m, (n + 1) % size] + config[(m - 1) % size, n] + config[m, (n - 1) % size]
                cost = 2 * s * nb

                # if cost is less than 0, keep the change
                if cost < 0:
                    s *= -1

                # otherwise accept the move with prob e^(-dE/T)
                elif rand() < np.exp(-cost * beta):
                    s *= -1

                config[m, n] = s

        return config

    def calcEnergy(self, config, size):

        '''Energy of a given configuration'''

        energy = 0
        for i in range(len(config)):
            for j in range(len(config)):
                S = config[i, j]
                nb = config[(i + 1) % size, j] + config[i, (j + 1) % size] + config[(i - 1) % size, j] + config[i, (j - 1) % size]
                energy += -nb * S

        return energy / 4.

    def calcMag(self, config):

        '''Magnetization of a given configuration'''

        mag = np.sum(config)

        return mag

    def simulate(self, size, temp, nsteps, show_plots):

        '''
        Goes through each time step of the 2D ising model
        :param size: m of a mxn square matrix (m=n)
        :param temp: temperature (float)
        :param nsteps: number of time steps to evolve the system
        :param show_plots: True of False
        :return: Dict of all configs of all possible state spaces at each time step
        '''

        # save all the state space evolutions to a dict
        state_space_evolution = {}

        # initial state
        config = self.initialstate(size)

        # ---- break this down into all possible subsets of cells

        # make dict of cell coordinates to enumerate the possible cell subsets
        ncells = size ** 2
        cells = list(range(ncells))
        coords = []
        for i in range(size):
            for j in range(size):
                coords.append((i, j))
        coords = dict(zip(cells, coords))

        # define all my possible substates here, enumerated by all possible subsets of cells
        substates = [[]]
        for cell in cells:
            orig = substates[:]
            new = cells[cell]
            for j in range(len(substates)):
                substates[j] = substates[j] + [new]
            substates = orig + substates
        substates = substates[1:]

        if show_plots:
            self.makeplot(config, i=0)

        # update the config at each time step
        for i in range(nsteps):

            config = self.mcmove(config, size, 1.0/temp)

            # for each substate, map the whole state onto it
            for statespace in substates:

                # save each time step to a dict
                # check to see if this subdict exists already
                if str(statespace) not in state_space_evolution:
                    state_space_evolution[str(statespace)] = {}

                # get coords of each cell in the state space to build the state of this subspace
                state = []
                for cell in statespace:
                    m, n = coords[cell]
                    cell_state = config[m][n]
                    state.append(cell_state)

                state_space_evolution[str(statespace)][i] = state

            if show_plots:

                # only plot the configuration at particular time points
                if i == 1:
                    self.makeplot(config, i)
                if i == 4:
                    self.makeplot(config, i)
                if i == 32:
                    self.makeplot(config, i)
                if i == 100:
                    self.makeplot(config, i)
                if i == 1000:
                    self.makeplot(config, i)

        return state_space_evolution

    def makeplot(self, config, i):

        ''' This modules plts the configuration once passed to it along with time etc '''

        f = plt.figure(figsize=(15, 15), dpi=80)
        
        plt.imshow(config, cmap=plt.cm.get_cmap('RdYlBu'))
        plt.title('Time=%d' % i)
        plt.axis('tight')

        plt.show()
        plt.clf()
