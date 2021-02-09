# So I didn't have to start from scratch, I started with this person's code from here:
# https://github.com/rajeshrinet/compPhy/blob/master/notebooks/2014/IsingModel.ipynb

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, time, re
import networkx as nx
import pandas as pd
import pickle
from multiprocessing import Pool, freeze_support
# --
from functions import IsingModel


# ============= Step 0: define the ising model parameters ==============

temps = list(np.linspace(2, 3, num=8))[:-1]
nsteps = 1000
min_size = 3
max_size = 3
trials = 100


# ============= Step 1: Load all the pickle files into a single df to plot ==============

# analyze the networks
def make_plots_df(temp):

    """
    Turns all the pickle files into a single df to plot in seaborn
    :param temp:
    :return: nothing, save to pickle files
    """

    for size in range(min_size, max_size+1):

        for trial in range(trials):

            print(trial)

            # load in these pickle files
            with open('pickle_jar/results_temp_%0.2f_size_%d_trial_%d.p' % (temp, size, trial), 'rb') as handle:

                results = pickle.load(handle)
                df = pd.DataFrame.from_dict(results, orient='index')
                df = df.reset_index()

                for index, row in df.iterrows():

                    # For the df over time, turn into a seaborn-friendly format
                    # Columns: ss, t, bit flips, det bit flips, etc
                    results_dict_for_df = {}  # faster to make final df

                    ss = row['index']

                    bit_flip_traj = row['bit_flip_traj']
                    det_bit_flip_traj = row['det_bit_flip_traj']
                    bw_ratio_traj = row['bw_ratio_traj']

                    # keep the rest of the values constant
                    deterministic_fraction = row['deterministic_fraction']
                    bits_flipped = row['bits_flipped']
                    steps = len(bit_flip_traj)

                    for step in range(steps):

                        # spread these out one value per column
                        bit_flips = bit_flip_traj[step]
                        det_bit_flips = det_bit_flip_traj[step]
                        bw_ratio = bw_ratio_traj[step]

                        # save to the final df
                        newrow = [size, temp, trial, ss, deterministic_fraction, bits_flipped, step, bw_ratio, bit_flips, det_bit_flips]
                        results_dict_for_df[step] = newrow

                    # convert from dict to df
                    results_df = pd.DataFrame.from_dict(results_dict_for_df, orient='index',
                            columns=['size', 'temp', 'trial', 'ss', 'deterministic_fraction', 'bits_flipped', 'step',
                                 'bw_ratio', 'bit_flips', 'det_bit_flips'])

                    # save to pickle file
                    with open('pickle_jar/df_temp_%0.2f_size_%d_trial_%d_index_%d.p' % (temp, size, trial, index), 'wb') as handle:
                        pickle.dump(results_df, handle)


if __name__ == '__main__':

    # parallelize to make this faster
    #freeze_support()
    #with Pool() as pool:
    #    pool.map(make_plots_df, temps)

    #quit()

    # ================ Make df of whole statespaces only ================
    """
    # since it has to check many files, parallelize loading them in and getting the results into a different file

    def get_whole_ss_data(file):

        with open(os.path.join('pickle_jar', file), 'rb') as handle:

            print(file)

            results = pickle.load(handle)
            results = results.to_dict('records')
            results = list(filter(lambda x: len(eval(x['ss']))==9, results))  # TODO: Change all 9s to size*size

            # if it found whole statespaces in this file
            if len(results) > 0:

                list_of_rows_for_df = []

                for result in results:
                    ss_size = 9
                    n_1s = result['bw_ratio']*ss_size
                    n_neg1s = ss_size-n_1s
                    result['Magnetization'] = abs((n_1s + -1*n_neg1s)/ss_size)
                    result['Bit Flips (normalized)'] = result['bit_flips']/ss_size
                    result['temp'] = round(result['temp'], 2)

                    list_of_rows_for_df.append(result)

                # save to pickle file
                with open('pickle_jar/wholess_' + file, 'wb') as handle:
                    pickle.dump(list_of_rows_for_df, handle)

    def avgs_whole_ss(file):

        with open(os.path.join('pickle_jar', file), 'rb') as handle:

            print(file)

            results = pickle.load(handle)
            whole_results = results['[0, 1, 2, 3, 4, 5, 6, 7, 8]']
            ss_size = 9

            # get the avg mag
            bw_ratio_traj = whole_results['bw_ratio_traj']
            mags = []
            for ratio in bw_ratio_traj:
                n_1s = ratio*ss_size
                n_neg1s = ss_size - n_1s
                mag = abs(n_1s + -1*n_neg1s)/ss_size
                mags.append(mag)
            avg_mag = sum(mags)/len(mags)

            # get the avg number of bit flips over time (normalize by size)
            bit_flip_traj = whole_results['bit_flip_traj']
            bit_flip_traj = list(map(lambda x: x/ss_size, bit_flip_traj))
            avg_bit_flips = sum(bit_flip_traj)/len(bit_flip_traj)

            # get the avg number of det bit flips over time
            det_bit_flip_traj = whole_results['det_bit_flip_traj']
            avg_det_bit_flips = sum(det_bit_flip_traj)/len(det_bit_flip_traj)

            # get temp
            temp = float(file.split('_')[2])

            # save dict
            whole_results['Average Magnetization'] = avg_mag
            whole_results['Average N Bit Flips'] = avg_bit_flips
            whole_results['Average N Det Bit Flips'] = avg_det_bit_flips
            whole_results['Temperature'] = temp

            # should not take long, so just return to local result
            return whole_results

    # list pickle files to process
    files = os.listdir('pickle_jar')
    files = list(filter(lambda x: re.search('results', x), files))

    # shouldn't need to parallelize this
    results_df = [avgs_whole_ss(file) for file in files]

    # final df to plot
    results_df = pd.DataFrame(results_df)
    results_df = results_df.rename(columns={"deterministic_fraction": "Deterministic Fraction", "bits_flipped": "Total Bits Flipped"})
    """

    # ================ Make df of all statespaces ================

    # since it has to check many files, parallelize loading them in and getting the results into a different file

    def avgs_all_ss(file):

        with open(os.path.join('pickle_jar', file), 'rb') as handle:

            print(file)

            results = pickle.load(handle)
            ss_names = results.keys()

            results_processed = []

            for ss in ss_names:

                ss_size = len(eval(ss))

                # get the avg mag
                bw_ratio_traj = results[ss]['bw_ratio_traj']
                mags = []
                for ratio in bw_ratio_traj:
                    n_1s = ratio*ss_size
                    n_neg1s = ss_size - n_1s
                    mag = abs(n_1s + -1*n_neg1s)/ss_size
                    mags.append(mag)
                avg_mag = sum(mags)/len(mags)

                # get the avg number of bit flips over time (normalize by size)
                bit_flip_traj = results[ss]['bit_flip_traj']
                bit_flip_traj = list(map(lambda x: x/ss_size, bit_flip_traj))
                avg_bit_flips = sum(bit_flip_traj)/len(bit_flip_traj)

                # get the avg number of det bit flips over time
                det_bit_flip_traj = results[ss]['det_bit_flip_traj']
                avg_det_bit_flips = sum(det_bit_flip_traj)/len(det_bit_flip_traj)

                # get temp
                temp = float(file.split('_')[2])

                for_results = results[ss]

                # save dict
                for_results['Average Magnetization'] = avg_mag
                for_results['Average N Bit Flips'] = avg_bit_flips
                for_results['Average N Det Bit Flips'] = avg_det_bit_flips
                for_results['Temperature'] = temp
                for_results['State Space Size'] = ss_size

                results_processed.append(for_results)

            # save to pickle file
            with open('pickle_jar/all_ss_avgs_' + file, 'wb') as handle:
                pickle.dump(results_processed, handle)

    # list pickle files to process
    files = os.listdir('pickle_jar')
    files = list(filter(lambda x: re.search('results', x), files))

    # parallelize to make this faster
    #freeze_support()
    #with Pool() as pool:
    #    pool.map(avgs_all_ss, files)

    #quit()

    # load and concat into a final df to plot
    files = os.listdir('pickle_jar')
    files = list(filter(lambda x: re.search('all_ss_avgs', x), files))

    results_df = []

    for i, file in enumerate(files):
        with open(os.path.join('pickle_jar', file), 'rb') as handle:
            results = pickle.load(handle)
            results_df.extend(results)
            print(i)

    results_df = pd.DataFrame(results_df)
    results_df = results_df.rename(columns={"deterministic_fraction": "Deterministic Fraction", "bits_flipped": "Total Bits Flipped"})

    # ================ Make df to see values change over time ================
    """
    # Now load in all these pickle files into one large df
    # list pickle files to load
    files = os.listdir('pickle_jar')
    files = list(filter(lambda x: re.search('df_temp', x), files))[:10000]

    list_of_dfs = []

    for i, file in enumerate(files):
        with open(os.path.join('pickle_jar', file), 'rb') as handle:
            results = pickle.load(handle)
            list_of_dfs.append(results)
            print(i)

    # final df to plot
    results_df = pd.concat(list_of_dfs)
    results_df = results_df.reset_index()
    """

    # ================ Make df for summary plots ================
    """
    # df for some of the plots (faster)
    files = os.listdir('pickle_jar')
    files = list(filter(lambda x: re.search('results_temp', x), files))

    summary_plot_data = []  # list of dicts

    for i, file in enumerate(files):
        with open(os.path.join('pickle_jar', file), 'rb') as handle:

            results = pickle.load(handle)
            temp = float(file.split('_')[2])
            trial = int(file.split('_')[6].split('.')[0])

            detfrac_dist = []
            bitsflipped_dist = []

            for ss in results:
                detfrac_dist.append((results[ss]['deterministic_fraction']))
                bitsflipped_dist.append(results[ss]['bits_flipped'])

            detfrac_dist.sort(reverse=True)
            bitsflipped_dist.sort(reverse=True)

            for x_value in range(len(detfrac_dist)):

                row = {'x_value': x_value,
                 'temp': temp,
                 'trial': trial,
                 'detfrac': detfrac_dist[x_value],
                 'bitsflipped': bitsflipped_dist[x_value]}

                summary_plot_data.append(row)

            print(i)

    # final df to plot
    results_df = pd.DataFrame(summary_plot_data)
    results_df = results_df.rename(columns={"x_value": "Ranked State Space",
        "detfrac": "Deterministic Fraction", "temp": "Temperature", "bitsflipped": "Total Bits Flipped"})
    """

    # ================ Make df for summary as a function of ss size ================
    # x = size
    # y = det fraction or number of bits flipped
    # hue = temp
    """
    # df for some of the plots (faster)
    files = os.listdir('pickle_jar')
    files = list(filter(lambda x: re.search('results_temp', x), files))

    summary_plot_data = []  # list of dicts

    for i, file in enumerate(files):
        with open(os.path.join('pickle_jar', file), 'rb') as handle:

            results = pickle.load(handle)
            temp = float(file.split('_')[2])
            trial = int(file.split('_')[6].split('.')[0])

            for ss in results:

                row = {'State Space Size': len(eval(ss)),
                       'Temperature': temp,
                       'trial': trial,
                       'Deterministic Fraction': results[ss]['deterministic_fraction'],
                       'Total Bits Flipped': results[ss]['bits_flipped'],
                       'Total Bits Flipped (normalized)': results[ss]['bits_flipped']/len(eval(ss)),
                       }

                summary_plot_data.append(row)

            print(i)

    # final df to plot
    results_df = pd.DataFrame(summary_plot_data)
    """

    # ================ for all plots: (avg over trials for error bars, color is temp) ================

    # ================ Plots of things over time ================

    # plot 1: temp vs magnetization for whole state space
    # get sub df with whole statespaces
    #sns.boxplot(data=results_df, x='Temperature', y='Average Magnetization', palette='flare')
    #plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    #plt.tight_layout()
    #plt.show()
    #plt.savefig('fig1.pdf', format='pdf')
    #quit()

    # plot 1b: temp vs bit flips
    #sns.boxplot(data=results_df, x='Temperature', y='Average N Bit Flips', palette='flare')
    #plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    #plt.tight_layout()
    #plt.show()
    #plt.savefig('fig1b.pdf', format='pdf')
    #quit()

    # plot 1c: temp vs bit flips
    #sns.boxplot(data=results_df, x='Temperature', y='Average N Det Bit Flips', palette='flare')
    #plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    #plt.tight_layout()
    #plt.show()
    #plt.savefig('fig1c.pdf', format='pdf')
    #quit()

    # plot 1d: bit vs det bit
    #sns.scatterplot(data=results_df, x='Average N Bit Flips', y='Average N Det Bit Flips',
    #    hue='Temperature', s=6, linewidth=0)
    #plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    #plt.tight_layout()
    #plt.show()
    #plt.savefig('fig1d.pdf', format='pdf')
    #quit()

    # these, but for all state spaces

    # plot 2: det bit flips (or bit flips) vs mag, color is temp and size of points is ss size
    #sns.scatterplot(data=results_df, x='Average N Det Bit Flips', y='Average Magnetization',
    #    hue='Temperature', size='State Space Size', linewidth=0, alpha=0.7, sizes=(1, 200), legend="full")
    #plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    #plt.tight_layout()
    #plt.show()
    #plt.savefig('fig_summary.pdf', format='pdf')
    #quit()

    #sns.scatterplot(data=results_df, x='Average N Bit Flips', y='Average Magnetization',
    #    hue='Temperature', size='State Space Size', linewidth=0, alpha=0.7, sizes=(1, 200), legend="full")
    #plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    #plt.tight_layout()
    #plt.show()
    #plt.savefig('fig_summaryb.pdf', format='pdf')
    #quit()

    # plot 5: boxplots of ss size vs mag, bit, or det. Color = temps
    #sns.boxplot(x="State Space Size", y="Average Magnetization", hue="Temperature", data=results_df, linewidth=0.5,
    #    fliersize=1, palette='flare')
    #plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    #plt.tight_layout()
    #plt.show()
    #plt.savefig('fig5.pdf', format='pdf')
    #quit()

    #sns.boxplot(x="State Space Size", y="Average N Bit Flips", hue="Temperature", data=results_df, linewidth=0.5,
    #    fliersize=1, palette='flare')
    #plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    #plt.tight_layout()
    #plt.show()
    #plt.savefig('fig5b.pdf', format='pdf')
    #quit()

    sns.boxplot(x="State Space Size", y="Average N Det Bit Flips", hue="Temperature", data=results_df, linewidth=0.5,
        fliersize=1, palette='flare')
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.tight_layout()
    #plt.show()
    plt.savefig('fig5c.pdf', format='pdf')
    quit()


    # ================ Plots of summary of runs ================

    # plot 2: ranked ss vs % deterministic distribution
    # get distribution lines and make them into a df
    #sns.lineplot(x="Ranked State Space", y="Deterministic Fraction", hue='Temperature', data=results_df)
    #plt.show()
    #plt.savefig('fig2.pdf', format='pdf')
    #plt.clf()

    #sns.lineplot(x="Ranked State Space", y="Total Bits Flipped", hue='Temperature', data=results_df)
    #plt.show()
    #plt.savefig('fig2b.pdf', format='pdf')

    # plot 3: % deterministic vs total number of bits flipped per state space (scatter with error bars)
    #sns.scatterplot(data=results_df, x="Deterministic Fraction", y="Total Bits Flipped", hue="Temperature",
    #    s=2, alpha=0.7, linewidth=0)
    #plt.show()
    #plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    #plt.tight_layout()
    #plt.savefig('fig3.pdf', format='pdf')

    # plot 4: % determinism or # bit flips as a function of ss size
    #sns.boxplot(x="State Space Size", y="Deterministic Fraction", hue="Temperature", data=results_df, linewidth=0.5,
    #    fliersize=1, palette='flare')
    #plt.show()
    #plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    #plt.tight_layout()
    #plt.savefig('fig4.pdf', format='pdf')

    #sns.boxplot(x="State Space Size", y="Total Bits Flipped (normalized)", hue="Temperature", data=results_df, linewidth=0.5,
    #    fliersize=1, palette='flare')
    #plt.show()
    #plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    #plt.tight_layout()
    #plt.savefig('fig4b.pdf', format='pdf')

    quit()


"""
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
