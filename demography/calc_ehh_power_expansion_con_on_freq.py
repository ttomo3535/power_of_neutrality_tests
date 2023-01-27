###### import subprocess
import numpy as np
import pandas as pd
import csv
import pyper
import random
import os
import multiprocessing
from my_module import forward_trajectory as fwd
from my_module import mbslib
import functools


def make_trajectoryfiles_forward(N0, generation, demography_in_year, t_mutation_in_year, s, h, resolution,
                         n_trajectory, path_to_traj):
    '''
    generate trajectory file

    Args:
        N0 (int):
        generation (int): generation time, years/generation
        demography_in_year (list): demographic history/
        t_mutation_in_year (float): time when mutation arises, in year
        s,
        h,
        resolution,
        n_trajectory (int): number of trajectories
        path_to_traj (str) : path to trajectory files (w/o extentions)

    '''
    for i in range(n_trajectory):

        # file name
        filename = f'{path_to_traj}_{i}.dat'

        # generate trajectory
        trajectory = fwd.mbs_input(t_mutation_in_year,
                                   demography_in_year,
                                   s, h,
                                   generation, N0, resolution,
                                   'NOTLOST')

        # save
        with open(filename, 'w') as f:

            writer = csv.writer(f, delimiter='\t')

            for freq in trajectory:
                writer.writerow(freq)


def run_mbs(nsam, per_site_theta, per_site_rho,
            lsites, selpos,
            n_trajectory, nrep_per_traj,
            path_to_mbs_output, path_to_traj):
    '''
    run mbs
    Args:
        nsam,
        per_site_theta,
        per_site_rho,
        lsites,
        selpos,
        n_trajectory (int): number of trajectory files
        nrep_per_traj (int): number of simulations per trajectory file
        path_to_mbs_output (str) : path to mbs output files (w/o extentions)
        path_to_traj (str) : path to trajectory files (w/o extentions)
    '''

    cmd = f'mbs {nsam} -t {per_site_theta} -r {per_site_rho} '
    cmd += f'-s {lsites} {selpos} '
    cmd += f'-f {n_trajectory} {nrep_per_traj} {path_to_traj} '
    cmd += f'> {path_to_mbs_output}'

    mbslib.run_command(cmd)


def parameter_sets_forward_expansion(k, N1, t, sel_advantages, data_num):
    '''

    Args:
        k: N1/N0, where N0 is current population size and N1 is population size before expansion
        N1: population size before expansion
        t: timing of expansion in generation
        sel_advantages: selection coefficients
        data_num: number of multiproccess

    Returns:

    '''
    params = dict()

    params['N0'] = int(N1/k)
    params['generation'] = 20

    # selection coefficients
    params['s'] = 0
    params['h'] = 0.5  # <--- co-dominance
    params['resolution'] = 100

    # number of trajectory
    params['n_trajectory'] = 1
    # coalescent simulation per trajectory
    params['nrep_per_traj'] = 1

    # number of chromosome
    params['nsam'] = 120
    # length of sequence
    params['lsites'] = 500000
    # target site position
    params['selpos'] = 1

    # mutation rate per site per generation
    params['per_site_theta'] = 1.0 * 10 ** (-8) * 4 * params['N0']
    # recombination rate per site per generation
    params['per_site_rho'] = 1.0 * 10 ** (-8) * 4 * params['N0']

    # time_expansion_in_year
    e_in_year = t * params['generation']
    #
    params['demography_in_year'] =  [[0, e_in_year, params['N0']],
                                    [e_in_year, 100 * params['N0'] * params['generation'], N1]
                                    ]
    # tentative value
    params['t_mutation_in_year'] = 1

    params_list = list()
    for s in sel_advantages:
        params['s'] = s
        for i in data_num:
            params['data_num'] = i
            params_list.append(params.copy())

    return params_list


def calc_EHH_conditional_on_frequency(params, distance_in_bp, data_dir, ehh_data_dir, nrep):
    # parameters set
    # expansion age in year
    e_age = int(params['demography_in_year'][0][1])
    # timing of expansion in generation
    t_e = int(e_age/params['generation'])
    # expansion_strength
    e_strength = int(params['demography_in_year'][0][2] / params['demography_in_year'][1][2])
    # number of replication
    num_EHH = nrep
    # max threshold
    max_iteration = 2
    num_run = 0

    # empty dataframe for allele frequencies
    labels = ['0.01', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45',
                  '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '0.99']
    EHH_data_dic = {}
    for i in labels:
        EHH_data_dic[i] = pd.DataFrame(
            columns=[' EHH_A ', ' EHH_D ', ' rEHH ', ' iHH_A ', ' iHH_D ', ' iHS ', ' f_current ', 'f_current_bin', 't_mutation_in_year'])

    # set mutation age
    max_age = int(5000 * params['generation'] * 16)
    mutation_age_candidate = np.arange(5000, max_age + 1, 1)
    # duration that popsize is N0
    N0_age = np.arange(5000, e_age + 1, 1)
    # duration pop size is N1
    N1_age = np.arange(e_age + 1, max_age + 1, 1)
    # weight
    weight = [10 for i in range(len(N0_age))] + [1 for i in range(len(N1_age))]

    while min([len(EHH_data_dic[i]) for i in labels]) < num_EHH:
        mutation_age = int(random.choices(mutation_age_candidate, weights=weight, k=1)[0])
        params['t_mutation_in_year'] = mutation_age

        # path to trajectory file
        path_to_traj = f"{data_dir}/traj_t{params['t_mutation_in_year']}_s{params['s']}_" \
                       f"eage{t_e}_edegree{e_strength}_datanum{params['data_num']}"
        # mbs output file name
        path_to_mbs_output = f"{data_dir}/mbs_nsam{params['nsam']}_tmutation{params['t_mutation_in_year']}_s{params['s']}" \
                             f"_eage{t_e}_edegree{e_strength}_datanum{params['data_num']}.dat"
        #print('Hello')

        make_trajectoryfiles_forward(params['N0'], params['generation'],
                             params['demography_in_year'], params['t_mutation_in_year'],
                             params['s'], params['h'], params['resolution'],
                             params['n_trajectory'], path_to_traj)

        # condition on selected allele is segregating
        traj_file = "{}/traj_t{}_s{}_eage{}_edegree{}_datanum{}_0.dat"\
            .format(data_dir, params['t_mutation_in_year'], params['s'], t_e, e_strength, params['data_num'])
        dt = pd.read_table(traj_file, header = None)
        d_freq = dt.iloc[0, 3]

        if d_freq == 1:
            os.remove(traj_file)
            pass
        else:
            # run mbs
            run_mbs(params['nsam'], params['per_site_theta'], params['per_site_rho'],
                    params['lsites'], params['selpos'],
                    params['n_trajectory'], params['nrep_per_traj'],
                    path_to_mbs_output, path_to_traj)

            # calc EHH
            # convert mbs format to ms format
            ms_file = '{}/mbs_tmutation{}_s{}_eage{}_edegree{}_datanum{}.txt'.format(
                    data_dir, params['t_mutation_in_year'], params['s'],
                    t_e, e_strength, params['data_num'])
            with open(ms_file, 'w') as f:
                # convert into ms format for each line
                f.write("ms {} {} -t {} -r {} {}\n\n".format(params['nsam'], params['n_trajectory'],
                                                             params['per_site_theta'] * params['lsites'],
                                                             params['per_site_rho'] * params['lsites'], params['lsites']))
                # change the position of mutation if it occurred at target site
                for i in mbslib.parse_mbs_data(path_to_mbs_output):
                    # change the position of mutation if it occurred at target site
                    if i['pos'][0] == 1.0:
                        h = mbslib.mbs_to_ms_output(i, params['selpos'], params['lsites'])
                        f.write("//\n")
                        # write segregation sites
                        f.write("segsites: {}\n".format(len(h['pos'])))

                        # write position
                        f.write("positions: ")
                        # convert int to str
                        pos_list = [str(i) for i in h['pos']]
                        # change position of the mutation occurred at the target site
                        pos_list[1] = str(2 / params['lsites'])
                        f.write(" ".join(pos_list))
                        f.write("\n")

                        # write seq data
                        f.write("\n".join(h["seq"]))
                        f.write("\n\n")

                    else:
                        h = mbslib.mbs_to_ms_output(i, params['selpos'], params['lsites'])
                        f.write("//\n")
                        # write segregating sites
                        f.write("segsites: {}\n".format(len(h['pos'])))

                        # write position
                        f.write("positions: ")
                        # convert int to str
                        pos_list = [str(i) for i in h['pos']]
                        f.write(" ".join(pos_list))
                        f.write("\n")

                        # write seq
                        f.write("\n".join(h["seq"]))
                        f.write("\n\n")

            # run R script to calculate EHH
            # set parameter values
            r = pyper.R(use_numpy='True', use_pandas='True')
            r("nrep <- {}".format(params['n_trajectory']))
            r("lsites <- {}".format(params['lsites']))
            r("distance_in_bp <- {}".format(distance_in_bp))
            # r("selpos <- {}".format(mrk))
	    r.assign("ms_file", ms_file)
            r.assign("traj_file", traj_file)
            r("source(file='calc_EHH_demography_pyper.R')")

            # extact EHH data
            #temp_data = r.get("temporary_list")
            EHH_data = pd.DataFrame(r.get('EHH_data'))

            # add EHH data according to its frequency
            label = ['0.01', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45',
                      '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '0.99']
            bins = [0, 0.025, 0.075, 0.125, 0.175, 0.225, 0.275, 0.325, 0.375, 0.425,
                    0.475, 0.525, 0.575, 0.625, 0.675, 0.725, 0.775, 0.825, 0.875,
                    0.925, 0.975, 1.0]

            EHH_data['f_current_bin'] = pd.cut(EHH_data[' f_current '], bins, labels=label, right=False)
            EHH_data['t_mutation_in_year'] = params['t_mutation_in_year']

            for i in labels:
                # extract for derived allele frequencies
                temp_df = EHH_data[EHH_data['f_current_bin'] == i]
                EHH_data_dic[i] = pd.concat([EHH_data_dic[i], temp_df], axis=0, ignore_index=True)

            #print('mutation_age:', params['t_mutation_in_year'], 'num_data', params['data_num'], 'num run', num_run)
            # remove intermediate file
            os.remove(traj_file)
            os.remove(path_to_mbs_output)
            os.remove(ms_file)

            num_run += 1
            if num_run == 500:
                print('data_num:', params['data_num'] ,num_run, 'times run')
            if num_run > max_iteration:
                break

    # save
    for i in labels:
        EHH_data_dic[i].to_csv(
            '{}/EHH_data_f{}_s{}_eage{}_estrength{}_datanum{}.csv'
                .format(ehh_data_dir, i, params['s'], t_e, e_strength , params['data_num']))

    print('data_num', params['data_num'], 'done', num_run, 'times run')


def calc_power(ehh_data_dir, data_num, t_e, k, s, nrep, path_to_percentile, power_dir, file_name):
    labels = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.99']

    # empty data frame
    EHH_data_dic = {}
    for i in labels:
        EHH_data_dic[i] = pd.DataFrame(
            columns=[' EHH_A ', ' EHH_D ', ' rEHH ', ' iHH_A ', ' iHH_D ', ' iHS ', ' f_current ', 'f_current_bin',
                     't_mutation_in_year'])
    # joint data
    for m in labels:
        for n in data_num:
            df = pd.read_csv(
                '{}/EHH_data_f{}_s{}_eage{}_estrength{}_datanum{}.csv'
                .format(ehh_data_dir, m, s, t_e, k, n), index_col=0)
            EHH_data_dic[m] = pd.concat([EHH_data_dic[m], df], axis=0, ignore_index=True)
        #print(len(EHH_data_dic[m]))
        EHH_data_dic[m].to_csv('{}/EHH_data_f{}_s{}_eage{}_estrength{}.csv'
                               .format(ehh_data_dir, m, s, t_e, k), index=False)

    # calc power
    rEHH_power = []
    # calc rEHH power
    rEHH_percentile = path_to_percentile[:11] + 'rEHH_' + path_to_percentile[11:]
    df = pd.read_csv(rEHH_percentile, index_col=0, header=0)
    threshold_list = df['95']
    for f in labels:
        threshold = threshold_list[float(f)]
        df = pd.read_csv('{}/EHH_data_f{}_s{}_eage{}_estrength{} .csv'
                         .format(ehh_data_dir, f, s, t_e, k))
        df = df[:nrep]
        df = df.replace([np.inf, -np.inf], np.nan)
        power = sum([i > threshold for i in df['rEHH']])/nrep
        rEHH_power.append(power)

    rEHH_power_df = pd.DataFrame(rEHH_power, columns=t_e, index=labels)
    rEHH_power_df.to_csv("{}/rEHH_{}".format(power_dir, file_name))

    # calc power
    iHS_power = []
    # calc rEHH power
    iHS_percentile = path_to_percentile[:11] + 'iHS_' + path_to_percentile[11:]
    df = pd.read_csv(iHS_percentile, index_col=0, header=0)
    threshold_list = df['5']
    for f in labels:
        threshold = threshold_list[float(f)]
        df = pd.read_csv('{}/EHH_data_f{}_s{}_eage{}_estrength{} .csv'
                         .format(ehh_data_dir, f, s, t_e, k))
        df = df[:nrep]
        df = df.replace([np.inf, -np.inf], np.nan)
        power = sum([i > threshold for i in df['iHS']]) / nrep
        iHS_power.append(power)

    iHS_power_df = pd.DataFrame(iHS_power, columns=t_e, index=labels)
    iHS_power_df.to_csv("{}/iHS_{}".format(power_dir, file_name))


def main():
    # initial values
    # selection
    sel_advantages = [0, 0.005]
    # N1/N0
    k = 0.1
    # N1 population size before expansion
    N1 = 4770
    # timing of expansion in generation
    t = 500
    #
    data_num = np.arange(1, 21, 1)
    # number of replication
    nrep = 5
    # parameters sets
    parameter_sets_forward_expansion(k, N1, t, sel_advantages, data_num)
    testlist = parameter_sets_forward_expansion(k, N1, t, sel_advantages, data_num)
    # simulated data directory name
    data_dir = 'results'
    # statistics data directory name
    ehh_data_dir = 'ehh_data'
    # the point at which EHH values are calculated
    distance_in_bp = 20000

    n_cpu = int(multiprocessing.cpu_count() / 2)
    with multiprocessing.Pool(processes=n_cpu) as p:
        p.map(functools.partial(calc_EHH_conditional_on_frequency, distance_in_bp=distance_in_bp,
                                data_dir=data_dir, ehh_data_dir=ehh_data_dir, nrep=nrep), testlist)

    # power data directory
    power_dir = 'power'
    # file name of power data
    file_name = 'power_eage{}_estrength{}_nullex_con_on_freq.csv'.format(t, int(1/k))
    # path_to_percentile_data
    path_to_percentile = 'percentile/percentile_eage{}_estrength{}.csv'.format(t, int(1/k))
    calc_power(ehh_data_dir, data_num, t_e=t, k=int(1/k), s=sel_advantages[1],
               nrep=nrep, path_to_percentile=path_to_percentile, power_dir=power_dir, file_name=file_name)



if __name__=="__main__":
    main()

