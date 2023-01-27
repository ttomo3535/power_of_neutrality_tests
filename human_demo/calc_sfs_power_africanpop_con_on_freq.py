import numpy as np
import pandas as pd
import csv
import random
import pyper
import multiprocessing
import os
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


def parameter_sets_forward(N0, gen, demo, sel_advantages, data_num):
    '''

    Args:
        N0(int): current populatino size
        gen(int): generation in year
        demo(list): demography
        sel_advantages(list): selection coefficients
        data_num(list):

    Returns:

    '''
    params = dict()

    params['N0'] = N0
    params['generation'] = gen
    params['demography_in_year'] = demo

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
    # position of target site
    params['selpos'] = 1

    # mutation rate per site per generation
    params['per_site_theta'] = 1.0 * 10 ** (-8) * 4 * params['N0']
    # recombination rate per site per generation
    params['per_site_rho'] = 1.0 * 10 ** (-8) * 4 * params['N0']
    # tentative value
    params['t_mutation_in_year'] = 1

    # candidate mutation ages in year
    params_list = list()
    for s in sel_advantages:
        params['s'] = s
        for i in data_num:
            params['data_num'] = i
            params_list.append(params.copy())

    return params_list


def calc_thetapi_thetah(ms_seq_data, ms_pos_data):
    seq = ms_seq_data
    nsam = len(seq)
    int_site_list = [[int(i) for i in list(j)] for j in seq]
    # calc heterozygosity and homozygosity
    k = 0
    l = 0
    for i in zip(*int_site_list):
        der = sum(i)
        k += der * (nsam - der)
        l += der ** 2

    # calc theta pi
    theta_pi = k * 2 / (nsam * (nsam - 1))
    # calc theta h
    theta_h = l * 2 / (nsam * (nsam - 1))
    # calc H
    h = theta_pi - theta_h

    return theta_pi, theta_h, h


def calc_thetapi_S_thetaw_thetal(ms_seq_data, ms_pos_data):
    seq = ms_seq_data
    nsam = len(seq)
    int_site_list = [[int(i) for i in list(j)] for j in seq]
    ##calc_theta_l##
    l = sum([sum(i) for i in int_site_list])
    thetal = l / (nsam - 1)
    # calc_theta_pi/theta_h
    k = 0
    h = 0
    for i in zip(*int_site_list):
        der = sum(i)
        k += der * (nsam - der)
        h += der ** 2
    # clac_thetapi/h
    thetapi = k * 2 / (nsam * (nsam - 1))
    thetah = h * 2 / (nsam * (nsam - 1))
    ##calc_theta_w##
    S = len(ms_pos_data)
    a = 0
    for j in range(1, nsam):
        a += 1 / j

    thetaw = S / a

    return thetapi, S, thetaw, thetal, thetah


def calc_H(thetapi, S, thetaw, thetal, thetah, n):
    a = sum([1 / i for i in range(1, n)])
    b = sum([1 / (i ** 2) for i in range(1, n)])
    v = (n - 2) * thetaw / (6 * (n - 1)) + (
                18 * n ** 2 * (3 * n + 2) * (b + 1 / n ** 2) - (88 * n ** 3 + 9 * n ** 2 - 13 * n + 6)) * (
                    S * (S - 1) / (a ** 2 + b)) / (9 * n * (n - 1) ** 2)

    if v == 0:
        return np.nan
    else:
        H = (thetapi - thetal) / v ** (1 / 2)
        return H


def calc_D(thetapi, S, thetaw, thetal, thetah, n):
    a1 = 0
    for i in range(1, n):
        a1 += 1 / i
    a2 = 0
    for i in range(1, n):
        a2 += 1 / i ** 2
    b1 = (n + 1) / (3 * (n - 1))
    b2 = 2 * (n ** 2 + n + 3) / (9 * n * (n - 1))
    c1 = b1 - 1 / a1
    c2 = b2 - (n + 2) / (a1 * n) + a2 / a1 ** 2
    e1 = c1 / a1
    e2 = c2 / (a1 ** 2 + a2)
    C = (e1 * S + e2 * S * (S - 1)) ** 0.5
    if C == 0:
        return np.nan
    else:
        D = (thetapi - thetaw) / C
        return D


def parameter_sets_forward(N0, gen, demo, sel_advantages, data_num):
    '''

    Args:
        N0(int): current populatino size
        gen(int): generation in year
        demo(list): demography
        sel_advantages(list): selection coefficients
        data_num(list):

    Returns:

    '''
    params = dict()

    params['N0'] = N0
    params['generation'] = gen
    params['demography_in_year'] = demo

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
    params['lsites'] = 10000
    # position of target site
    params['selpos'] = 2/params['lsites']

    # mutation rate per site per generation
    params['per_site_theta'] = 1.0 * 10 ** (-8) * 4 * params['N0']
    # recombination rate per site per generation
    params['per_site_rho'] = 1.0 * 10 ** (-8) * 4 * params['N0']
    # tentative value
    params['t_mutation_in_year'] = 1

    # candidate mutation ages in year
    params_list = list()
    for s in sel_advantages:
        params['s'] = s
        for i in data_num:
            params['data_num'] = i
            params_list.append(params.copy())

    return params_list


def calc_sfs_conditional_on_frequency(params, pop_name, nrep, data_dir, save_dir):
    # calculation set
    num_sfs = nrep
    max_iteration = 2
    num_run = 0

    # empty dataframe
    labels = ['0.01', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45',
                      '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '0.99']
    statistics_data_dic = {}
    for i in labels:
        statistics_data_dic[i] = pd.DataFrame(columns=['D', 'H', 'f_current', 'f_current_bin', 't_mutation_in_year'])
    
    theta_data_dic = {}
    for i in labels:
        theta_data_dic[i] = pd.DataFrame(columns=['theta_pi', 'S', 'theta_w', 'theta_l', 'theta_h', 
                                                  'f_current', 'f_current_bin', 't_mutation_in_year'])

    max_age = int(5000 * params['generation'] * 16)
    mutation_age_candidate = np.arange(4000, max_age + 1, 1)
    # Ne = 10000
    t1 = params['demography_in_year'][0][1]
    t2 = params['demography_in_year'][1][1]
    t3 = params['demography_in_year'][2][1]
    t4 = params['demography_in_year'][3][1]

    N0 = params['N0']
    N1 = params['demography_in_year'][1][2]
    N2 = params['demography_in_year'][2][2]
    N3 = params['demography_in_year'][3][2]
    N4 = params['demography_in_year'][4][2]

    # popsize range in year
    N0_ages = np.arange(4000, t1 + 1, 1)
    N1_ages = np.arange(t1 + 1, t2 + 1, 1)
    N2_ages = np.arange(t2 + 1, t3 + 1, 1)
    N3_ages = np.arange(t3 + 1, t4 + 1, 1)
    N4_ages = np.arange(t4 + 1, max_age + 1, 1)

    # weight
    weight = [N0 / N0 for i in range(len(N0_ages))] + [N1 / N0 for i in range(len(N1_ages))] + \
             [N2 / N0 for i in range(len(N2_ages))] + [N3 / N0 for i in range(len(N3_ages))] + \
             [N4 / N0 for i in range(len(N4_ages))]

    while min([len(theta_data_dic[i]) for i in labels]) < num_sfs:
        mutation_age = int(random.choices(mutation_age_candidate, weights=weight, k=1)[0])
        params['t_mutation_in_year'] = mutation_age

        # path to trajectory
        path_to_traj = f"{data_dir}/traj_tmutation{params['t_mutation_in_year']}_s{params['s']}_popname{pop_name}_datanum{params['data_num']}"

        # path to mbs output
        path_to_mbs_output = f"{data_dir}/mbs_nsam{params['nsam']}_tmutation{params['t_mutation_in_year']}_s{params['s']}" \
                             f"_popname{pop_name}_datanum{params['data_num']}.dat"

        make_trajectoryfiles_forward(params['N0'], params['generation'],
                                     params['demography_in_year'], params['t_mutation_in_year'],
                                     params['s'], params['h'], params['resolution'],
                                     params['n_trajectory'], path_to_traj)

        # condition on selected allele is segregating
        traj_file = "{}/traj_tmutation{}_s{}_popname{}_datanum{}_0.dat" \
            .format(data_dir, params['t_mutation_in_year'], params['s'], pop_name, params['data_num'])
        dt = pd.read_table(traj_file, header=None)
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

            # sfs
            n = params['nsam']
            theta_list = [calc_thetapi_S_thetaw_thetal(m['seq'], m['pos']) for m in mbslib.parse_mbs_data(path_to_mbs_output)]
            D_list = [calc_D(*i, n) for i in theta_list]
            H_list = [calc_H(*i, n) for i in theta_list]
            statistics_list = []
            statistics_list.append(D_list)
            statistics_list.append(H_list)
            statistics_list = np.array(statistics_list).T.tolist()

            #print(len(theta_list))
            if len(theta_list) != 1:
                pass

            else:
                # extract current allele frequency
                df = pd.read_table('{}_{}.dat'.format(path_to_traj, 0), header = None)
                freq = df.iat[0, 3]
                theta_list = list(theta_list[0])
                theta_list.append(freq)
                #print(freq)
                statistics_list[0].append(freq)

                # data frame
                theta_df = pd.DataFrame([theta_list], columns = ['theta_pi', 'S', 'theta_w', 'theta_l', 'theta_h', 'f_current'])
                statistics_df = pd.DataFrame([statistics_list[0]], columns = ['D', 'H', 'f_current'])

                label = ['0.01', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45',
                          '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '0.99']
                bins = [0, 0.025, 0.075, 0.125, 0.175, 0.225, 0.275, 0.325, 0.375, 0.425,
                        0.475, 0.525, 0.575, 0.625, 0.675, 0.725, 0.775, 0.825, 0.875,
                        0.925, 0.975, 1.0]

                theta_df['f_current_bin'] = pd.cut(theta_df['f_current'], bins, labels=label, right=False)
                theta_df['t_mutation_in_year'] = params['t_mutation_in_year']
                statistics_df['f_current_bin'] = pd.cut(theta_df['f_current'], bins, labels=label, right=False)
                statistics_df['t_mutation_in_year'] = params['t_mutation_in_year']

                for i in labels:
                    temp_df = theta_df[theta_df['f_current_bin'] == i]
                    theta_data_dic[i] = pd.concat([theta_data_dic[i], temp_df], axis=0, ignore_index=True)
                    temp_df = statistics_df[statistics_df['f_current_bin'] == i]
                    statistics_data_dic[i] = pd.concat([statistics_data_dic[i], temp_df], axis=0, ignore_index=True)

                num_run += 1
                if num_run == 500:
                    print('data_num:', params['data_num'] ,num_run, 'times run')
                if num_run > max_iteration:
                    break

    for i in labels:
        theta_data_dic[i].to_csv(
            '{}/theta_data_f{}_s{}_popname{}_datanum{}.csv'
            .format(save_dir, i, params['s'], pop_name, params['data_num']))
        statistics_data_dic[i].to_csv(
            '{}/statistics_data_f{}_s{}_popname{}_datanum{}.csv'
            .format(save_dir, i, params['s'], pop_name, params['data_num']))

    print('data_num:', params['data_num'], num_run, 'times run', pop_name)


def calc_power(path_to_percentile, data_dir, s, pop_name, data_num, nrep, power_dir, file_name):
    labels = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.99']
    # empty dataframe
    sfs_data_dic = {}
    for i in labels:
        sfs_data_dic[i] = pd.DataFrame(
            columns=['D', 'H', 'f_current', 'f_current_bin', 't_mutation_in_year'])

    for m in labels:
        for n in data_num:
            df = pd.read_csv(
                '{}/statistics_data_f{}_s{}_popname{}_datanum{}.csv'
                .format(data_dir , m, s, pop_name, n), index_col=0)
            sfs_data_dic[m] = pd.concat([sfs_data_dic[m], df], axis=0, ignore_index=True)
        sfs_data_dic[m].to_csv(
            '{}/statistics_data_f{}_s{}_popname{}.csv'
                .format(data_dir, m, s, pop_name), index=False)

    # empty dataframe
    sfs_data_dic = {}
    for i in labels:
        sfs_data_dic[i] = pd.DataFrame(
            columns=['theta_pi', 'S', 'theta_w', 'theta_l', 'theta_h', 'f_current', 'f_current_bin',
                     't_mutation_in_year'])

    for m in labels:
        for n in data_num:
            df = pd.read_csv(
                '{}/theta_data_f{}_s{}_popname{}_datanum{}.csv'
                .format(data_dir, m, s, pop_name, n), index_col=0)

            sfs_data_dic[m] = pd.concat([sfs_data_dic[m], df], axis=0, ignore_index=True)
        sfs_data_dic[m].to_csv(
            '{}/theta_data_f{}_s{}_popname{}.csv'
                .format(data_dir, m, s, pop_name), index=False)

    df = pd.read_csv(path_to_percentile)
    D_thres = df['5'][0]
    H_thres = df['5'][1]

    # calc power
    D_power_list = []
    H_power_list = []
    for f in labels:
        df = pd.read_csv(
            "{}/statistics_f{}_s{}_popname{}.csv"
                .format(data_dir, f, s, pop_name))
        D_power = sum([i < D_thres for i in df["D"]])/nrep
        H_power = sum([i < H_thres for i in df["H"]])/nrep
        D_power_list.append(D_power)
        H_power_list.append(H_power)

    # save power
    df_D = pd.DataFrame([D_power_list], index = ['{}'.format(b_end)], columns = labels)
    df_H = pd.DataFrame([H_power_list], index = ['{}'.format(b_end)], columns = labels)
    df_D.to_csv('{}/D_{}'.format(power_dir, file_name))
    df_H.to_csv('{}/H_{}'.format(power_dir, file_name))


# initial values
def main():#data_num = np.arange(1, 21, 1)
    data_num = [1, 2]
    # selection coefficient
    sel_advantages = [0]
    # generation in year
    gen = 20
    # pop size in unite of N0
    N0 = 100000
    N1 = 24000
    N2 = 2500
    N3 = 24000
    N4 = 12500
    # demographic event timing in year
    t1 = 200 * gen
    t2 = 3500 * gen
    t3 = 3600 * gen
    t4 = 17000 * gen
    demo = [[0, t1, N0],
             [t1, t2, N1],
             [t2, t3, N2],
             [t3, t4, N3],
             [t4, 100 * N0 * gen, N4]
             ]
    # parameters sets
    testlist = parameter_sets_forward(N0, gen, demo, sel_advantages, data_num)
    # simulated data directory name
    data_dir = 'results'
    # statistics data directory name
    save_dir = 'sfs_data'
    #
    pop_name = 'african'
    # nuber of replication
    nrep = 2

    n_cpu = int(multiprocessing.cpu_count() / 2)
    with multiprocessing.Pool(processes=n_cpu) as p:
        p.map(functools.partial(calc_sfs_conditional_on_frequency,
                                pop_name = pop_name, data_dir=data_dir, save_dir=save_dir, nrep=nrep), testlist)

    # power data directory
    power_dir = 'power'
    # file name of power data
    file_name = 'power_popname{}_con_on_freq.csv'.format(pop_name)
    # path_to_percentile_data
    path_to_percentile = 'percentile/percentile_sfs_popname{}.csv'.format(pop_name)
    # calc power
    calc_power(path_to_percentile, save_dir, sel_advantages[0], pop_name, data_num, nrep, power_dir, file_name)

if __name__=="__main__":
    main()






