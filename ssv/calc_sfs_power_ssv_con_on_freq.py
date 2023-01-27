import os
import csv
import random
import numpy as np
import multiprocessing
import pandas as pd
from my_module import mbslib
from my_module import trajectory_given_t_and_p as trj
import functools

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


def calc_thetapi_S_thetaw_thetal(ms_seq_data, ms_pos_data):
    """ Calculate theta

     Args:
         ms_seq_data(list(str): ms 0-1 sequence data of one replication
         ms_pos_data(list(float)): SNP position data of one replication

     Return:
         thetapi:
         S:
         thetaw:
         thetal:
         thetah:

     """
    # number of sample
    nsam = len(ms_seq_data)

    # get sequence data in int
    int_site_list = [[int(i) for i in list(j)] for j in ms_seq_data]

    # calc theta_l
    # calc number of segregating sites
    l = sum([sum(i) for i in int_site_list])
    # calc average number of segregating sites
    thetal = l / (nsam - 1)

    # calc theta_pi, theta_h
    # calc sum of pairwise differences
    k = 0
    h = 0
    for i in zip(*int_site_list):
        der = sum(i)
        k += der * (nsam - der)
        h += der ** 2
    # clac_thetapi/h
    thetapi = k * 2 / (nsam * (nsam - 1))
    thetah = h * 2 / (nsam * (nsam - 1))

    # calc theta_w
    # calc number of segregating sites
    S = len(ms_pos_data)
    # calc theta_w
    a = 0
    for j in range(1, nsam):
        a += 1 / j
    thetaw = S / a

    return thetapi, S, thetaw, thetal, thetah


def calc_H(thetapi, S, thetaw, thetal, thetah, n):
    """ Calculate normalized version of Fay and Wu's H

    Args:
        thetapi:
        S:
        thetaw:
        thetal:
        thetah:
        n: sample size

    Return:
        H: Fay and Wu H
    """
    # calc variation of H
    a = sum([1 / i for i in range(1, n)])
    b = sum([1 / (i ** 2) for i in range(1, n)])
    v = (n - 2) * thetaw / (6 * (n - 1)) + (
                18 * n ** 2 * (3 * n + 2) * (b + 1 / n ** 2) - (88 * n ** 3 + 9 * n ** 2 - 13 * n + 6)) * (
                    S * (S - 1) / (a ** 2 + b)) / (9 * n * (n - 1) ** 2)
    # return nan if number of segregating sites is too small to calculate H
    if v == 0:
        return np.nan
    else:
        H = (thetapi - thetal) / v ** (1 / 2)
        return H


def calc_D(thetapi, S, thetaw, thetal, thetah, n):
    """ Calculate Tajima's D

    Args:
        thetapi:
        S: number of segregating sites
        thetaw:
        thetal:
        thetah:
        n: sample size

    Return:
         D: Tajima's D
    """
    # calc variation of D
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


def parameter_sets_given_t_p(sel_advantages, data_num, initial_freq):
    params = dict()
    params['N0'] = 5000
    params['generation'] = 20
    params['demography_in_year'] = [[0, 100 * params['N0'] * params['generation'], params['N0']]]

    # selection coefficients
    params['h'] = 0.5  # <--- co-dominance
    params['resolution'] = 0.01

    # number of trajectory
    params['n_trajectory'] = 1
    # coalescent simulation per trajectory
    params['nrep_per_traj'] = 1

    # number of chromosome
    params['nsam'] = 120
    # length of sequence
    params['lsites'] = 10000
    # position of target site
    params['selpos'] = int(params['lsites'] / 2)

    # mutation rate per site per generation
    params['per_site_theta'] = 1.0 * 10 ** (-8) * 4 * params['N0']
    # recombination rate per site per generation
    params['per_site_rho'] = 1.0 * 10 ** (-8) * 4 * params['N0']

    # time selection started in year
    # tentative value
    params['t0_in_year'] = 1
    # time selection started in unit of 4N generation
    params['t0'] = params['t0_in_year'] / (4 * params['N0'] * params['generation'])
    # derived allele frequency at which selection started acting
    params['p0'] = initial_freq

    params_list = list()
    for s in sel_advantages:
        params['s'] = s
        params['selection_in_year'] = [[0, params['t0_in_year'], params['s'], params['h']]]
        for d in data_num:
            # time selection started in year
            params['data_num'] = d
            params_list.append(params.copy())

    return params_list


def make_tarjctoyfiles_given_t_p(N0, t0, p0, generation, demography_in_year, selection_in_year, resolution,
                                 n_trajectory, path_to_traj):
    '''
        Args:
            N0 (int):
            t0 (int):
            p0 (float):
            generation (int): generation time, years/generation
            demography_in_year (list): demographic history/
            selection_in_year (lise): selection history
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
        dem_under_neutrality, history = trj.prepare_history(demography_in_year, selection_in_year, t0, N0, generation)
        trajectory = trj.generate_trajectory(dem_under_neutrality, history, t0, p0, N0, resolution, 'NOTLOST')
        #print(trajectory)

        # save
        with open(filename, 'w') as f:

            writer = csv.writer(f, delimiter='\t')

            for freq in trajectory:
                writer.writerow(freq)


def calc_stat_conditional_on_frequency(params, data_dir, save_dir, nrep):
    # calculation set
    num_sfs = nrep
    max_iteration = 3
    num_run = 0


    initial_freq = params['p0']
    labels = ['0.01', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45',
              '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '0.99']
    labels = [i for i in labels if float(i) > initial_freq]
    #print('label:', label_a)

    bins = [0.025, 0.075, 0.125, 0.175, 0.225, 0.275, 0.325, 0.375, 0.425,
            0.475, 0.525, 0.575, 0.625, 0.675, 0.725, 0.775, 0.825, 0.875,
            0.925, 0.975, 1.0]
    bins = [i for i in bins if i > initial_freq - 0.025]

    # generate empty dataframe for each derived allele frequency
    statistics_data_dic = {}
    for i in labels:
        statistics_data_dic[i] = pd.DataFrame(
            columns=['D', 'H', 'E', 'f_current', 'f_current_bin', 't0_in_year'])

    theta_data_dic = {}
    for i in labels:
        theta_data_dic[i] = pd.DataFrame(columns=['theta_pi', 'S', 'theta_w', 'theta_l', 'theta_h',
                                                  'f_current', 'f_current_bin', 't0_in_year'])

    # set candidate t0 range
    max_age = int(5000 * params['generation'] * 16)
    t0_age_candidate = np.arange(1000, max_age + 1, 1)
    # weight according to popsize
    weight = [1 for i in range(len(t0_age_candidate))]

    while min([len(theta_data_dic[i]) for i in labels]) < num_sfs:
        t0_in_year = int(random.choices(t0_age_candidate, weights=weight, k=1)[0])
        # time selection started in year
        params['t0_in_year'] = t0_in_year
        # time selection started in unit of 4N generation
        params['t0'] = params['t0_in_year'] / (4 * params['N0'] * params['generation'])
        # selection selection history
        params['selection_in_year'] = [[0, params['t0_in_year'], params['s'], params['h']]]

        # path to trajectory file
        path_to_traj = f"{data_dir}/traj_t0{params['t0_in_year']}_p0{params['p0']}_s{params['s']}_datanum{params['data_num']}"
        # mbs out put file
        path_to_mbs_output = f"{data_dir}/mbs_nsam{params['nsam']}_t0{params['t0_in_year']}" \
                             f"_p0{params['p0']}_s{params['s']}_datanum{params['data_num']}.dat"

        # generate trajectory
        make_tarjctoyfiles_given_t_p(params['N0'], params['t0'], params['p0'], params['generation'],
                                     params['demography_in_year'], params['selection_in_year'], params['resolution'],
                                     params['n_trajectory'], path_to_traj)

        # condition on derived allele is segregating
        traj_file = "{}/traj_t0{}_p0{}_s{}_datanum{}_0.dat" \
            .format(data_dir, params['t0_in_year'], params['p0'], params['s'], params['data_num'])
        dt = pd.read_table(traj_file, header=None)
        d_freq = dt.iloc[0, 3]
        #print(d_freq)

        if d_freq == 1:
            os.remove(traj_file)
            pass
        else:
            # run mbs to make SNP data
            run_mbs(params['nsam'], params['per_site_theta'], params['per_site_rho'],
                    params['lsites'], params['selpos'],
                    params['n_trajectory'], params['nrep_per_traj'],
                    path_to_mbs_output, path_to_traj)

            # calc SFS
            n = params['nsam']
            theta_list = [calc_thetapi_S_thetaw_thetal(m['seq'], m['pos']) for m in mbslib.parse_mbs_data(path_to_mbs_output)]
            statistics_list = []
            statistics_list.append([calc_D(*i, n) for i in theta_list])
            statistics_list.append([calc_H(*i, n) for i in theta_list])
            statistics_list = np.array(statistics_list).T.tolist()

            if len(theta_list) != 1:
                pass
            else:
                # extract current allele frequency
                theta_list = list(theta_list[0])
                theta_list.append(d_freq)
                statistics_list[0].append(d_freq)

                # list to dataframe
                theta_df = pd.DataFrame([theta_list], columns = ['theta_pi', 'S', 'theta_w', 'theta_l', 'theta_h', 'f_current'])
                statistics_df = pd.DataFrame([statistics_list[0]], columns = ['D', 'H', 'f_current'])

                theta_df['f_current_bin'] = pd.cut(theta_df['f_current'], bins, labels=labels, right=False)
                theta_df['t0_in_year'] = params['t0']
                statistics_df['f_current_bin'] = pd.cut(theta_df['f_current'], bins, labels=labels, right=False)
                statistics_df['t0_in_year'] = params['t0_in_year']

                # add data based on allele frequency
                for i in labels:
                    # extract based on allele frequency
                    temp_df = theta_df[theta_df['f_current_bin'] == i]
                    # add
                    theta_data_dic[i] = pd.concat([theta_data_dic[i], temp_df], axis=0, ignore_index=True)
                    # extract based on allele frequency
                    temp_df = statistics_df[statistics_df['f_current_bin'] == i]
                    # add
                    statistics_data_dic[i] = pd.concat([statistics_data_dic[i], temp_df], axis=0, ignore_index=True)

                os.remove(traj_file)
                os.remove(path_to_mbs_output)
                num_run += 1
                if num_run == 500:
                    print('data_num:', params['data_num'] ,num_run, 'times run')
                if num_run > max_iteration:
                    break
    # csv file
    for i in labels:
        theta_data_dic[i].to_csv('{}/theta_data_f{}_p0{}_s{}_datanum{}.csv'
                                 .format(save_dir, i, params['p0'], params['s'], params['data_num']))
        statistics_data_dic[i].to_csv('{}/statistics_data_f{}_p0{}_s{}_datanum{}.csv'
                                      .format(save_dir, i, params['p0'], params['s'], params['data_num']))
    print('data_num:', params['data_num'], num_run, 'times run')


def calc_power(save_dir, labels, data_num, p, s, nrep, path_to_percentile, power_dir, file_name):
    # empty dataframe
    sfs_data_dic = {}
    for i in labels:
        sfs_data_dic[i] = pd.DataFrame(
            columns=['D', 'H', 'f_current', 'f_current_bin', 't_mutation_in_year'])

    for m in labels:
        for n in data_num:
            df = pd.read_csv(
                '{}/statistics_data_f{}_p0{}_s{}_datanum{}.csv'
                .format(save_dir, m, p, s, n), index_col=0)
            sfs_data_dic[m] = pd.concat([sfs_data_dic[m], df], axis=0, ignore_index=True)
        sfs_data_dic[m].to_csv('{}/statistics_data_f{}_p0{}_s{}.csv'
                .format(save_dir, m, p, s), index=False)


    # empty dataframe
    sfs_data_dic = {}
    for i in labels:
        sfs_data_dic[i] = pd.DataFrame(
            columns=['theta_pi', 'S', 'theta_w', 'theta_l', 'theta_h', 'f_current', 'f_current_bin',
                     't_mutation_in_year'])

    for m in labels:
        for n in data_num:
            df = pd.read_csv(
                '{}/theta_data_f{}_p0{}_s{}_datanum{}.csv'
                .format(save_dir, m, p, s, n), index_col=0)

            sfs_data_dic[m] = pd.concat([sfs_data_dic[m], df], axis=0, ignore_index=True)
        sfs_data_dic[m].to_csv(
            '{}/theta_data_f{}_p0{}_s{}.csv'
                .format(save_dir, m, p, s), index=False)

    # calc power
    D_power = []
    H_power = []
    df = pd.read_csv(path_to_percentile, index_col=0, header=0)
    D_thres = df['5']['D']
    H_thres = df['5']['H']
    for f in labels:
        df = pd.read_csv('{}/statistics_data_f{}_p0{}_s{}.csv'.format(save_dir, f, p, s))
        df = df[:nrep]
        df = df.replace([np.inf, -np.inf], np.nan)
        D_power.append(sum([i < D_thres for i in df['D']])/nrep)
        H_power.append(sum([i < H_thres for i in df['H']])/nrep)

    # save
    df_D = pd.DataFrame([D_power], index=['{}'.format(p)], columns=labels)
    df_H = pd.DataFrame([H_power], index=['{}'.format(p)], columns=labels)
    df_D.to_csv('{}/D_{}'.format(power_dir, file_name))
    df_H.to_csv('{}/H_{}'.format(power_dir, file_name))


def main():
    sel_advantages = [0.005]
    initial_freq = 0.1
    data_num = np.arange(1, 2, 1)
    # number of replication
    nrep = 2
    # simulated data directory name
    data_dir = 'results'
    # statistics data directory name
    save_dir = 'sfs_data'

    testlist = parameter_sets_given_t_p(sel_advantages, data_num, initial_freq)
    n_cpu = int(multiprocessing.cpu_count() / 2)
    with multiprocessing.Pool(processes=n_cpu) as p:
        p.map(functools.partial(calc_stat_conditional_on_frequency,
                                data_dir=data_dir, save_dir=save_dir, nrep=nrep), testlist)

    f_list = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.99']
    f_list = [i for i in f_list if float(i) > initial_freq]
    # power data directory name
    power_dir = 'power'
    # path_to_percentile_data
    path_to_percentile = '../constant_pop/percentile_data/percentile_D_H_E.csv'
    # power file name
    file_name = 'power_p{}_s{}_con_on_freq.csv'.format(initial_freq, sel_advantages[0])
    calc_power(save_dir, f_list, data_num, initial_freq, sel_advantages[0], nrep, path_to_percentile, power_dir, file_name)

if __name__=="__main__":
    main()




