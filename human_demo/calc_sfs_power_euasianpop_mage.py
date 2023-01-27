import numpy as np
import pandas as pd
import csv
from my_module import forward_trajectory as fwd
from my_module import mbslib
import multiprocessing
import functools

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
    # calc_theta_l
    l = sum([sum(i) for i in int_site_list])
    thetal = l / (nsam - 1)
    #calc_theta_pi/theta_h
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


def run_mbs_calc_SFSbased_statistics(params, data_dir, save_dir, pop_name):
    # path to trajectory
    path_to_traj = f"{pop_name}/traj_tmutation{params['t_mutation_in_year']}_s{params['s']}_{pop_name}"

    # generate trajectory
    make_trajectoryfiles_forward(params['N0'], params['generation'],
                                  params['demography_in_year'], params['t_mutation_in_year'],
                                  params['s'], params['h'], params['resolution'],
                                  params['n_trajectory'], path_to_traj)
    # path to mbsoutput
    path_to_mbs_output = f"{data_dir}/mbs_nsam{params['nsam']}_tmutation{params['t_mutation_in_year']}_s{params['s']}_{pop_name}.dat"

    # run mbs
    run_mbs(params['nsam'], params['per_site_theta'], params['per_site_rho'],
            params['lsites'], params['selpos'],
            params['n_trajectory'], params['nrep_per_traj'],
            path_to_mbs_output, path_to_traj)

    n = params['nsam']
    theta_list = [calc_thetapi_S_thetaw_thetal(m['seq'], m['pos']) for m in mbslib.parse_mbs_data(path_to_mbs_output)]
    D_list = [calc_D(*i, n) for i in theta_list]
    H_list = [calc_H(*i, n) for i in theta_list]

    with open("{}/theta_tmutation{}_s{}_popname{}.csv".format(save_dir, params["t_mutation_in_year"], params["s"], pop_name),
              "w", encoding="Shift_jis") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(['theta_pi', 'S', 'theta_w', 'theta_l', 'theta_h'])
        writer.writerows(theta_list) 
        
    statistics_list = []
    statistics_list.append(D_list)
    statistics_list.append(H_list)
    statistics_list = np.array(statistics_list).T.tolist()
    #print(statistics_list)
    with open("{}/statistics_tmutation{}_s{}_popname{}.csv"
              .format(params["t_mutation_in_year"], params["s"], pop_name),
              "w", encoding="Shift_jis") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(['D', 'H'])
        writer.writerows(statistics_list) 

    print(params["t_mutation_in_year"], params["s"], pop_name, "done")


def parameter_sets(N0, gen, demo, nrep, mutation_ages, sel_advantages):

    params = dict()

    params['N0'] = N0
    params['generation'] = gen
    params['demography_in_year'] = demo

    # selection coefficients
    params['s'] = 0
    params['h'] = 0.5  # <--- co-dominance
    params['resolution'] = 100

    # number of trajectory
    params['n_trajectory'] = nrep
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

    params_list = list()
    for t_mutation_in_year in mutation_ages:
        params['t_mutation_in_year'] = t_mutation_in_year
        for s in sel_advantages:
            params['s'] = s
            params_list.append(params.copy())

    return params_list


def calc_power(path_to_percentile, data_dir, sel_advantages, t_list, pop_name, nrep, power_dir, file_name):
    df = pd.read_csv(path_to_percentile)
    D_thres = df['5'][0]
    H_thres = df['5'][1]

    # calc power
    D_power_list = []
    H_power_list = []
    for s in sel_advantages:
        temp_D = []
        temp_H = []
        for t in t_list:
            df = pd.read_csv(
                "{}/statistics_tmutation{}_s{}_popname{}.csv"
                    .format(data_dir, t, s, pop_name))
            D_power = sum([i < D_thres for i in df["D"]])/nrep
            H_power = sum([i < H_thres for i in df["H"]])/nrep
            temp_D.append(D_power)
            temp_H.append(H_power)
        D_power_list.append(temp_D)
        H_power_list.append(temp_H)

    # save power
    df_D = pd.DataFrame(D_power_list, index = sel_advantages, columns = t_list)
    df_H = pd.DataFrame(H_power_list, index = sel_advantages, columns = t_list)
    df_D.to_csv('{}/D_{}'.format(power_dir, file_name))
    df_H.to_csv('{}/H_{}'.format(power_dir, file_name))


def calc_power_of_sfsbased_test(statistics_data, t_list, s_list, threshold_list, data_dir, n_run):
    """

    Args:
        f_list:
        s_list:
        threshold_list: 5 percentile threshold of each of test statistics, [D, H, E]
        data_dir:
        n_run: number of trajectories
        n0: population size

    Returns:

    """
    # calc power of D
    significant_D = [sum(np.array(i[0]) <= threshold_list[0])/n_run
                   for i in statistics_data]
    power_D = np.reshape(significant_D, (len(t_list), len(s_list)))
    df_power_D = pd.DataFrame(power_D, index=t_list, columns=s_list)
    df_power_D.to_csv("{}/power_of_D_forward.csv".format(data_dir))

    # calc power of H
    significant_H = [sum(np.array(i[1]) <= threshold_list[1]) / n_run
                     for i in statistics_data]
    power_H = np.reshape(significant_H, (len(t_list), len(s_list)))
    df_power_H = pd.DataFrame(power_H, index=t_list, columns=s_list)
    df_power_H.to_csv("{}/power_of_H_forward.csv".format(data_dir))


def main():
    mutation_ages = [3000, 4000, 5000, 6000, 7000, 8000, 9000,
                     10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000,
                     100000, 200000, 300000, 40000, 500000]
    sel_advantages = [0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
                      0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                      0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                      0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    # generation in year
    gen = 25
    # pop size
    N0 = 20000
    N1 = 2000
    N2 = 14000
    N3 = 10000

    # demographic event timing in year
    t1 = 400 * gen
    t2 = 2800 * gen
    t3 = 10000 * gen
    demo = [[0, t1, N0],
             [t1, t2, N1],
             [t2, t3, N2],
             [t3, 100 * N0 * gen, N3]
             ]
    # number of replication
    nrep = 10

    testlist = parameter_sets(N0, gen, demo, nrep, mutation_ages, sel_advantages)
    pop_name = 'euasian'
    # simulated data directory name
    data_dir = 'sfs_data'
    # statistics data directory
    save_dir = 'sfs_data'
    n_cpu = int(multiprocessing.cpu_count() / 2)
    with multiprocessing.Pool(processes=n_cpu) as p:
        p.map(functools.partial(run_mbs_calc_SFSbased_statistics, data_dir,  save_dir, pop_name), testlist)

    # power data directory name
    power_dir = 'power'
    # path to percentile
    path_to_percentile = 'percentile/percentile_popname{}.csv'.format(pop_name)
    # power file neme
    file_name = 'power_popname{}.csv'.format(pop_name)
    calc_power(path_to_percentile, data_dir, sel_advantages, mutation_ages, pop_name, nrep, power_dir, file_name)


if __name__=="__main__":
    main()



