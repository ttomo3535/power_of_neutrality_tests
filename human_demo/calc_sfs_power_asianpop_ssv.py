import csv
import numpy as np
import multiprocessing
from my_module import mbslib
from my_module import trajectory_given_t_and_p as trj
import functools
import pandas as pd


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
    ##calc_theta_pi/theta_h##
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

        filename = f'{path_to_traj}_{i}.dat'

        # generate trajectory
        dem_under_neutrality, history = trj.prepare_history(demography_in_year, selection_in_year, t0, N0, generation)
        trajectory = trj.generate_trajectory(dem_under_neutrality, history, t0, p0, N0, resolution, 'NOTLOST')
        #print(trajectory)

        with open(filename, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            for freq in trajectory:
                writer.writerow(freq)


def parameter_sets_given_t_p(N0, demo, nrep, t0_ages, sel_advantages, initial_freq):
    params = dict()

    params['N0'] = N0
    params['generation'] = 20
    params['demography_in_year'] = demo
    # selection coefficients
    params['s'] = 0
    params['h'] = 0.5  # <--- co-dominance
    params['resolution'] = 0.01

    # number of trajectory
    params['n_trajectory'] = nrep
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

    params_list = list()
    for t in t0_ages:
        # time selection started in year
        params['t0_in_year'] = t
        # time selection started in unit of 4N generation
        params['t0'] = params['t0_in_year'] / (4 * params['N0'] * params['generation'])
        for s in sel_advantages:
            params['s'] = s
            params['selection_in_year'] = [[0, params['t0_in_year'], params['s'], params['h']]]
            for p in initial_freq:
                params['p0'] = p
                params_list.append(params.copy())

    return params_list


def calc_stat(params, data_dir, save_dir, pop_name):
    # path to trajectory
    path_to_traj = f"{data_dir}/traj_t0{params['t0_in_year']}_p0{params['p0']}_s{params['s']}"
    # path to mbs output
    path_to_mbs_output = f"{data_dir}/mbs_nsam{params['nsam']}_t0{params['t0_in_year']}_p0{params['p0']}_s{params['s']}.dat"

    ### generate trajectory
    make_tarjctoyfiles_given_t_p(params['N0'], params['t0'], params['p0'], params['generation'],
                                 params['demography_in_year'], params['selection_in_year'], params['resolution'],
                                 params['n_trajectory'], path_to_traj)

    # run mbs to make SNP data
    run_mbs(params['nsam'], params['per_site_theta'], params['per_site_rho'],
            params['lsites'], params['selpos'],
            params['n_trajectory'], params['nrep_per_traj'],
            path_to_mbs_output, path_to_traj)


    n = params['nsam']
    theta_list = [calc_thetapi_S_thetaw_thetal(m['seq'], m['pos']) for m in mbslib.parse_mbs_data(path_to_mbs_output)]
    D_list = [calc_D(*i, n) for i in theta_list]
    H_list = [calc_H(*i, n) for i in theta_list]

    # h_list = np.array(h_list).T.tolist()
    # print(theta_list)

    with open("{}/theta_t0{}_p0{}_s{}_popname{}.csv"
                      .format(save_dir, params["t0_in_year"], params["p0"], params['s'], pop_name),
              "w", encoding="Shift_jis") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(['theta_pi', 'S', 'theta_w', 'theta_l', 'theta_h'])
        writer.writerows(theta_list)

    statistics_list = []
    statistics_list.append(D_list)
    statistics_list.append(H_list)
    statistics_list = np.array(statistics_list).T.tolist()
    # print(statistics_list)
    with open("{}/statistics_t0{}_p0{}_s{}_popname{}.csv"
                      .format(save_dir, params["t0_in_year"], params['p0'], params["s"], pop_name),
              "w", encoding="Shift_jis") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(['D', 'H'])
        writer.writerows(statistics_list)

    print(params["t0_in_year"], params['p0'], params["s"], pop_name, "done")


def calc_power(path_to_percentile, data_dir, s, t_list, initial_freq, pop_name, nrep, power_dir, file_name):
    df = pd.read_csv(path_to_percentile)
    D_thres = df['5'][0]
    H_thres = df['5'][1]

    # calc power
    D_power_list = []
    H_power_list = []
    for t in t_list:
        df = pd.read_csv(
            "{}/statistics_t0{}_p0{}_s{}_popname{}.csv"
                .format(data_dir, t, intial_freq, s, pop_name))
        D_power = sum([i < D_thres for i in df["D"]])/nrep
        H_power = sum([i < H_thres for i in df["H"]])/nrep
        D_power_list.append(D_power)
        H_power_list.append(H_power)

    # save power
    df_D = pd.DataFrame([D_power_list], index = [pop_name], columns = t_list)
    df_H = pd.DataFrame([H_power_list], index = [pop_name], columns = t_list)
    df_D.to_csv('{}/D_{}'.format(power_dir, file_name))
    df_H.to_csv('{}/H_{}'.format(power_dir, file_name))


def main():
    # generation in year
    gen = 20

    # pop size in unite of N0
    N0 = 100000
    N1 = 7700
    N2 = 500
    N3 = 7700
    N4 = 600
    N5 = 24000
    N6 = 12500
    # demographic event timing in year
    t1 = 400 * gen
    t2 = 2000 * gen
    t3 = 2100 * gen
    t4 = 3500 * gen
    t5 = 3600 * gen
    t6 = 17000 * gen

    sel_advantages = [0.005]
    # onset of selection in year
    t0_ages = [t3]
    initial_freq = [0.1, 1 / (2 * N0)]

    # selection coefficient
    demo = [[0, t1, N0],
            [t1, t2, N1],
            [t2, t3, N2],
            [t3, t4, N3],
            [t4, t5, N4],
            [t5, t6, N5],
            [t6, 100 * N0 * gen, N6]
            ]
    # nuber of  replication
    nrep = 20
    # mbs data directory name
    data_dir = 'results'
    # statistics data directory
    save_dir = 'sfs_data'
    pop_name = 'asian'
    testlist = parameter_sets_given_t_p(N0, demo, nrep, t0_ages, sel_advantages, initial_freq)

    n_cpu = int(multiprocessing.cpu_count() / 2)
    with multiprocessing.Pool(processes=n_cpu) as p:
        p.map(functools.partial(calc_stat, data_dir=data_dir, save_dir=save_dir, pop_name=pop_name), testlist)

    # power data directory name
    power_dir = 'power'
    # path_to_percentile_data
    path_to_percentile = 'percentile/percentile_popname{}.csv'.format(pop_name)
    # power file neme
    file_name = 'power_popname{}_ssv.csv'.format(pop_name)
    calc_power(path_to_percentile, save_dir, sel_advantages[0], t0_ages, initial_freq[0], pop_name, nrep,
               power_dir, file_name)


if __name__=="__main__":
    main()




