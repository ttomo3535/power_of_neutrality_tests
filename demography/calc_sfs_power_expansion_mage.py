import numpy as np
import pandas as pd
import csv
from my_module import forward_trajectory as fwd
import multiprocessing
import functools
from my_module import mbslib


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


def run_mbs_calc_SFSbased_statistics_expansion(params, data_dir, save_dir):
    # expansion parameter
    # time_expansion_in_year
    e_in_year = int(params['demography_in_year'][1][0])
    # timing of expansion in generation
    t_e = int(e_in_year/params['generation'])
    # expansion_strength
    e_strength = int(params['demography_in_year'][0][2] / params['demography_in_year'][1][2])

    # path to trajectory file
    path_to_traj = f"{data_dir}/traj_tmutation{params['t_mutation_in_year']}_s{params['s']}_eage{t_e}_estrength{e_strength}"

    # generate trajectory file
    make_trajectoryfiles_forward(params['N0'], params['generation'],
                                 params['demography_in_year'], params['t_mutation_in_year'],
                                 params['s'], params['h'], params['resolution'],
                                 params['n_trajectory'], path_to_traj)

    # mbs output
    path_to_mbs_output = f"{data_dir}/mbs_nsam{params['nsam']}_tmutation{params['t_mutation_in_year']}_s{params['s']}" \
                         f"_eage{t_e}_estrength{e_strength}.dat"

    # run mbs
    run_mbs(params['nsam'], params['per_site_theta'], params['per_site_rho'],
            params['lsites'], params['selpos'],
            params['n_trajectory'], params['nrep_per_traj'],
            path_to_mbs_output, path_to_traj)

    n = params['nsam']
    theta_list = [calc_thetapi_S_thetaw_thetal(m['seq'], m['pos']) for m in mbslib.parse_mbs_data(path_to_mbs_output)]
    D_list = [calc_D(*i, n) for i in theta_list]
    H_list = [calc_H(*i, n) for i in theta_list]

    with open("{}/theta_tmutation{}_s{}_eage{}_estrength{}.csv"
                      .format(save_dir, params["t_mutation_in_year"], params["s"], t_e, e_strength),
              "w", encoding="Shift_jis") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(['theta_pi', 'S', 'theta_w', 'theta_l', 'theta_h'])
        writer.writerows(theta_list)

    statistics_list = []
    statistics_list.append(D_list)
    statistics_list.append(H_list)
    statistics_list = np.array(statistics_list).T.tolist()

    with open("{}/statistics_tmutation{}_s{}_eage{}_estrength{}.csv"
                      .format(save_dir, params["t_mutation_in_year"], params["s"], t_e, e_strength),
              "w", encoding="Shift_jis") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(['D', 'H', 'E'])
        writer.writerows(statistics_list)

    # extract current derived allele frequency
    freq_list = []
    for i in range(params['n_trajectory']):
        df = pd.read_table('{}_{}.dat'.format(path_to_traj, i), header=None)
        freq = df.iat[0, 3]
        temp_list = []
        temp_list.append(freq)
        freq_list.append(temp_list)

    with open("{}/freq_tmutation{}_s{}_eage{}_estrength{}.csv"
                      .format(save_dir, params["t_mutation_in_year"], params["s"],
                              t_e, e_strength),
              "w", encoding="Shift_jis") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(['freq'])
        writer.writerows(freq_list)

    # calc SFS
    sfs_all_list = []
    for m in mbslib.parse_mbs_data(path_to_mbs_output):
        seq = m['seq']
        int_site_list = [[int(i) for i in list(j)] for j in seq]
        sfs_all = [sum(i) for i in zip(*int_site_list)]
        # print(sfs_all)
        sfs_all_list.append(sfs_all)

    np.save('{}/SFS_tmutation{}_s{}_eage{}_estrength{}.npy'
            .format(save_dir, params["t_mutation_in_year"], params["s"], t_e, e_strength),
            sfs_all_list)

    print(params["t_mutation_in_year"], params["s"], t_e, e_strength, "done")


def calc_power(path_to_percentile, data_dir, s, t_list, e_t, k, nrep, power_dir, file_name):
    df = pd.read_csv(path_to_percentile)
    D_thres = df['5'][0]
    H_thres = df['5'][1]

    # calc power
    D_power_list = []
    H_power_list = []
    for t in t_list:
        df = pd.read_csv(
            "{}/statistics_tmutation{}_s{}_eage{}_estrength{}.csv".format(data_dir, t, s, e_t, k))
        D_power = sum([i < D_thres for i in df["D"]]) / nrep
        H_power = sum([i < H_thres for i in df["H"]]) / nrep
        D_power_list.append(D_power)
        H_power_list.append(H_power)

    # save power
    df_D = pd.DataFrame([D_power_list], index = ['{}'.format(e_t)], columns = t_list)
    df_H = pd.DataFrame([H_power_list], index = ['{}'.format(e_t)], columns = t_list)
    df_D.to_csv('{}/D_{}'.format(power_dir, file_name))
    df_H.to_csv('{}/H_{}'.format(power_dir, file_name))


def parameter_sets_forward_expansion(k, N1, t, nrep, mutation_ages, sel_advantages):
    '''

    Args:
        k: N1/N0, where N0 is current population size and N1 is poplation size before expansion
        N1: population size before expansion
        t: timing of expansion in generation
        nrep: nuber of replication
        mutation_ages:
        sel_advantages:

    Returns:

    '''
    params = dict()

    params['N0'] = N1/k
    params['generation'] = 20

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

    ###demography_parameter
    #demography_parameter_in_year =
    #[0, expansion in year, N0],
    #[expansion in year, 100*N0, N1]
    # time_expansion_in_year
    e_in_year = t*params['generation']

    params['demography_in_year'] = [[0, e_in_year, params['N0']],
                                   [e_in_year, 100*params['N0']*params['generation'], N1]
                                   ]
    params_list = list()
    for t_mutation_in_year in mutation_ages:
        params['t_mutation_in_year'] = t_mutation_in_year
        for s in sel_advantages:
            params['s'] = s
            params_list.append(params.copy())

    return params_list


def main():
    # initial values
    # in units of year
    t_list = np.arange(6000, 80000, 2000)
    # selection
    sel_advantages = [0.005]
    # N1/N0
    k = 0.1
    # N1 population size before expansion
    N1 = 4770
    # timing of expansion in generation
    t = 500
    # nuber of replication
    nrep = 10
    # parameters sets
    testlist = parameter_sets_forward_expansion(k, N1, t, nrep, t_list, sel_advantages)
    # simulated data directory name
    data_dir = 'results'
    # statistics data directory name
    save_dir = 'sfs_data'

    # run mbs
    n_cpu = int(multiprocessing.cpu_count() / 2)
    with multiprocessing.Pool(processes=n_cpu) as p:
        p.map(functools.partial(run_mbs_calc_SFSbased_statistics_expansion,
                                      data_dir=data_dir, save_dir=save_dir), testlist)

    # power data directory name
    power_dir = 'power'
    # path_to_percentile_data
    path_to_percentile = 'percentile/percentile_eage{}_estrength{}.csv'.format(t, int(1/k))
    # power file neme
    file_name = 'power_eage{}_estrength{}_nullex.csv'.format(t, int(1/k))
    calc_power(path_to_percentile, save_dir, sel_advantages[0], t_list, t, int(1/k), nrep, power_dir, file_name)


if __name__=="__main__":
    main()