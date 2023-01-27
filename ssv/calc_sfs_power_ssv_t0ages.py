import csv
import numpy as np
import multiprocessing
import functools
from my_module import mbslib
import pandas as pd
from my_module import trajectory_given_t_and_p as trj


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


def parameter_sets_given_t_p(t0_ages, sel_advantages, initial_freq, nrep):
    params = dict()

    params['N0'] = 5000
    params['generation'] = 20
    params['demography_in_year'] = [[0, 100 * params['N0'] * params['generation'], params['N0']]]

    # selection coefficients
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
    params['selpos'] = int(params['lsites'] / 2)

    # mutation rate per site per generation
    params['per_site_theta'] = 1.0 * 10 ** (-8) * 4 * params['N0']
    # recombination rate per site per generation
    params['per_site_rho'] = 1.0 * 10 ** (-8) * 4 * params['N0']

    # derived allele frequency at which selection started acting
    params['p0'] = initial_freq

    params_list = list()
    for t in t0_ages:
        # time selection started in year
        params['t0_in_year'] = t
        # time selection started in unit of 4N generation
        params['t0'] = params['t0_in_year'] / (4 * params['N0'] * params['generation'])
        for s in sel_advantages:
            params['s'] = s
            params['selection_in_year'] = [[0, params['t0_in_year'], params['s'], params['h']]]
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


def calc_stat(params, data_dir, save_dir):
    # path to trajectory
    path_to_traj = f"{data_dir}/traj_t0{params['t0_in_year']}_p0{params['p0']}_s{params['s']}"
    # mbs output
    path_to_mbs_output = f"{data_dir}/mbs_nsam{params['nsam']}_t0{params['t0_in_year']}_p0{params['p0']}_s{params['s']}.dat"

    # generate trajectory
    make_tarjctoyfiles_given_t_p(params['N0'], params['t0'], params['p0'], params['generation'],
                                 params['demography_in_year'], params['selection_in_year'], params['resolution'],
                                 params['n_trajectory'], path_to_traj)

    # run mbs to make SNP data
    run_mbs(params['nsam'], params['per_site_theta'], params['per_site_rho'],
            params['lsites'], params['selpos'],
            params['n_trajectory'], params['nrep_per_traj'],
            path_to_mbs_output, path_to_traj)

    # output
    n = params['nsam']
    theta_list = [calc_thetapi_S_thetaw_thetal(m['seq'], m['pos']) for m in mbslib.parse_mbs_data(path_to_mbs_output)]
    D_list = [calc_D(*i, n) for i in theta_list]
    H_list = [calc_H(*i, n) for i in theta_list]

    with open("{}/theta_t0{}_p0{}_s{}.csv".format(save_dir, params["t0_in_year"], params["p0"], params['s']),
              "w", encoding="Shift_jis") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(['theta_pi', 'S', 'theta_w', 'theta_l', 'theta_h'])
        writer.writerows(theta_list)

    statistics_list = []
    statistics_list.append(D_list)
    statistics_list.append(H_list)
    statistics_list = np.array(statistics_list).T.tolist()
    with open("{}/statistics_t0{}_p0{}_s{}.csv"
                      .format(save_dir, params["t0_in_year"], params['p0'], params["s"]),
              "w", encoding="Shift_jis") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(['D', 'H'])
        writer.writerows(statistics_list)

    # get derived frequency
    freq_list = []
    for i in range(params['n_trajectory']):
        df = pd.read_table('{}_{}.dat'.format(path_to_traj, i), header=None)
        freq = df.iat[0, 3]
        temp_list = []
        temp_list.append(freq)
        freq_list.append(temp_list)

    with open("{}/freq_t0{}_p0{}_s{}.csv"
                      .format(save_dir, params["t0_in_year"], params['p0'] ,params["s"]),
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

    # print(sfs_all_list)
    np.save('{}/SFS_t0{}_p0{}_s{}.npy'
            .format(save_dir, params["t0_in_year"], params['p0'], params["s"]),
            sfs_all_list)

    print(params["t0_in_year"], params['p0'], params["s"], "done")


def calc_power(path_to_percentile, data_dir, s, t0_list, p, nrep, power_dir, file_name):
    df = pd.read_csv(path_to_percentile)
    D_thres = df['5'][0]
    H_thres = df['5'][1]

    # calc power
    D_power_list = []
    H_power_list = []
    for t0 in t0_list:
        df = pd.read_csv("{}/statistics_t0{}_p0{}_s{}.csv".format(data_dir, t0, p, s))
        D_power = sum([i < D_thres for i in df["D"]]) / nrep
        H_power = sum([i < H_thres for i in df["H"]]) / nrep
        D_power_list.append(D_power)
        H_power_list.append(H_power)

    # save power
    df_D = pd.DataFrame([D_power_list], index = ['{}'.format(p)], columns = t0_list)
    df_H = pd.DataFrame([H_power_list], index = ['{}'.format(p)], columns = t0_list)
    df_D.to_csv('{}/D_{}'.format(power_dir, file_name))
    df_H.to_csv('{}/H_{}'.format(power_dir, file_name))


def main():
    # selection coefficients
    sel_advantages = [0.005]
    # onset of selection in year
    t0_ages = np.arange(6000, 80000, 2000)
    # initial frequency
    initial_freq = 0.1
    # number of replication
    nrep = 2

    testlist = parameter_sets_given_t_p(t0_ages, sel_advantages, initial_freq, nrep)
    # simulated data directory name
    data_dir = 'results'
    # statistics data directory name
    save_dir = 'sfs_data'

    n_cpu = int(multiprocessing.cpu_count() / 2)
    with multiprocessing.Pool(processes=n_cpu) as p:
        p.map(functools.partial(calc_stat, data_dir=data_dir, save_dir=save_dir), testlist)

    # power data directory name
    power_dir = 'power'
    # path_to_percentile_data
    path_to_percentile = '../constant_pop/percentile_data/percentile_D_H_E.csv'
    # power file name
    file_name = 'power_p{}_s{}_t0ages.csv'.format(initial_freq, sel_advantages[0])
    calc_power(path_to_percentile, save_dir, sel_advantages[0], t0_ages, initial_freq, nrep, power_dir, file_name)


if __name__=="__main__":
    main()




