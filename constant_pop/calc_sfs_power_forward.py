import csv
import numpy as np
import multiprocessing
import pandas as pd
import functools
from my_module import forward_trajectory as fwd
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
    generate trajectory files

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
    a = sum([1/i for i in range(1, n)])
    b = sum([1/(i**2) for i in range(1, n)])
    v = (n-2)*thetaw/(6*(n-1)) + (18*n**2*(3*n+2)*(b+1/n**2)-(88*n**3+9*n**2-13*n+6))*(S*(S-1)/(a**2+b))/(9*n*(n-1)**2)

    # return nan if number of segregating sites is too small to calculate H
    if v == 0:
        return np.nan
    else:
        H = (thetapi-thetal)/v**(1/2)
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
    for i in range(1,n):
        a1 += 1/i
    a2 = 0
    for i in range(1, n):
        a2 += 1/i**2
    b1 = (n + 1)/(3*(n-1))
    b2 = 2*(n**2 + n + 3)/(9*n*(n-1))
    c1 = b1 - 1/a1
    c2 = b2 - (n+2)/(a1*n) + a2/a1**2
    e1 = c1/a1
    e2 = c2/(a1**2 + a2)
    C = (e1*S + e2*S*(S-1))**0.5

    if C == 0:
        return np.nan
    else:
        D = (thetapi - thetaw)/C
        return D


def run_mbs_calc_sfsbased_statistics(params, data_dir):
    # path to trajectory file
    path_to_traj = f"results/traj_f{params['t_mutation_in_year']}s{params['s']}"

    # generate trajectory file
    make_trajectoryfiles_forward(params['N0'], params['generation'],
                                  params['demography_in_year'], params['t_mutation_in_year'],
                                  params['s'], params['h'], params['resolution'],
                                  params['n_trajectory'], path_to_traj)

    # mbs output file name
    path_to_mbs_output = f"results/mbs_nsam{params['nsam']}_fcurrent{params['t_mutation_in_year']}_s{params['s']}.dat"

    # run mbs
    run_mbs(params['nsam'], params['per_site_theta'], params['per_site_rho'],
            params['lsites'], params['selpos'],
            params['n_trajectory'], params['nrep_per_traj'],
            path_to_mbs_output, path_to_traj)

    n = params['nsam']
    # theta list: list of theta, [thetapi, S, thetalw, thetal, thetah]
    theta_list = [calc_thetapi_S_thetaw_thetal(m['seq'], m['pos']) for m in mbslib.parse_mbs_data(path_to_mbs_output)]
    D_list = [calc_D(*i, n) for i in theta_list]
    H_list = [calc_H(*i, n) for i in theta_list]

    with open("{}/theta_tmutation{}_s{}.csv".format(data_dir, params["t_mutation_in_year"], params["s"]),
              "w", encoding="Shift_jis") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(['theta_pi', 'S', 'theta_w', 'theta_l', 'theta_h'])
        writer.writerows(theta_list)

    # statistics list: list of test statistics, [D, H, E]
    statistics_list = []
    statistics_list.append(D_list)
    statistics_list.append(H_list)
    statistics_list = np.array(statistics_list).T.tolist()

    with open("{}/statistics_tmutation{}_s{}.csv".format(data_dir, params["t_mutation_in_year"], params["s"]),
              "w", encoding="Shift_jis") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(['D', 'H'])
        writer.writerows(statistics_list)

    print(params["t_mutation_in_year"], params["s"], "done")

    return D_list, H_list


def parameter_sets_forward(mutation_ages, sel_advantages, number_of_trajectory):
    params = dict()

    params['N0'] = 5000
    params['generation'] = 20
    params['demography_in_year'] = [[0, 100 * params['N0'] * params['generation'], params['N0']]]

    # selection coefficients
    params['s'] = 0
    params['h'] = 0.5  # <--- co-dominance
    params['resolution'] = 100

    # number of trajectory
    params['n_trajectory'] = number_of_trajectory
    # coalescent simulation per trajectory
    params['nrep_per_traj'] = 1

    # number of chromosome
    params['nsam'] = 120
    # length of sequence
    params['lsites'] = 10000
    # position of target site
    params['selpos'] = 2 / params['lsites']

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
    significant_D = [sum(np.array(i[0]) <= threshold_list[0]) / n_run
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
    # initial values
    # in units of year
    mutation_ages = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
                     10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000,
                     100000, 200000, 300000, 40000, 500000]

    # selection
    sel_advantages = [0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
                      0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                      0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                      0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # number of trajectory
    number_of_trajectory = 200
    # parameters sets
    testlist = parameter_sets_forward(mutation_ages, sel_advantages, number_of_trajectory)
    # simulated data directory name
    data_dir = 'sfs_data'
    # power data directory name
    power_dir = 'power'

    # thresholds list
    percentile_data_dir = 'percentile_data'
    df = pd.read_csv('{}/percentile_D_H.csv'.format(percentile_data_dir))
    D_thres = df['5'][0]
    H_thres = df['5'][1]
    threshold = [D_thres, H_thres]
    n_cpu = int(multiprocessing.cpu_count() / 2)
    with multiprocessing.Pool(processes=n_cpu) as p:
        res = p.map(functools.partial(run_mbs_calc_sfsbased_statistics, data_dir=data_dir), testlist)

    calc_power_of_sfsbased_test(statistics_data=res, t_list=mutation_ages,
                                s_list=sel_advantages, threshold_list=threshold, data_dir=power_dir,
                                n_run=number_of_trajectory)


if __name__=="__main__":
    main()






