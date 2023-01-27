import numpy as np
import pandas as pd
import csv
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


def ms_bottleneck(nsam, nreps, theta, r_rate, l, b_end, b_strength, b_start, e_strength, path_to_data):
    '''

    Args:
        nsam: number of chromosomes
        nreps: number of replications
        theta: mutation rate per generation over locus
        r_rate: recombination rate per generation over locus
        l: length of sequence
        b_end: timing of bottleneck ended in 4N0 generation
        b_strength: N1/N0, where N0 is current population size and N1 is population size during bottleneck
        b_start: timing of bottleneck started in 4N0 generation
        e_strength: N2/N0, where N0 is current population size and N2 is population size before bottleneck

    Returns:

    '''
    cmd = "ms {} {} -t {} -r {} {} -eN {} {} -eN {} {} > {}"\
        .format(nsam, nreps, theta, r_rate, l, b_end,
                b_strength, b_start, e_strength, path_to_data)
    mstools.run_command(cmd)


def calc_percentile(nsam, nreps, theta, r_rate, nsite, b_end, b_strength, b_start,
                    e_strength, t_end, t_duration, path_to_data, percentile_dir):
    '''

    Args:
        n: the number of chromsomes
        nreps: the number of replication
        theta: 4N0 * mutation rate per generation over locus
        r_rate: 4N0 * recombination rate per generation over locus
        l: length of sequence
        b_end: timing of bottleneck ended in 4N0 generation
        b_strength: N1/N0, where N0 is current population size and N1 is population size during bottleneck
        b_start: timing of bottleneck started in 4N0 generation
        e_strength: N2/N0, where N0 is current population size and N2 is population size before bottleneck
        t_end: timing of bottleneck ended in generation
        t_duration: duration of bottleneck in generation

    Returns:

    '''
    # run ms
    ms_bottleneck(nsam, nreps, theta, r_rate, nsite, b_end, b_strength, b_start, e_strength, path_to_data)

    # calc statistics
    theta_list = [calc_thetapi_S_thetaw_thetal(m['seq'], m['pos']) for m in mstools.parse_ms_data(path_to_data)]
    D_list = [calc_D(*i, nsam) for i in theta_list]
    H_list = [calc_H(*i, nsam) for i in theta_list]

    # calc percentile
    bins = np.arange(1, 100, 1)
    D_percentile = np.nanpercentile(D_list, bins)
    H_percentile = np.nanpercentile(H_list, bins)
    percentile_list = []
    percentile_list.append(D_percentile)
    percentile_list.append(H_percentile)

    df = pd.DataFrame(percentile_list, columns=bins, index=["D", "H"])
    df.to_csv("{}/percentile_sfs_bottleneck_bage{}_bduration{}_bstrength{}.csv"
              .format(percentile_dir, t_end, t_duration, int(1/b_strength)))


def main():
    # initial vales
    # population size during bottleneck
    N1 = 418
    # N1/N0, where N0 is current population size
    b_strength = 0.05
    # current population size
    N0 = int(N1/b_strength)
    # timing of bottleneck ended in generation
    t_end = 1000
    # duration of bottleneck in generation
    t_duration = 500
    # timing of bottleneck in 4N0 generation
    b_end = int(t_end/(4*N0))
    # duration of bottleneck in 4N0 generation
    b_duration = int(t_duration/(4*N0))
    # timing of bottleneck started in 4N0 generation
    b_start = b_end + b_duration
    # N2/N0, where N2 is population size before bottleneck
    e_strength = 1
    # mutation rate per site per generation
    μ = 1.0 * 10 ** (-8)
    # recombination rate per site per generation
    r = 1.0 * 10 ** (-8)
    # length of sequence
    l = 10000
    # mutation rate over locus
    theta = 4*N0*μ*l
    # recombination rate over locus
    r_rate = 4*N0*r*l
    # number of chromosomes
    nsam = 120
    # number of replication
    nreps = 1000
    percentile_dir = "percentile"
    path_to_data = 'results/ms_data.txt'
    calc_percentile(nsam, nreps, theta, r_rate, l, b_end, b_strength, b_start,
                    e_strength, t_end, t_duration, path_to_data, percentile_dir)


if __name__=="__main__":
    main()