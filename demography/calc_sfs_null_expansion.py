import numpy as np
import pandas as pd
from my_module import mstools


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
    # 変異の数を計算
    l = sum([sum(i) for i in int_site_list])

    # 変異の平均個数
    thetal = l / (nsam - 1)

    ##calc_theta_pi/theta_h##
    # サイトごとの相違数の計算とその和
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
    # Sの計算
    S = len(ms_pos_data)

    # aとthetaの計算
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


def ms_expansion(nsam, nreps, theta, r_rate, nsite, e_age, e_strength, path_to_data):
    '''

    Args:
        nsam: number of chromosomes
        nreps: number of replication
        theta: mutation rate per generation over locus
        r_rate: recombination rate per generation over locus
        nsite: length of sequence
        e_age: timing of expansion in unite of 4N generation
        e_strength: N1/N0, where N0 is current population size and N1 is population size before expansion
        path_to_data:

    Returns:

    '''
    cmd = "ms {} {} -t {} -r {} {} -eN {} {} > {}"\
        .format(nsam, nreps, theta, r_rate, nsite, e_age, e_strength, path_to_data)
    mstools.run_command(cmd)


def calc_percentile(path_to_data, n, nreps, theta, rms, l, t, k, percentile_dir, t_in_gen):
    '''

    Args:
        n: the number of chromsomes
        nreps: the number of replication
        theta: 4N0 * mutation rate per generation over locus
        rms: 4N0 * recombination rate per generation over locus
        l: length of sequence
        t: timing of expansion in 4N0 generation
        k: N1/N0, where N0 is current population size and N1 is population size before expansion

    Returns:

    '''
    # run ms
    ms_expansion(n, nreps, theta, rms, l, t, k, path_to_data)

    # calc statistics
    theta_list = [calc_thetapi_S_thetaw_thetal(m['seq'], m['pos']) for m in mstools.parse_ms_data(path_to_data)]
    D_list = [calc_D(*i, n) for i in theta_list]
    H_list = [calc_H(*i, n) for i in theta_list]

    # calc percentile
    bins = np.arange(1, 100, 1)
    D_percentile = np.nanpercentile(D_list, bins)
    H_percentile = np.nanpercentile(H_list, bins)
    percentile_list = []
    percentile_list.append(D_percentile)
    percentile_list.append(H_percentile)

    df = pd.DataFrame(percentile_list, columns=bins, index=["D", "H"])
    df.to_csv("{}/percentile_sfs_expansion_eage{}_estrength{}.csv".format(percentile_dir, t_in_gen, int(1 / k)))


def main():
    # initial vales
    # N1/N0
    k = 0.1
    # population size before expansion
    N1 = 4770
    # current population size
    N0 = N1 / k
    # timing of expansion in generation
    t_in_gen = 500
    # timing of expansion in unit of 4N0 generation
    t = t_in_gen / (4 * N0)
    # mutation rate per site per generation
    μ = 1.0 * 10 ** (-8)
    # recombination rate per site per generation
    r = 1.0 * 10 ** (-8)
    # length of sequence
    l = 10000
    # mutation rate over locus
    theta = 4 * N0 * μ * l
    # recombination rate over locus
    rms = 4 * N0 * r * l
    # number of chromosomes
    n = 120
    # number of replication
    nreps = 1000
    percentile_dir = "percentile"
    path_to_data = 'results/ms_data.txt'
    calc_percentile(path_to_data, n, nreps, theta, rms, l, t, k, percentile_dir, t_in_gen)


if __name__=="__main__":
    main()
