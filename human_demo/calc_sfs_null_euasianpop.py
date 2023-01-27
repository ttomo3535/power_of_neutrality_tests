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
    # calc_theta_l
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
    # calc_theta_w
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


def calc_percentile(path_to_data, n, percentile_dir, pop_name):
    '''

    Args:
        n: the number of chromsomes

    Returns:

    '''
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

    df = pd.DataFrame(percentile_list, columns = bins, index= ["D", "H"])
    df.to_csv("{}/percentile_sfs_{}_pop.csv".format(percentile_dir, pop_name))


def main():
    #### parameter sets
    # pop size in unite of N0
    N0 = 20000
    N1 = 2000 / N0
    N2 = 14000 / N0
    N3 = 10000 / N0

    # demographic event timing in unit of 4N0 generation
    t1 = 400 / (4 * N0)
    t2 = 2800 / (4 * N0)
    t3 = 10000 / (4 * N0)

    # mutation rate per site per gen
    μ = 1.0 * 10 ** (-8)
    # recombination rate per site per gen
    r = 1.0 * 10 ** (-8)
    # length
    l = 10000
    # theta per region per gen
    theta = 4 * N0 * μ * l
    # ro per region per gen
    r4ms = 4 * N0 * r * l

    # sampling parameter
    # the number of sampling
    nsam = 120
    # the number of replication
    nrep = 1000

    # file name
    pop_name = 'euasian'
    path_to_data = 'results/{}_msdata.txt'.format(pop_name)
    percentile_dir = 'percentile'

    # run ms
    cmd = 'ms {} {} -t {} -r {} {} -eN {} {} -eN {} {} -eN {} {} > {}'.format(nsam, nrep, theta, r4ms, l, t1, N1, t2,
                                                                              N2, t3, N3, path_to_data)
    # print(cmd)
    mstools.run_command(cmd)
    # calculate percentile
    calc_percentile(path_to_data, nsam, percentile_dir, pop_name)


if __name__=="__main__":
    main()