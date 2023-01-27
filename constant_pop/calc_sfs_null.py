import numpy as np
from my_module import mstools
import pandas as pd


def ms(nsam, nreps, theta, r_rate, nsite, filename):
    """ Run ms

    Args:
        nsam: number of chromosomes
        nreps: number of replication
        theta: population mutation parameter per region
        r_rate: population recombination parameter per region
        nsite: length of simulated region
    """
    cmd = "ms {} {} -t {} -r {} {} > {}".format(nsam, nreps, theta, r_rate, nsite, filename)
    mstools.run_command(cmd)  


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
    #number of sample
    nsam = len(ms_seq_data)

    #get sequence data in int
    int_site_list = [[int(i) for i in list(j)] for j in ms_seq_data]
    
    #calc theta_l
    #calc number of segregating sites
    l = sum([sum(i) for i in int_site_list])
    #calc average number of segregating sites
    thetal = l/(nsam - 1)

    #calc theta_pi, theta_h
    #calc sum of pairwise differences
    k = 0
    h = 0
    for i in zip(*int_site_list):
        der = sum(i)
        k += der*(nsam-der)
        h += der**2
    #clac_thetapi/h
    thetapi = k*2/(nsam*(nsam-1))
    thetah = h*2/(nsam*(nsam-1))

    #calc theta_w
    #calc number of segregating sites
    S = len(ms_pos_data)
    #calc theta_w
    a = 0
    for j in range(1,nsam):
        a += 1/j
    thetaw = S/a
    
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
    #calc variation of H
    a = sum([1/i for i in range(1, n)])
    b = sum([1/(i**2) for i in range(1, n)])
    v = (n-2)*thetaw/(6*(n-1)) + (18*n**2*(3*n+2)*(b+1/n**2)-(88*n**3+9*n**2-13*n+6))*(S*(S-1)/(a**2+b))/(9*n*(n-1)**2)
    #return nan if number of segregating sites is too small to calculate H
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
    #calc variation of D
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


def calc_percentile(nsam, nreps, theta, r, lsites, filename, data_dir):
    ms(nsam, nreps, theta, r, lsites, filename)
    #calc statistics
    #theta list: list of theta, [thetapi, S, thetalw, thetal, thetah]
    theta_list = [calc_thetapi_S_thetaw_thetal(m['seq'], m['pos'])
                  for m in mstools.parse_ms_data(filename)]
    D_list = [calc_D(*i, nsam) for i in theta_list]
    H_list = [calc_H(*i, nsam) for i in theta_list]

    #calc percentile
    bins = np.arange(1, 100, 1)
    D_percentile = np.nanpercentile(D_list, bins)
    H_percentile = np.nanpercentile(H_list, bins)

    percentile_list = []
    percentile_list.append(D_percentile)
    percentile_list.append(H_percentile)

    df_percentile = pd.DataFrame(percentile_list, columns=bins, index=["D", "H"])
    df_percentile.to_csv("{}/percentile_D_H.csv".format(data_dir))


def main():
    # initial vales
    nsam = 120
    nreps = 1000
    theta = 2
    r = 2
    lsites = 10000
    filename = "results/ms_data_r{}.txt".format(r)
    data_dir = "percentile_data"
    calc_percentile(nsam, nreps, theta, r, lsites, filename, data_dir)


if __name__=="__main__":
    main()
