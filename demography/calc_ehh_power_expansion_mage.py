import numpy as np
import pandas as pd
import csv
from my_module import forward_trajectory as fwd
from my_module import mbslib
import functools
import multiprocessing


def make_trajectoryfiles(N0, generation, demography_in_year, t_mutation_in_year, s, h, resolution,
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

    # number of trajectory file
    params['n_trajectory'] = nrep
    # coalescent simulation per trajectory
    params['nrep_per_traj'] = 1

    # number of chromosome
    params['nsam'] = 120
    # length of sequence
    params['lsites'] = 500000
    # target site position
    params['selpos'] = 1

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


def run_mbs_calc_EHH(params, data_dir, ehh_data_dir, distance_in_bp):
    # parameters set
    # expansion age in year
    e_age = int(params['demography_in_year'][0][1])
    # timing of expansion in generation
    t_e = int(e_age/params['generation'])
    # expansion_strength
    e_strength = int(params['demography_in_year'][0][2] / params['demography_in_year'][1][2])

    # path to trajectory file
    path_to_traj = f"{data_dir}/traj_t{params['t_mutation_in_year']}_s{params['s']}_eage{t_e}_estrength{e_strength}"
    # trajectoryの生成
    make_trajectoryfiles(params['N0'], params['generation'],
                         params['demography_in_year'], params['t_mutation_in_year'],
                         params['s'], params['h'], params['resolution'],
                         params['n_trajectory'], path_to_traj)

    # mbs output file
    path_to_mbs_output = f"{data_dir}/mbs_nsam{params['nsam']}_tmutation{params['t_mutation_in_year']}_s{params['s']}" \
                         f"_eage{t_e}_estrength{e_strength}.dat"

    # run mbs
    run_mbs(params['nsam'], params['per_site_theta'], params['per_site_rho'],
            params['lsites'], params['selpos'],
            params['n_trajectory'], params['nrep_per_traj'],
            path_to_mbs_output, path_to_traj)

    # convert mbs format to ms format
    ms_file = '{}/mbs_tmutation{}_s{}_eage{}_estrength{}.txt'.format(
        data_dir, params['t_mutation_in_year'], params['s'],
        t_e, e_strength)

    with open(ms_file, 'w') as f:
        # convert into ms format for each line
        f.write("ms {} {} -t {} -r {} {}\n\n".format(params['nsam'], params['n_trajectory'],
                                                     params['per_site_theta'] * params['lsites'],
                                                     params['per_site_rho'] * params['lsites'], params['lsites']))
        # change the position of mutation if it occurred at target site
        for i in mbslib.parse_mbs_data(path_to_mbs_output):
            # change the position of mutation if it occurred at target site
            if i['pos'][0] == 1.0:
                h = mbslib.mbs_to_ms_output(i, params['selpos'], params['lsites'])
                f.write("//\n")
                # write segregation sites
                f.write("segsites: {}\n".format(len(h['pos'])))

                # write position
                f.write("positions: ")
                # convert int to str
                pos_list = [str(i) for i in h['pos']]
                # change position of the mutation occurred at the target site
                pos_list[1] = str(2 / params['lsites'])
                f.write(" ".join(pos_list))
                f.write("\n")

                # write seq data
                f.write("\n".join(h["seq"]))
                f.write("\n\n")

            else:
                h = mbslib.mbs_to_ms_output(i, params['selpos'], params['lsites'])
                f.write("//\n")
                # write segregating sites
                f.write("segsites: {}\n".format(len(h['pos'])))

                # write position
                f.write("positions: ")
                # convert int to str
                pos_list = [str(i) for i in h['pos']]
                f.write(" ".join(pos_list))
                f.write("\n")

                # write seq
                f.write("\n".join(h["seq"]))
                f.write("\n\n")


    # run R to calculate EHH
    mbslib.run_command('Rscript calc_EHH_under_expansion.R {} {} {} {} {} {} {} {} {}'
                       .format(params['t_mutation_in_year'], params['n_trajectory'], params['lsites'],
                               params['s'], t_e, e_strength, distance_in_bp,
                               data_dir, ehh_data_dir))

    print(params['t_mutation_in_year'], params['s'], e_age, e_strength, "done")


def calc_power(t_mutation_in_year, s, t_e, k, ehh_data_dir, path_to_percentile , power_dir, n_run):

    rEHH_power_list = []
    iHS_power_list = []

    # calc rEHH power
    rEHH_percentile = path_to_percentile[:11] + 'rEHH_' + path_to_percentile[11:]
    df = pd.read_csv(rEHH_percentile, index_col=0, header=0)
    threshold = df['95']
    labels = ['0.01', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45',
              '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '0.99']
    bins = [0, 0.025, 0.075, 0.125, 0.175, 0.225, 0.275, 0.325, 0.375, 0.425,
            0.475, 0.525, 0.575, 0.625, 0.675, 0.725, 0.775, 0.825, 0.875,
            0.925, 0.975, 1.0]
    for t in t_mutation_in_year:
        df = pd.read_csv('{}/EHH_data_tmutation{},_s{}_eage{}_estrength{} .csv'
                               .format(ehh_data_dir, t, s, t_e, k))
        df['f_current_bin'] = pd.cut(df['f_current'], bins, labels=labels, right=False)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df[df['iHH_A'] != 0]
        df = df[df['iHH_D'] != 0]
        count = 0
        for m, n in zip(labels, threshold):
            count = count + sum([i > n for i in df[df['f_current_bin'] == m]['rEHH']])
        rEHH_power_list.append(count/n_run)
    rEHH_power_df = pd.DataFrame(rEHH_power_list, columns=t_e, index=t_mutation_in_year)
    rEHH_power_df.to_csv("{}/rEHH_power_rEHH_eage{}_estrength{}_forward.csv".format(power_dir, t_e, k))

    # iHS
    iHS_percentile = path_to_percentile[:11] + 'iHS_' + path_to_percentile[11:]
    df = pd.read_csv(iHS_percentile, index_col=0, header=0)
    threshold = df['5']
    for t in t_mutation_in_year:
        df = pd.read_csv('{}/EHH_data_tmutation{},_s{}_eage{}_estrength{} .csv'
                               .format(ehh_data_dir, t, s, t_e, k))
        df['f_current_bin'] = pd.cut(df['f_current'], bins, labels=labels, right=False)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df[df['iHH_A'] != 0]
        df = df[df['iHH_D'] != 0]
        count = 0
        for m, n in zip(labels, threshold):
            count = count + sum([i < n for i in df[df['f_current_bin'] == m]['iHS']])
        iHS_power_list.append(count/n_run)
    iHS_power_df = pd.DataFrame(rEHH_power_list, columns=t_e, index=t_mutation_in_year)
    iHS_power_df.to_csv("{}/iHS_power_eage{}_estrength{}_forward.csv".format(power_dir, t_e, k))


def main():
    ## parameters sets
    k = 0.1
    N1 = 3890
    # expansion age in generation
    t = 2500
    # in units of year
    mutation_ages = np.arange(6000, 80000, 2000)
    # selection coefficients
    sel_advantages = [0, 0.005]
    # number of replication
    nrep = 10
    # the point at which EHH values are calculated
    distance_in_bp = 20000
    # simulated data directory name
    data_dir = 'results'
    # statistics data directory name
    ehh_data_dir = 'ehh_data'
    testlist = parameter_sets_forward_expansion(k, N1, t, nrep, mutation_ages, sel_advantages)
    n_cpu = int(multiprocessing.cpu_count()/2)
    with multiprocessing.Pool(processes=n_cpu) as p:
        p.map(functools.partial(run_mbs_calc_EHH, data_dir = data_dir, ehh_data_dir = ehh_data_dir,
                                distance_in_bp = distance_in_bp), testlist)

    # power data directory name
    power_dir = 'power'
    # path_to_percentile_data
    path_to_percentile = 'percentile/percentil_eage{}_estrength{}.csv'.format(t, int(1 / k))
    # power file name
    # to calculate false positive rate, set s = 0
    calc_power(t_mutation_in_year = mutation_ages, s = sel_advantages[1],
               t_e = t, k = int(1/k), ehh_data_dir = ehh_data_dir, path_to_percentile = path_to_percentile,
               power_dir = power_dir, n_run = nrep)


if __name__=="__main__":
    main()
