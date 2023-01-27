import numpy as np
import pandas as pd
import csv
import multiprocessing
from my_module import forward_trajectory as fwd
from my_module import mbslib
import functools


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


def mbs2msoutput(mbs_input_file, ms_output_file, nsam, n_traj,
                 per_site_theta, per_site_rho, selpos,lsites):

    # calc_EHH
    with open(ms_output_file, 'w') as f:
        # convert into ms format for each line
        f.write("ms {} {} -t {} -r {} {}\n\n".format(nsam, n_traj,
                                                     per_site_theta * lsites,
                                                     per_site_rho * lsites, lsites))

        # change the position of mutation if it occurred at target site
        for i in mbslib.parse_mbs_data(mbs_input_file):
            # change the position of mutation if it occurred at target site
            if i['pos'][0] == 1.0:
                h = mbslib.mbs_to_ms_output(i, selpos, lsites)
                # print(h['seq'])
                f.write("//\n")
                # write segregation sites
                f.write("segsites: {}\n".format(len(h['pos'])))

                # write position
                f.write("positions: ")
                # convert int to str
                pos_list = [str(i) for i in h['pos']]
                # change position of the mutation occurred at the target site
                pos_list[1] = str(2 / lsites)
                f.write(" ".join(pos_list))
                f.write("\n")

                # write seq data
                f.write("\n".join(h["seq"]))
                f.write("\n\n")
                pass

            else:
                h = mbslib.mbs_to_ms_output(i, selpos, lsites)
                # print(h['seq'])
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
    params['lsites'] = 500000
    # position of target site
    params['selpos'] = 1

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


def calc_stat(params, data_dir, pop_name, ehh_data_dir, distance_in_bp):
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

    # rewrite mbs to ms
    ms_output_file = '{}/mbs_t0{}_p0{}_s{}.txt'.format(data_dir, params['t0_in_year'],params['p0'], params['s'])
    mbs2msoutput(path_to_mbs_output, ms_output_file, params['nsam'], params['n_trajectory'],
                 params['per_site_theta'], params['per_site_rho'], params['selpos'],params['lsites'])

    mbslib.run_command('Rscript calc_EHH_given_t_p.R {} {} {} {} {} {} {} {} {}'.format(params['t0_in_year'],
                                                                            params['p0'],
                                                                            params['n_trajectory'],
                                                                            params['lsites'],
                                                                            params['s'],
                                                                            pop_name,
                                                                            distance_in_bp,
                                                                            data_dir,
                                                                            ehh_data_dir))
    print(params["t0_in_year"], params['p0'], params["s"], "done")


def calc_power(t_mutation_in_year, initial_freq,  s, pop_name, ehh_data_dir, path_to_percentile , power_dir, n_run):

    rEHH_power_list = []
    iHS_power_list = []

    # calc rEHH power
    rEHH_percentile = path_to_percentile + '/rEHH_percentile_popname{}'.format(pop_name)
    df = pd.read_csv(rEHH_percentile, index_col=0, header=0)
    threshold = df['95']
    labels = ['0.01', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45',
              '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '0.99']
    bins = [0, 0.025, 0.075, 0.125, 0.175, 0.225, 0.275, 0.325, 0.375, 0.425,
            0.475, 0.525, 0.575, 0.625, 0.675, 0.725, 0.775, 0.825, 0.875,
            0.925, 0.975, 1.0]
    for t in t_mutation_in_year:
        df = pd.read_csv('{}/EHH_data_t0{}_p0{}_s{}_popname{} .csv'
                               .format(ehh_data_dir, t, initial_freq, s, pop_name))
        df['f_current_bin'] = pd.cut(df['f_current'], bins, labels=labels, right=False)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df[df['iHH_A'] != 0]
        df = df[df['iHH_D'] != 0]
        count = 0
        for m, n in zip(labels, threshold):
            count = count + sum([i > n for i in df[df['f_current_bin'] == m]['rEHH']])
        rEHH_power_list.append(count/n_run)
    rEHH_power_df = pd.DataFrame(rEHH_power_list, columns=[pop_name], index=t_mutation_in_year)
    rEHH_power_df.to_csv("{}/rEHH_power_p0{}_popname{}.csv"
                         .format(power_dir, initial_freq, pop_name))

    # iHS
    iHS_percentile = path_to_percentile + '/iHS_percentile_popname{}'.format(pop_name)
    df = pd.read_csv(iHS_percentile, index_col=0, header=0)
    threshold = df['5']
    for t in t_mutation_in_year:
        df = pd.read_csv('{}/EHH_data_t0{}_p0{}_s{}_popname{} .csv'
                         .format(ehh_data_dir, t, initial_freq, s, pop_name))
        df['f_current_bin'] = pd.cut(df['f_current'], bins, labels=labels, right=False)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df[df['iHH_A'] != 0]
        df = df[df['iHH_D'] != 0]
        count = 0
        for m, n in zip(labels, threshold):
            count = count + sum([i < n for i in df[df['f_current_bin'] == m]['iHS']])
        iHS_power_list.append(count/n_run)
    iHS_power_df = pd.DataFrame(rEHH_power_list, columns=[pop_name], index=t_mutation_in_year)
    iHS_power_df.to_csv("{}/iHS_power_p0{}_popname{}_forward.csv"
                        .format(power_dir, initial_freq, pop_name))



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
    ehh_data_dir = 'ehh_data'
    # the point at which EHH values are calculated
    distance_in_bp = 20000
    pop_name = 'asian'
    testlist = parameter_sets_given_t_p(N0, demo, nrep, t0_ages, sel_advantages, initial_freq)

    n_cpu = int(multiprocessing.cpu_count() / 2)
    with multiprocessing.Pool(processes=n_cpu) as p:
        p.map(functools.partial(calc_stat, data_dir = data_dir, ehh_data_dir = ehh_data_dir,
                                distance_in_bp = distance_in_bp, pop_name = pop_name), testlist)


if __name__=="__main__":
    main()
