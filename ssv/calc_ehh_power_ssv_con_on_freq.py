import os
import csv
import numpy as np
import multiprocessing
import pandas as pd
import pyper
import functools
from my_module import mbslib
from my_module import trajectory_given_t_and_p as trj
import random


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
                 per_site_theta, per_site_rho, selpos, lsites):
    # generate file
    with open(ms_output_file, 'w') as f:
        # convert mbs format to ms format
        f.write("ms {} {} -t {} -r {} {}\n\n".format(nsam, n_traj,
                                                     per_site_theta * lsites,
                                                     per_site_rho * lsites, lsites))

        # convert into ms format for each line
        for i in mbslib.parse_mbs_data(mbs_input_file):
            # change the position of mutation if it occurred at target site
            if i['pos'][0] == 1.0:
                h = mbslib.mbs_to_ms_output(i, selpos, lsites)
                f.write("//\n")
                # write segregation sites
                f.write("segsites: {}\n".format(len(h['pos'])))

                # write position
                f.write("positions: ")
                # convert int to str
                pos_list = [str(i) for i in h['pos']]
                # change position of the mutation occurred at the target site
                pos_list[1] = str(2/lsites)
                f.write(" ".join(pos_list))
                f.write("\n")

                # write seq data
                f.write("\n".join(h["seq"]))
                f.write("\n\n")

            else:
                h = mbslib.mbs_to_ms_output(i, selpos, lsites)
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


def parameter_sets_given_t_p(sel_advantages, data_num, initial_freq):
    params = dict()

    params['N0'] = 5000
    params['generation'] = 20
    params['demography_in_year'] = [[0, 100 * params['N0'] * params['generation'], params['N0']]]

    # selection coefficients
    params['h'] = 0.5  # <--- co-dominance
    params['resolution'] = 0.01

    # number of trajectory
    params['n_trajectory'] = 1
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

    # derived allele frequency at which selection started acting
    params['p0'] = initial_freq
    # time selection started in year
    params['t0_in_year'] = 1
    # time selection started in unit of 4N generation
    params['t0'] = params['t0_in_year'] / (4 * params['N0'] * params['generation'])

    params_list = list()
    for s in sel_advantages:
        params['s'] = s
        params['selection_in_year'] = [[0, params['t0_in_year'], params['s'], params['h']]]
        for d in data_num:
            # time selection started in year
            params['data_num'] = d
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


def calc_stat(params, data_dir, ehh_data_dir, nrep):
    # calculation setting
    num_EHH = nrep
    max_iteration = 3
    num_run = 0

    # create empty dataframe
    print(params['p0'])
    initial_freq = params['p0']
    #print(initial_freq)

    labels = ['0.01', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45',
              '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '0.99']
    labels = [i for i in labels if float(i) > initial_freq]

    bins = [0.025, 0.075, 0.125, 0.175, 0.225, 0.275, 0.325, 0.375, 0.425,
            0.475, 0.525, 0.575, 0.625, 0.675, 0.725, 0.775, 0.825, 0.875,
            0.925, 0.975, 1.0]
    bins = [i for i in bins if i > initial_freq - 0.025]

    EHH_data_dic = {}
    for i in labels:
        EHH_data_dic[i] = pd.DataFrame(
            columns=[' EHH_A ', ' EHH_D ', ' rEHH ', ' iHH_A ', ' iHH_D ', ' iHS ', ' f_current ', 'f_current_bin',
                     't0_in_year'])

    # set candidate t0 range
    max_age = int(5000 * params['generation'] * 4)
    t0_age_candidate = np.arange(1000, max_age + 1, 1)
    # weight according to popsize
    # SNM
    weight = [1 for i in range(len(t0_age_candidate))]

    while min([len(EHH_data_dic[i]) for i in labels]) < num_EHH:
        t0_in_year = int(random.choices(t0_age_candidate, weights=weight, k=1)[0])
        # time selection started in year
        params['t0_in_year'] = t0_in_year
        # time selection started in unit of 4N generation
        params['t0'] = params['t0_in_year'] / (4 * params['N0'] * params['generation'])
        # selection selection history
        params['selection_in_year'] = [[0, params['t0_in_year'], params['s'], params['h']]]

        # path to trajectory file
        path_to_traj = f"{data_dir}/traj_t0{params['t0_in_year']}_p0{params['p0']}_s{params['s']}_datanum{params['data_num']}"
        # mbs out put file
        path_to_mbs_output = f"{data_dir}/mbs_nsam{params['nsam']}_t0{params['t0_in_year']}" \
                             f"_p0{params['p0']}_s{params['s']}_datanum{params['data_num']}.dat"

        # generate trajectory
        make_tarjctoyfiles_given_t_p(params['N0'], params['t0'], params['p0'], params['generation'],
                                     params['demography_in_year'], params['selection_in_year'],
                                     params['resolution'],
                                     params['n_trajectory'], path_to_traj)

        # condition on derived allele is segregating
        traj_file = "{}/traj_t0{}_p0{}_s{}_datanum{}_0.dat" \
            .format(data_dir, params['t0_in_year'], params['p0'], params['s'], params['data_num'])
        dt = pd.read_table(traj_file, header=None)
        d_freq = dt.iloc[0, 3]
        #print(d_freq)

        if d_freq == 1:
            os.remove(traj_file)
            pass
        else:
            # run mbs to make SNP data
            run_mbs(params['nsam'], params['per_site_theta'], params['per_site_rho'],
                    params['lsites'], params['selpos'],
                    params['n_trajectory'], params['nrep_per_traj'],
                    path_to_mbs_output, path_to_traj)

            # condition on derived allele is segregating
            for m in mbslib.parse_mbs_data(path_to_mbs_output):
                d_num = m['allele'].count('d')
                a_num = m['allele'].count('a')

            if d_num < 2 or a_num < 2:
                os.remove(traj_file)
                os.remove(path_to_mbs_output)
                pass

            else:
                # rewrite mbs to ms
                ms_output_file = '{}/mbs_t0{}_p0{}_s{}_datanum{}.txt'\
                    .format(data_dir, params['t0_in_year'], params['p0'], params['s'], params['data_num'])

                mbs2msoutput(path_to_mbs_output, ms_output_file, params['nsam'], params['n_trajectory'],
                             params['per_site_theta'], params['per_site_rho'], params['selpos'],params['lsites'])

                # run R script to calculate EHH
                r = pyper.R(use_numpy='True', use_pandas='True')
                r("t0_in_year <- {}".format(params['t0_in_year']))
                r("nrep <- {}".format(params['n_trajectory']))
                r("lsites <- {}".format(params['lsites']))
                r("s <- {}".format(params['s']))
                r("p0 <- {}".format(params['p0']))
                r("data_num <- {}".format(params['data_num']))
                r("source(file='calc_EHH_given_t_p_pyper.R')")

                # get EHH data
                EHH_data = pd.DataFrame(r.get('EHH_data'))
                #print(EHH_data)

                # add EHH data according to its frequency
                EHH_data['f_current_bin'] = pd.cut(EHH_data[' f_current '], bins, labels=labels, right=False)
                EHH_data['t0_in_year'] = params['t0_in_year']

                for i in labels:
                    # extract for derived allele frequencies
                    temp_df = EHH_data[EHH_data['f_current_bin'] == i]
                    EHH_data_dic[i] = pd.concat([EHH_data_dic[i], temp_df], axis=0, ignore_index=True)

                # remove intermediate file
                os.remove(traj_file)
                os.remove(path_to_mbs_output)
                num_run += 1
                if num_run == 500:
                    print('data_num:', params['data_num'], num_run, 'times run')
                if num_run > max_iteration:
                    break

    for i in labels:
        EHH_data_dic[i].to_csv(
            '{}/EHH_data_f{}_p0{}_s{}_datanum{}.csv'
                .format(ehh_data_dir, i, params['p0'] ,params['s'], params['data_num']))
    print('data_num:', params['data_num'], 'done:', num_run, 'times run', 'p0:', params['p0'], 's:', params['s'])


def calc_power(ehh_data_dir, data_num, labels, p, s, nrep, path_to_percentile, power_dir, file_name):

    # empty data frame
    EHH_data_dic = {}
    for i in labels:
        EHH_data_dic[i] = pd.DataFrame(
            columns=[' EHH_A ', ' EHH_D ', ' rEHH ', ' iHH_A ', ' iHH_D ', ' iHS ', ' f_current ', 'f_current_bin',
                     't_mutation_in_year'])
    # joint data
    for m in labels:
        for n in data_num:
            df = pd.read_csv(
                '{}/EHH_data_f{}_p0{}_s{}_datanum{}.csv'.format(ehh_data_dir, m, p, s, n), index_col=0)
            EHH_data_dic[m] = pd.concat([EHH_data_dic[m], df], axis=0, ignore_index=True)
        #print(len(EHH_data_dic[m]))
        EHH_data_dic[m].to_csv('{}/EHH_data_f{}_p0{}_s{}.csv'.format(ehh_data_dir, m, p, s), index=False)

    # calc power
    rEHH_power = []
    # calc rEHH power
    rEHH_percentile = path_to_percentile[:11] + 'rEHH_' + path_to_percentile[11:]
    df = pd.read_csv(rEHH_percentile, index_col=0, header=0)
    threshold_list = df['95']
    for f in labels:
        threshold = threshold_list[float(f)]
        df = pd.read_csv('{}/EHH_data_f{}_p0{}_s{}.csv'.format(ehh_data_dir, f, p, s))
        df = df[:nrep]
        df = df.replace([np.inf, -np.inf], np.nan)
        power = sum([i > threshold for i in df['rEHH']])/nrep
        rEHH_power.append(power)

    rEHH_power_df = pd.DataFrame(rEHH_power, columns=t_e, index=labels)
    rEHH_power_df.to_csv("{}/rEHH_{}".format(power_dir, file_name))

    # calc power
    iHS_power = []
    # calc rEHH power
    iHS_percentile = path_to_percentile[:11] + 'iHS_' + path_to_percentile[11:]
    df = pd.read_csv(iHS_percentile, index_col=0, header=0)
    threshold_list = df['5']
    for f in labels:
        threshold = threshold_list[float(f)]
        df = pd.read_csv('{}/EHH_data_f{}_p0{}_s{}.csv'.format(ehh_data_dir, f, p, s))
        df = df[:nrep]
        df = df.replace([np.inf, -np.inf], np.nan)
        power = sum([i > threshold for i in df['iHS']]) / nrep
        iHS_power.append(power)

    iHS_power_df = pd.DataFrame(iHS_power, columns=t_e, index=labels)
    iHS_power_df.to_csv("{}/iHS_{}".format(power_dir, file_name))


def main():
    # initial values
    # selection coefficients
    sel_advantages = [0.005]
    # derived allele frequency at which selection started acting
    initial_freq = 0.1
    data_num = np.arange(1, 2, 1)
    # number of replication
    nrep = 2
    # simulated data directory name
    data_dir = 'results'
    # statistics data directory name
    save_dir = 'sfs_data'
    # ehh data directory
    ehh_data_dir = 'ehh_data'

    testlist = parameter_sets_given_t_p(sel_advantages, data_num, initial_freq)
    n_cpu = int(multiprocessing.cpu_count()/2)
    with multiprocessing.Pool(processes=n_cpu) as p:
        p.map(functools.partial(calc_stat, data_dir, ehh_data_dir, nrep), testlist)

    f_list = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.99']
    f_list = [i for i in f_list if float(i) > initial_freq]
    # power data directory name
    power_dir = 'power'
    # path_to_percentile_data
    path_to_percentile = '../constant_pop/percentile_data/_percentile.csv'
    # power file name
    file_name = 'power_p{}_s{}_con_on_freq.csv'.format(initial_freq, sel_advantages[0])
    calc_power(save_dir, f_list, data_num, initial_freq, sel_advantages[0], nrep, path_to_percentile, power_dir,
               file_name)

if __name__=="__main__":
    main()




