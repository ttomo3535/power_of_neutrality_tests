import numpy as np
import pandas as pd
import csv
import random
import pyper
import multiprocessing
import os
from my_module import forward_trajectory as fwd
from my_module import mbslib
import functools

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


def parameter_sets_forward(N0, gen, demo, sel_advantages, data_num):
    '''

    Args:
        N0(int): current populatino size
        gen(int): generation in year
        demo(list): demography
        sel_advantages(list): selection coefficients
        data_num(list):

    Returns:

    '''
    params = dict()

    params['N0'] = N0
    params['generation'] = gen
    params['demography_in_year'] = demo

    # selection coefficients
    params['s'] = 0
    params['h'] = 0.5  # <--- co-dominance
    params['resolution'] = 100

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
    # tentative value
    params['t_mutation_in_year'] = 1

    # candidate mutation ages in year
    params_list = list()
    for s in sel_advantages:
        params['s'] = s
        for i in data_num:
            params['data_num'] = i
            params_list.append(params.copy())

    return params_list


def calc_EHH_conditional_on_frequency(params, pop_name, distance_in_bp, nrep, data_dir, ehh_data_dir):
    # calculation parameter sets
    # num_EHH
    num_EHH = nrep
    max_iteration = 5
    num_run = 0

    # f_list = [str(i) for i in nap.arange(0, 101, 5)]
    labels = ['0.01', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45',
                  '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '0.99']
    EHH_data_dic = {}
    for i in labels:
        EHH_data_dic[i] = pd.DataFrame(
            columns=[' EHH_A ', ' EHH_D ', ' rEHH ', ' iHH_A ', ' iHH_D ', ' iHS ', ' f_current ', 'f_current_bin', 't_mutation_in_year'])

    t1 = params['demography_in_year'][0][1]
    t2 = params['demography_in_year'][1][1]
    t3 = params['demography_in_year'][2][1]

    N0 = params['N0']
    N1 = params['demography_in_year'][1][2]
    N2 = params['demography_in_year'][2][2]
    N3 = params['demography_in_year'][3][2]

    max_age = int(5000 * params['generation'] * 16)
    mutation_age_candidate = np.arange(5000, max_age + 1, 1)
    # popsize range in year
    N0_ages = np.arange(5000, t1 + 1, 1)
    N1_ages = np.arange(t1 + 1, t2 + 1, 1)
    N2_ages = np.arange(t2 + 1, t3 + 1, 1)
    N3_ages = np.arange(t3 + 1, max_age, 1)

    # weight
    weight = [N0 / N0 for i in range(len(N0_ages))] + [N1 / N0 for i in range(len(N1_ages))] + \
             [N2 / N0 for i in range(len(N2_ages))] + [N3 / N0 for i in range(len(N3_ages))]

    while min([len(EHH_data_dic[i]) for i in labels]) < num_EHH:
        mutation_age = int(random.choices(mutation_age_candidate, weights=weight, k=1)[0])
        params['t_mutation_in_year'] = mutation_age
        #print(mutation_age)

        # path to trajectory
        path_to_traj = f"{data_dir}/traj_tmutation{params['t_mutation_in_year']}_s{params['s']}_popname{pop_name}_datanum{params['data_num']}"

        # path to mbs output
        path_to_mbs_output = f"{data_dir}/mbs_nsam{params['nsam']}_tmutation{params['t_mutation_in_year']}_s{params['s']}" \
                             f"_popname{pop_name}_datanum{params['data_num']}.dat"


        make_trajectoryfiles_forward(params['N0'], params['generation'],
                             params['demography_in_year'], params['t_mutation_in_year'],
                             params['s'], params['h'], params['resolution'],
                             params['n_trajectory'], path_to_traj)

        # condition on selected allele is segregating
        traj_file = "{}/traj_tmutation{}_s{}_popname{}_datanum{}_0.dat" \
            .format(data_dir, params['t_mutation_in_year'], params['s'], pop_name, params['data_num'])
        dt = pd.read_table(traj_file, header=None)
        d_freq = dt.iloc[0, 3]

        if d_freq == 1:
            os.remove(traj_file)
            pass

        else:
            # run mbs
            run_mbs(params['nsam'], params['per_site_theta'], params['per_site_rho'],
                    params['lsites'], params['selpos'],
                    params['n_trajectory'], params['nrep_per_traj'],
                    path_to_mbs_output, path_to_traj)

            # extract number of derived and ancestral alleles
            for m in mbslib.parse_mbs_data(path_to_mbs_output):
                d_num = m['allele'].count('d')
                a_num = m['allele'].count('a')

            # condition on number of derived and ancestral alleles are more than two
            if d_num < 2 or a_num < 2:
                os.remove(traj_file)
                os.remove(path_to_mbs_output)
                pass

            else:
                ms_file = '{}/mbs_tmutation{}_s{}_popname{}_datanum{}.txt'.format(
                        data_dir, params['t_mutation_in_year'], params['s'],pop_name,params['data_num'])
                ###calc_EHH
                with open(ms_file, 'w') as f:
                    # convert into ms format for each line
                    f.write("ms {} {} -t {} -r {} {}\n\n".format(params['nsam'], params['n_trajectory'],
                                                                 params['per_site_theta'] * params['lsites'],
                                                                 params['per_site_rho'] * params['lsites'],
                                                                 params['lsites']))
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


                # run R script to
                r = pyper.R(use_numpy='True', use_pandas='True')
                r("nrep <- {}".format(params['n_trajectory']))
                r("lsites <- {}".format(params['lsites']))
                r("distance_in_bp <- {}".format(distance_in_bp))
                #r("ms_file <- {}".format(ms_file))
                r.assign("ms_file", ms_file)
                r.assign("traj_file", traj_file)
                #r("traj_file <- {}".format(traj_file))
                r("source(file='calc_EHH_humandemo_pyper.R')")

                # extact EHH data
                temp_data = r.get("EHH_data")
                print(temp_data)
                EHH_data = pd.DataFrame(r.get('EHH_data'))

                # add EHH data according to its frequency
                label = ['0.01', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45',
                         '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '0.99']
                bins = [0, 0.025, 0.075, 0.125, 0.175, 0.225, 0.275, 0.325, 0.375, 0.425,
                        0.475, 0.525, 0.575, 0.625, 0.675, 0.725, 0.775, 0.825, 0.875,
                        0.925, 0.975, 1.0]

                EHH_data['f_current_bin'] = pd.cut(EHH_data[' f_current '], bins, labels=label, right=False)
                EHH_data['t_mutation_in_year'] = params['t_mutation_in_year']

                for i in labels:
                    # extract for derived allele frequencies
                    temp_df = EHH_data[EHH_data['f_current_bin'] == i]
                    EHH_data_dic[i] = pd.concat([EHH_data_dic[i], temp_df], axis=0, ignore_index=True)

                # print('mutation_age:', params['t_mutation_in_year'], 'num_data', params['data_num'], 'num run', num_run)
                # remove intermediate file
                os.remove(traj_file)
                os.remove(path_to_mbs_output)
                os.remove(ms_file)

                num_run += 1
                if num_run == 500:
                    print('data_num:', params['data_num'], num_run, 'times run')
                if num_run > max_iteration:
                    break
    # save
    for i in labels:
        EHH_data_dic[i].to_csv(
            '{}/EHH_data_null_f{}_s{}_popname{}_datanum{}.csv'
                .format(ehh_data_dir, i, params['s'], pop_name, params['data_num']))
    print(pop_name, 'data_num', params['data_num'], 'done', num_run, 'times run')


def calc_null_percentile(ehh_data_dir, data_num, pop_name, percentile_data_dir):
    labels = ['0.01', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45',
              '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '0.99']

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
                '{}/EHH_data_null_f{}_s0_popname{}_datanum{}.csv'
                    .format(ehh_data_dir, m, pop_name, n), index_col=0)
            EHH_data_dic[m] = pd.concat([EHH_data_dic[m], df], axis=0, ignore_index=True)

    # condition on number of derived and ancestral alleles are more than two
    for i in labels:
        EHH_data_dic[i] = EHH_data_dic[i][EHH_data_dic[i][' iHH_D '] != 0]
        EHH_data_dic[i] = EHH_data_dic[i][EHH_data_dic[i][' iHH_A '] != 0]
        EHH_data_dic[i] = EHH_data_dic[i][:1000]


    # calc percentile
    rEHH_percentile_list = []
    iHS_percentile_list = []
    f_list = ['0.01', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45',
              '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '0.99']

    bins = [1, 5, 10, 50, 90, 95, 99]
    for i in f_list:
        EHH_data_dic[i] = EHH_data_dic[i].replace({' rEHH ': {np.inf: float('inf')}})
        rEHH_percentile_list.append(list(np.percentile(EHH_data_dic[i][' rEHH '], bins)))
        iHS_percentile_list.append(list(np.percentile(EHH_data_dic[i][' iHS '], bins)))

    name_list = ['rEHH', 'iHS']
    columns = bins
    index = ['0.01', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45',
             '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '0.99']

    list_list = [rEHH_percentile_list, iHS_percentile_list]
    for i, m in zip(list_list, name_list):
        df = pd.DataFrame(i, columns=columns, index=index)
        df.to_csv('{}/{}_percentile_popname{}.csv'
                  .format(percentile_data_dir, m, pop_name))


def main():
    ### demography_parameter
    data_num = np.arange(1, 21, 1)
    # selection coefficient
    sel_advantages = [0]
    # generation in year
    gen = 25

    # pop size
    N0 = 20000
    N1 = 2000
    N2 = 14000
    N3 = 10000

    # demographic event timing in year
    t1 = 400 * gen
    t2 = 2800 * gen
    t3 = 10000 * gen

    demo = [[0, t1, N0],
             [t1, t2, N1],
             [t2, t3, N2],
             [t3, 100 * N0 * gen, N3]
            ]

    # parameters sets
    testlist = parameter_sets_forward(N0, gen, demo, sel_advantages, data_num)
    pop_name = 'euasian'
    # the point at which EHH values are calculated
    distance_in_bp = 20000
    # ms data directory name
    data_dir = 'results'
    # ehh data directory
    ehh_data_dir = 'ehh_data'
    # number of replication
    nrep = 10

    n_cpu = int(multiprocessing.cpu_count() / 2)
    with multiprocessing.Pool(processes=n_cpu) as p:
        p.map(functools.partial(calc_EHH_conditional_on_frequency,
                                distance_in_bp=distance_in_bp, data_dir=data_dir, ehh_data_dir=ehh_data_dir,
                                nrep=nrep, pop_name = pop_name), testlist)

    # calc percentile
    percentile_data_dir = 'percentile'
    calc_null_percentile(ehh_data_dir, data_num, pop_name, percentile_data_dir)


if __name__=="__main__":
    main()


