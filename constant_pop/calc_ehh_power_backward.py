import numpy as np
import pandas as pd
import csv
import multiprocessing
from my_module import backward_trajectory as back
from my_module import mbslib
import functools


def make_trajectoryfiles_backward(N0, generation, demography_in_year, f_current, s, h, resolution, 
                         n_trajectory, path_to_traj):
    '''
        generate trajectory file
        Args:
            N0 (int):
            generation (int): generation time, years/generation
            demography_in_year (list): demographic history/
            f_current (float): current frequency of derived allele
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
        trajectory = back.mbs_input(f_current,
                                    demography_in_year,
                                    s, h,
                                    generation, N0, resolution,
                                    )

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
    
    cmd =  f'mbs {nsam} -t {per_site_theta} -r {per_site_rho} '
    cmd += f'-s {lsites} {selpos} '
    cmd += f'-f {n_trajectory} {nrep_per_traj} {path_to_traj} '
    cmd += f'> {path_to_mbs_output}'
    
    mbslib.run_command(cmd)


def parameter_sets_backward(current_frequency, sel_advantages, nrep):

    params = dict()
    
    params['N0'] = 5000
    params['generation'] = 20
    params['demography_in_year'] = [[0, 100*params['N0']*params['generation'], params['N0']]]

    # selection coefficients
    params['s'] = 0
    params['h'] = 0.5  # <--- co-dominance
    params['resolution'] = 100

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

    #params['f_current']
    params_list = list()
    for f_current in current_frequency:
        params['f_current'] = f_current
        for s in sel_advantages:
            params['s'] = s
            params_list.append(params.copy())

    return params_list


def run_mbs_to_msoutput(params, ms_data_dir, ehh_data_dir):
    
    # path to a trajectory file
    path_to_traj = f"results/traj_f{params['f_current']}_s{params['s']}"

    # generate a trajectory file
    make_trajectoryfiles_backward(params['N0'], params['generation'], 
                                  params['demography_in_year'], params['f_current'], 
                                  params['s'], params['h'], params['resolution'], 
                                  params['n_trajectory'], path_to_traj)

    # path to mbs oputput
    path_to_mbs_output = f"results/mbs_nsam{params['nsam']}_fcurrent{params['f_current']}_s{params['s']}.dat"

    # run mbs
    run_mbs(params['nsam'], params['per_site_theta'], params['per_site_rho'], 
            params['lsites'], params['selpos'], 
            params['n_trajectory'], params['nrep_per_traj'], 
            path_to_mbs_output, path_to_traj)

    # convert mbs format into ms format
    with open('{}/mbs_f{}_s{}.txt'.format(ms_data_dir, params['f_current'], params['s']), 'w') as f:
        # ms command line
        f.write("ms {} {} -t {} -r {} {}\n\n".format(params['nsam'], params['n_trajectory'],
                                                     params['per_site_theta'] * params['lsites'],
                                                     params['per_site_rho'] * params['lsites'], params['lsites']))
        # convert into ms format for each line
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

    # set the point at which EHH values are calculated
    distance_in_bp = 25000

    # R script to calculate EHH statistics
    mbslib.run_command('Rscript calc_EHH_backward.R {} {} {} {} {} {} {}'.format(params['f_current'], params['n_trajectory'],
                                                                                 params['lsites'], params['s'],
                                                                                 ms_data_dir, ehh_data_dir,
                                                                                 distance_in_bp))

    print('f', params['f_current'], 's', params['s'], 'done')


def calc_power_of_EHH_test(current_frequency, sel_advantages, data_dir, percentile_data_dir, power_dir, n_run):
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
    rEHH_power_list = []
    iHS_power_list = []

    # calc rEHH power
    df = pd.read_csv('{}/rEHH_percentile.csv'.format(percentile_data_dir), index_col=0, header=0)
    rEHH_threshold_list = df['95']

    for f in current_frequency:
        threshold = rEHH_threshold_list[float(f)]
        temporary_list = []
        for s in sel_advantages:
            EHH_data = pd.read_csv("{}/EHH_data_f{}_s{}.csv".format(data_dir, f, s))
            EHH_data = EHH_data.replace([np.inf, -np.inf], np.nan)
            power = sum([i > threshold for i in EHH_data['rEHH']]) / n_run
            temporary_list.append(power)

        rEHH_power_list.append(temporary_list)

    rEHH_power_df = pd.DataFrame(rEHH_power_list, columns=sel_advantages, index=current_frequency)
    rEHH_power_df.to_csv("{}/power_rEHH_backward.csv".format(power_dir))

    # iHS
    df = pd.read_csv('{}/iHS_percentile.csv'.format(percentile_data_dir), index_col=0, header=0)
    iHS_threshold_list = df['5']
    for f in f_list:
        threshold = iHS_threshold_list[float(f)]
        temporary_list = []
        for s in s_list:
            EHH_data = pd.read_csv("{}/EHH_data_f{}_s{}.csv".format(data_dir, f, s))
            EHH_data = EHH_data.replace([np.inf, -np.inf], np.nan)
            power = sum([i < threshold for i in EHH_data['iHS']]) / n_run
            temporary_list.append(power)

        iHS_power_list.append(temporary_list)

    iHS_power_df = pd.DataFrame(iHS_power_list, columns=sel_advantages, index=current_frequency)
    iHS_power_df.to_csv("{}/power_iHS_backward.csv".format(power_dir))



def main():
    # initial values
    # current frequency of derived allele
    current_frequency = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25,
                         0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
                         0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
                         0.9, 0.95, 0.99]
    # selection coefficients
    sel_advantages = [0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
                      0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                      0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                      0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # number of trajectory
    number_of_trajectory = 10
    # parameters sets
    testlist = parameter_sets_backward(current_frequency, sel_advantages, number_of_trajectory)
    # ms data directory name
    ms_data_dir = 'results'
    # ehh data directory
    ehh_data_dir = 'ehh_data'
    # percentile data directory
    percentile_data_dir = 'percentile_data'
    # power data directory
    power_dir = 'power'

    # calculate EHH statistics
    n_cpu = int(multiprocessing.cpu_count()/2)
    with multiprocessing.Pool(processes=n_cpu) as p:
        p.map(functools.partial(run_mbs_to_msoutput, ms_data_dir=ms_data_dir, ehh_data_dir=ehh_data_dir), testlist)

    # calculate power
    calc_power_of_EHH_test(current_frequency, sel_advantages, ehh_data_dir, percentile_data_dir, power_dir, number_of_trajectory)

if __name__=="__main__":
    main()



