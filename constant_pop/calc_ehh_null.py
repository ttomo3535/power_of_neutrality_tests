import numpy as np
import pandas as pd
import csv
import multiprocessing
import functools

from my_module import backward_trajectory as back
from my_module import mbslib


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


def parameter_sets_backward(current_frequency, sel_advantages):
    '''

    Args:
        current_frequency: current frequencies of selected alleles
        sel_advantages: selection coefficients

    Returns:

    '''
    params = dict()

    params['N0'] = 5000
    params['generation'] = 20
    params['demography_in_year'] = [[0, 100 * params['N0'] * params['generation'], params['N0']]]

    # selection coefficients
    params['s'] = 0
    params['h'] = 0.5  # <--- co-dominance
    params['resolution'] = 100

    # number of trajectory
    params['n_trajectory'] = 1000
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
    for f_current in current_frequency:
        params['f_current'] = f_current
        for s in sel_advantages:
            params['s'] = s
            params_list.append(params.copy())

    return params_list


def run_mbs_to_msoutput(params, ms_data_dir, ehh_data_dir):
    '''
    run mbs and convert outputs in ms format to calculate EHH statistics
    Args:
        params:
        ms_data_dir:
        ehh_data_dir:

    Returns:

    '''
    # path to trajectory files
    path_to_traj = f"results/traj_f{params['f_current']}_s{params['s']}"

    # generate trajectory files
    make_trajectoryfiles_backward(params['N0'], params['generation'], 
                                  params['demography_in_year'], params['f_current'], 
                                  params['s'], params['h'], params['resolution'], 
                                  params['n_trajectory'], path_to_traj)

    # path to mbs output
    path_to_mbs_output = f"results/mbs_nsam{params['nsam']}_fcurrent{params['f_current']}_s{params['s']}.dat"

    # run mbs
    run_mbs(params['nsam'], params['per_site_theta'], params['per_site_rho'], 
            params['lsites'], params['selpos'], 
            params['n_trajectory'], params['nrep_per_traj'], 
            path_to_mbs_output, path_to_traj)

    # set the point at which EHH values are calculated
    distance_in_bp = 15000

    # convert mbs format into ms format
    with open('{}/mbs_f{}_s{}.txt'.format(ms_data_dir, params['f_current'], params['s']), 'w') as f:
        # ms command line
        f.write("ms {} {} -t {} -r {} {}\n\n".format(params['nsam'] ,params['n_trajectory'], 
                                                     params['per_site_theta']*params['lsites'], 
                                                     params['per_site_rho']*params['lsites'], params['lsites']))
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
                pos_list[1] = str(2/params['lsites'])
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

    # run R script to calculate EHH statistics
    mbslib.run_command('Rscript calc_ehh_backward.R {} {} {} {} {} {} {}'.format(params['f_current'], params['n_trajectory'],
                                                                    params['lsites'], params['s'], ms_data_dir, ehh_data_dir,
                                                                              distance_in_bp))

    print(params['f_current'], params['s'], 'done')


def calc_percentile(current_frequency, ehh_data_dir, percentile_data_dir):
    # percentile lists
    rEHH_percentile_list = []
    iHS_percentile_list = []
    bins = np.arange(1, 100, 1)
    for i in current_frequency:
        EHH_data = pd.read_csv("{}/EHH_data_f{}_s0.csv".format(ehh_data_dir, i))
        #EHH_data = EHH_data.replace({'rEHH': {np.inf: float('inf')}})
        # condition on the case that more than two of ancestral and derived alleles are contained in samples
        EHH_data = EHH_data[EHH_data['iHH_A']!=0]
        EHH_data = EHH_data[EHH_data['iHH_D']!=0]
        EHH_data = EHH_data[:1000]
        # calculate percentile
        rEHH_percentile_list.append(list(np.percentile(EHH_data['rEHH'], bins)))
        iHS_percentile_list.append(list(np.percentile(EHH_data['iHS'], bins)))

    df_rEHH = pd.DataFrame(rEHH_percentile_list, columns = bins, index = current_frequency)
    df_rEHH.to_csv('{}/rEHH_percentile.csv'.format(percentile_data_dir))

    df_iHS = pd.DataFrame(iHS_percentile_list, columns = bins, index = current_frequency)
    df_iHS.to_csv('{}/iHS_percentile.csv'.format(percentile_data_dir))


def main():
    # initial values
    # current frequency of derived allele
    current_frequency = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25,
                         0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
                         0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
                         0.9, 0.95, 0.99]
    # selection coefficients
    sel_advantages = [0]
    # parameters sets
    testlist = parameter_sets_backward(current_frequency, sel_advantages)
    # ms data directory name
    ms_data_dir = 'results'
    # ehh data directory
    ehh_data_dir = 'ehh_data'
    # percentile data directory
    percentile_data_dir = 'percentile_data'


    n_cpu = int(multiprocessing.cpu_count()/2)
    with multiprocessing.Pool(processes=n_cpu) as p:
        p.map(functools.partial(run_mbs_to_msoutput, ms_data_dir=ms_data_dir, ehh_data_dir=ehh_data_dir), testlist)

    # calculate percentile
    calc_percentile(current_frequency, ehh_data_dir, percentile_data_dir)

if __name__=="__main__":
    main()


