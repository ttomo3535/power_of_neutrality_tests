import sys
import numpy as np
import matplotlib.pyplot as plt


def check_demography_input(demography):
    ''' Confirm the format of demography setting
    
    Args:
        demography (list): list of demographic parameters
    '''
    
    # check if the end of a phase is 
    #  the same as the beginning of the next phase
    y = 0
    for ph in demography:
        y0, y1, s = ph
        
        if y != y0:
            print('Phase mismatch')
            print(f'{y} != {y0}')
            print('t_start and t_end of the previous phase should be the same')
        y = y1


def year_to_gen(gen, N0, demography):
    ''' Convert demographic parameters in units of generations
    
    Convert beginning and end of each phase in generations.
    Add the oldest extra phase which starts 999*4N generations ago.
    
    Args:
        gen (int): generation time in year.
        demography (list): list of demographic parameters in year
        
    Return:
        in_gen (list): list of demographic parameters in generations
    '''
    
    in_gen = []
    
    # from year to generation
    for ph in demography:
        y0, y1, s = ph
        
        in_gen.append([int(y0/gen), int(y1/gen), s])
        
    # add extra phase
    # continue the same size until long time ago
    t_last = int(y1/gen)
    if t_last < 999*4*N0:
        end = 999*4*N0
    elif t_last < 9999*4*N0:
        end = 9999*4*N0
    else:
        end = 99999*4*N0
        
    in_gen.append([t_last, end, s])
        
    return in_gen


def reverse_demography(dem):
    ''' Reverse the list of demographic parameters
    
    Intended to conver the list from backward to forward in time.
    
    Args:
        dem (list): list of demographic parameters
    Returns 
        forward_in_time (list): Reversed list of demographic parameters
    '''
    # read the oldest phase breakpoint
    # the oldest time is set 0 in the reversed list
    t_last = dem[-1][1]
    
    #
    # convert
    #
    forward_in_time = []
    
    for ph in reversed(dem):
        ts, te, s = ph
        forward_in_time.append([t_last-te, t_last-ts, s])
        
    return forward_in_time


def convert_parameters(t_mutation_in_year, demography_in_year, gen, N0):
    
    check_demography_input(demography_in_year)
    demography_in_generations = year_to_gen(gen, N0, demography_in_year)
    demography_forward_in_time = reverse_demography(demography_in_generations)
    
    # convert time of the mutation 
    #  from year to generations
    # and from backward to forward
    t_mut = int(t_mutation_in_year/gen)
    t_mut_forward = demography_forward_in_time[-1][1]-t_mut
    
    return t_mut_forward, demography_forward_in_time


def freq_change(n0, s=0, h=0.5, size=100, resolution=100):
    ''' Simulate frequency change forward in time
    
    -------------
    AA   Aa    aa
     1  1+2hs 1+2s
    p^2  2pq   q^2
    -------------
    
    Args:
        n0 (int):  number of derived allele
        s (float): selection coefficient
        h (float): dominant coefficient
        size (int): diploid population size (N)
    Return:
        [int, int]: 
            final freq of the derived allele
            time (generations) when allele fixed or lost
    '''

    # initial conditions
    derived = n0
    q = derived/(2*size)
    t = 0

    # simulate until a specific amount of frequency change is observed
    while int(n0/(2*size)*resolution) == int(q*resolution):
        
        # calc freq
        p = 1-q
        
        # expected freq change of 'a'
        delta_q = 2*s*p*q*(q+h*(p-q))
        
        # when expected freq is less than 0 or more than 1,
        #  adjust next q, q=0 or 1
        if q+delta_q < 0:
            q_exp = 0
            
        elif q+delta_q < 1:
            q_exp = q+delta_q
        else:
            q_exp = 1
        
        # number of derived alleles in the next generation
        derived = np.random.binomial(n=(2*size), p=q_exp)
        
        # frequency of the mutant in the next generation
        q = derived/(2*size)
        t += 1
        
        # when fixed or lost, stop simulation
        if derived==0 or derived==2*size:
            break

    return t, derived


def simulate_forward_trajectory(s, h, t_mut_forward, demography_forward_in_time, 
                                resolution=100):
    ''' Generate trajectory
    
    Args:
        s (float): selection coefficient of 'a'. 'aa' 1+2s
        h (float): dominant coefficient. 'Aa' 1+2hs
        t_mut_forward (int): whem mutation arises. 
                             forward in time, in generations.
        demography_forward_in_time (lst): list of demographic params.
                             forward in time, in generations.
                             
    Returns:
        trajectory (list): list of trajectory info.
                    [[begining time of a phase, 
                     end time, 
                     number of mutant allele, 
                     freq of mutant, 
                     diploid pop size], ...]
    '''
    # init
    poly = 0
    t = 0
    n = 0
    p = 0

    trajectory = []

    # for each demographic phase
    for phase in demography_forward_in_time:

        # Set demographic params
        #  t_begining of the phase (smaller)
        #  t_end of the phase (larger)
        #  population size
        t_beg, t_end, pop_size = phase
        
        # reset the number of mutant allele
        n = p * 2*pop_size

        # when monomorphic
        if poly == 0:
            
            # before the mutation arises, or
            # after loss or fixation of the mutation
            if t_mut_forward<t_beg or t_end<t_mut_forward:

                # record the demographic params and freqs
                # demographic params unchanged
                trajectory.append([t_beg, t_end, n, p, pop_size])
                t = t_end
                continue
            
            # when mutation arises
            # Disect the phase; before and after the mutation
            else:
                # record the phase until mutation
                trajectory.append([t_beg, t_mut_forward, n, p, pop_size])
                
                # set polymorphic tag
                poly = 1
                n = 1     # <--- means mutations
                p = 1/(2*pop_size)
                t = t_mut_forward

        # when ancestral and derived alleles co-exist
        if poly == 1:
            
            # simulate until the end of the demographic phase
            while t<t_end:

                # simulate frequency change of the mutant
                elapsed_time, n_after = freq_change(n, s, h, pop_size, 
                                                    resolution)

                # freq change
                #  in the demographic phase
                if t+elapsed_time < t_end:

                    # add a new trajectory phase
                    trajectory.append([t, t+elapsed_time, n, n/(2*pop_size), pop_size])

                    t += elapsed_time
                    n = n_after
                    p = n/(2*pop_size)

                    # when mutant fixed or lost,
                    if n_after==0 or n_after==2*pop_size:
                        
                        # record the rest of the demographic phase
                        trajectory.append([t, t_end, n, n/(2*pop_size), pop_size])
                        
                        # set monomorphic
                        #  and go to the next demographic phase
                        poly=0
                        t = t_end
                        continue
                        
                # end of the demographic phase comes faster than freq change
                else:
                    # record until the end of the demographic phase
                    # and move to the next phase
                    trajectory.append([t, t_end, n, n/(2*pop_size), pop_size])
                    t = t_end
                    p = n/(2*pop_size)
                    continue
                    
    return trajectory


def gen_traj(
    t_mutation_in_year, demography_in_year, 
    s, h, 
    generation=20, N0=100, resolution=100, condition='ALL'):
    
    ''' Returns trajectory 
    
    Args:
        t_mut_forward (int):
                          whem mutation arises. 
                          forward in time, in generations.
        demography_forward_in_time (list):
                          list of demographic params.
                          forward in time, in generations.
        s (float):        selection coefficient of 'a'. 'aa' 1+2s
        h (float):        dominant coefficient. 'Aa' 1+2hs
        generation (int): generation time in year.
        N0 (int):         standard diploid population size
        resolution (int): step size of trajectory
                          100 if 1%, 1000 if 0.1%
        condition (str):  'ALL' outputs all trajectories.
                          'FIXED' outputs only fixed trajectories.
                          'LOST' outputs only lost trajectories.
                          'NOTLOST' outputs fixed and segregating trajectories.
                          'POLY' outputs segregating trajectories.
                          It may take quite long depending on assumptions.
                             
    Returns:
        trajectory (list): list of trajectory info.
                    [[begining time of a phase, 
                     end time, 
                     number of mutant allele, 
                     freq of mutant, 
                     diploid pop size], ...]
    '''
    # check condition
    condition = str.upper(condition)
    if condition not in ['ALL', 'FIXED', 'LOST', 'NOTLOST', 'POLY']:
        print('please select the condition of the trajectory from ')
        print('ALL, FIXED, LOST, NOTLOST, POLY')
        sys.exit()

    t_mut_forward, demography_forward_in_time = convert_parameters(
        t_mutation_in_year, demography_in_year, 
        generation, N0)
    
    agree = 0
    
    while agree==0:
    
        traj = simulate_forward_trajectory(
            s, h, 
            t_mut_forward, demography_forward_in_time, 
            resolution)
        
        curr_freq = traj[-1][3]
        curr_n = traj[-1][-1]
        
        if condition=='ALL':
            agree = 1
        elif condition=='FIXED':
            if curr_freq > 1-1/(2*curr_n):
                agree = 1
        elif condition=='LOST':
            if curr_freq < 1/(2*curr_n):
                agree = 1
        elif condition=='NOTLOST':
            if 1/(2*curr_n)<curr_freq:
                agree = 1
        elif condition=='POLY':
            if 1/(2*curr_n)<curr_freq and curr_freq< 1-1/(2*curr_n):
                agree = 1

    return traj


def mbs_input(t_mutation_in_year, demography_in_year, 
              s, h, 
              generation=20, N0=100, resolution=100, 
              condition='ALL'):
    ''' Generate mbs input data
    
    Args:
        t_mut_forward (int):
                          whem mutation arises. 
                          forward in time, in generations.
        demography_forward_in_time (list):
                          list of demographic params.
                          forward in time, in generations.
        s (float):        selection coefficient of 'a'. 'aa' 1+2s
        h (float):        dominant coefficient. 'Aa' 1+2hs
        generation (int): generation time in unit of year.
        N0 (int):         standard diploid population size
        resolution (int): step size of trajectory
                          100 if 1%, 1000 if 0.1%
        condition (str):  'ALL'   outputs all trajectories.
                          'FIXED' outputs only fixed trajectories.
                          'LOST'  outputs only lost trajectories.
                          'NOTLOST' outputs fixed and segregating trajectories.
                          'POLY'  outputs segregating trajectories.
                          It may take quite long depending on assumptions.
    Return:
        traj (list):      List of freq trajectories, 
                          backward in time, 
                          in units of 4N0.
                          [[begining time of the phase, 
                            end time of the phase, 
                            pop size in units of 4N0,
                            freq of mutant], ...]

    '''
    
    condition = str.upper(condition)
    if condition not in ['ALL', 'FIXED', 'LOST', 'NOTLOST', 'POLY']:
        print('please select the condition of the trajectory from ')
        print('ALL, FIXED, LOST, NOTLOST, POLY')
        sys.exit()
    
    # generate forward trajectory
    forward_trajectory_in_generation = gen_traj(
        t_mutation_in_year, demography_in_year, 
        s, h, 
        generation, N0, resolution, condition)

    # convert 
    #  forward trajectory --> backward trajectory,
    #  in unit of generations --> in units of 4N0
    traj = []
    t_last = forward_trajectory_in_generation[-1][1]

    for phase in reversed(forward_trajectory_in_generation):

        traj.append([(t_last-phase[1])/(4*N0), 
                     (t_last-phase[0])/(4*N0),  
                     phase[4]/N0,
                     phase[3]
                    ])
        
    return(traj)


if __name__ == '__main__':
    pass