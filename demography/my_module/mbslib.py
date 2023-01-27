import re
import subprocess

""" run, read analyze ms data """


def run_command(cmd):
    """ run command

    Args:
        cmd (str): command line to execute.

    Raises:
        exception when command returns error.
    """

    try:
        subprocess.check_output(cmd,
                stderr=subprocess.PIPE,
                shell=True)

    except subprocess.CalledProcessError as e:
        print(e.returncode)
        print(e.cmd)
        print(e.output.decode())


def parse_ms_data(filename):
    """ Generator function to parse ms data

    Args:
        filename (str): path to ms output file

    Yields:
        A dict of ms simulation of one replication.
        {'seq': list of 0-1 strings,
         'pos': list of SNP positions in float,
         'graph': genealogy in newick format in str}
    """

    # regular expressions
    ms_match = re.compile('ms\s(\d+)\s(\d+)')
    mbs_match = re.compile('command.+mbs\s(\d+)\s')
    seed_match = re.compile('seed|^\d+\s\d+\s\d+\s*$|^0x.+')
    blank_match = re.compile('^\s*$')
    graph_match = re.compile('^\(.+\);$')
    newdata_match = re.compile('//')
    segsites_match = re.compile('segsites:\s(\d+)')
    pos_match = re.compile('positions')

    #
    # initialization with fake values
    #
    samplesize = -1
    # nrep = -1
    # segsites = -1
    sn = 0

    pos = []
    graph = []

    # open data file
    with open(filename, 'r') as f:
        for line in f:

            # remove newline from the tail
            line = line.rstrip()

            # record sample_size and num_replication
            m = ms_match.search(line)
            if m:
                samplesize = int(m.group(1))
                # nrep = int(m.group(2))
                continue
                
            m = mbs_match.search(line)
            if m:
                samplesize = int(m.group(1))
                # nrep = int(m.group(2))
                continue

            # skip random number seed
            if seed_match.search(line):
                continue

            # skip blank line
            if blank_match.search(line):
                continue

            # tree info
            if graph_match.search(line):
                graph.append(line)
                continue

            # positions
            if pos_match.search(line):
                pos = list(map(float, line.split()[1:]))
                continue

            # new replication data start
            if newdata_match.search(line):
                # initialize variables
                seq = []
                continue

            # record num of segsites
            m = segsites_match.search(line)
            if m:
                segsites = int(m.group(1))

                # when there is no variation,
                # returun array of '0' seq
                if segsites == 0:
                    seq = ['0'] * samplesize
                    sn = 0
                    # yield seq
                    yield {'seq': seq, 'pos': [], 'graph': None}

                continue

            # read and append 0-1 seq
            seq.append(line)
            sn += 1

            # when all samples are read
            if sn == samplesize:
                sn = 0
                # yield
                yield {'seq': seq, 'pos': pos, 'graph': graph}


def parse_mbs_data(filename):
    """ Generator function to parse ms data

    Args:
        filename (str): path to ms output file

    Yields:
        A dict of ms simulation of one replication.
        {'seq': list of 0-1 strings,
         'pos': list of SNP positions in float,
         'graph': genealogy in newick format in str}
    """

    # regular expressions
    ms_match = re.compile('ms\s(\d+)\s(\d+)')
    mbs_match = re.compile('command.+mbs\s(\d+)\s')
    seed_match = re.compile('seed|^\d+\s\d+\s\d+\s*$|^0x.+')
    blank_match = re.compile('^\s*$')
    graph_match = re.compile('^\(.+\);$')
    newdata_match = re.compile('//')
    segsites_match = re.compile('segsites:\s(\d+)')
    pos_match = re.compile('positions')

    #
    # initialization with fake values
    #
    samplesize = -1
    # nrep = -1
    # segsites = -1
    sn = 0

    pos = []
    graph = []

    # open data file
    with open(filename, 'r') as f:
        for line in f:

            # remove newline from the tail
            line = line.rstrip()

            # record sample_size and num_replication
            m = ms_match.search(line)
            if m:
                samplesize = int(m.group(1))
                # nrep = int(m.group(2))
                continue
                
            m = mbs_match.search(line)
            if m:
                samplesize = int(m.group(1))
                # nrep = int(m.group(2))
                continue

            # skip random number seed
            if seed_match.search(line):
                continue

            # skip blank line
            if blank_match.search(line):
                continue

            # tree info
            if graph_match.search(line):
                graph.append(line)
                continue

            # positions
            if pos_match.search(line):
                pos = list(map(float, line.split()[1:]))
                continue

            # new replication data start
            if newdata_match.search(line):
                # initialize variables
                seq = []
                alleles = line.split(':')[1].split()
                continue

            # record num of segsites
            m = segsites_match.search(line)
            if m:
                segsites = int(m.group(1))

                # when there is no variation,
                # returun array of '0' seq
                if segsites == 0:
                    seq = ['0'] * samplesize
                    sn = 0
                    # yield seq
                    yield {'seq': seq, 'pos': [], 'graph': None, 'allele': alleles}

                continue

            # read and append 0-1 seq
            seq.append(line)
            sn += 1

            # when all samples are read
            if sn == samplesize:
                sn = 0
                # yield
                yield {'seq': seq, 'pos': pos, 'graph': graph, 'allele': alleles}
                

def to_01_allele(s):
    if s == 'a':
        return '0'
    elif s == 'd':
        return '1'
    else:
        return '9'
    

def mbs_to_ms_output(mbs_data, selpos, lsites):
    ''' Convert mbs output to ms type
    
    Args:
        mbs_data (dict): mbs output processed with parse_mbs_data()
        selpos (int):    position of the target site of selection
        lsites (int):    length of the simulated region
    Return:
        ms_data (dict):  converted simulation data
    '''
    
    # identify the index of the target site
    sel_idx = 0
    for p in mbs_data['pos']:
        if p>=selpos:
            break
        sel_idx += 1
    
    # convert allelic state of the target site
    target_site = [ to_01_allele(s) for s in mbs_data['allele'] ]
    
    # insert target position
    mbs_data['pos'].insert(sel_idx, selpos)
    # convert to relative positions
    positions = list(map(lambda x: x/lsites, mbs_data['pos']))
    
    ms_data = {
        # insert target allele
        'seq':[ seq[:sel_idx]+tgt+seq[sel_idx:] 
               for tgt, seq in zip (target_site, mbs_data['seq']) ],
        # positions
        'pos': positions,
        # graph
        'graph': mbs_data['graph']
    }
    return ms_data


def calc_pi(msdata):
    """ calculate pi

    This function receives a list of 0-1 ms sequences.
    Average number of nucleotide differences between a pair of seqs
    per region is calculated.

    Args:
        msdata (list(str)): ms 0-1 sequence data of one replication.

    Returns:
        PI (float): the value of PI per region

    """
    nsam = len(msdata)
    combination = nsam * (nsam - 1) / 2

    return sum([(nsam - sum(list(map(int, s)))) * sum(list(map(int, s)))
                for s in zip(*[list(m) for m in msdata])]) / combination


if __name__ == '__main__':
    pass

