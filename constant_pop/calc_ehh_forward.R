#import packages
library(rehh)
library(gap)
library(dplyr)

args1 = commandArgs(trailingOnly=TRUE)[1]
args2 = commandArgs(trailingOnly=TRUE)[2]
args3 = commandArgs(trailingOnly=TRUE)[3]
args4 = commandArgs(trailingOnly=TRUE)[4]
args5 = commandArgs(trailingOnly=TRUE)[5]
args6 = commandArgs(trailingOnly=TRUE)[6]
args7 = commandArgs(trailingOnly=TRUE)[7]

# parameter sets
# t_mutaion_in_year
t_mutation_in_year = args1
# number of trajectories
nrep = as.integer(args2)
# length of simulated region (bp)
lsites = as.integer(args3)
# selection coefficients
s = args4
# ms data directory
ms_data_dir = args5
# ehh data directory
ehh_data_dir = args6
# position of the target site of selection (bp)
selpos = 1
# 0.025cM distance
distance = as.integer(args7)

# generate empty dataframe
createEmptyDf = function( nrow, ncol, colnames = c() ){
  data.frame( matrix( vector(), nrow, ncol, dimnames = list( c(), colnames ) ) )
}
EHH_data= createEmptyDf( nrep, 7, 
                         colnames = c( "EHH_A", "EHH_D", "rEHH",
                         "iHH_A", "iHH_D", "iHS", "f_current") )

for (i in 1:nrep){
  hap_file <- paste(ms_data_dir, '/mbs_tmutation', t_mutation_in_year, '_s',
                    s, '.txt', sep = "")
  hh <- data2haplohh(hap_file = hap_file,
                     chr.name = i,
                     allele_coding = '01',
                     position_scaling_factor = lsites)

  # calc ehh
  res <- calc_ehh(hh,
                  mrk = selpos,
                  include_nhaplo = TRUE,
                  discard_integration_at_border = FALSE)

  # extract ehh value at calculation point
  res1 <- tail(res$ehh[res$ehh[1] < distance, ], n = 1)

  EHH_A <- res1[2]
  EHH_D <- res1[3]
  rEHH <- res1[3]/res1[2]
  
  # calc unstandardized iHS
  res2 <- res$ihh 
  iHH_A <- res2[1]
  iHH_D <- res2[2]
  iHS <- log(res2[1]/res2[2])
  
  # get trajectory file
  number_of_trajectory = i - 1
  traj_file <- paste(ms_data_dir, '/traj_tmutation', t_mutation_in_year,
                     '_s', s ,'_',
                     number_of_trajectory,
                     '.dat', sep = "")

  dt <- read.table(traj_file, header = FALSE)
  # get current frequency of derived allele
  f_current <- dt[1,4]

  temporary_list <- c(EHH_A, EHH_D, rEHH, iHH_A, iHH_D, iHS, f_current)
  EHH_data[ i, ] = temporary_list
}
     
file_name <- paste(ehh_data_dir, '/EHH_data_tmutation', t_mutation_in_year, '_s',
                   s, '.csv', sep = "")
write.csv(EHH_data, file_name , row.names = FALSE)
     