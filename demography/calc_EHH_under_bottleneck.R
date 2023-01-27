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
args8 = commandArgs(trailingOnly=TRUE)[8]
args9 = commandArgs(trailingOnly=TRUE)[9]
args10 = commandArgs(trailingOnly=TRUE)[10]

#parameter sets
#t_mutaion_in_year
t_mutation_in_year = args1
#number of trajectories
nrep = as.integer(args2)
#length of simulated region (bp)
lsites = as.integer(args3)
#selection coefficients
s = args4
#bottleneck age in year
b_age = args5
#bottleneck duration in year
b_duration = args6
#bottleneck strength
b_strength = args7
#the position at which EHH value is calculated
distance_in_bp = args8

# ms data directory
ms_data_dir = args9
# ehh data directory
ehh_data_dir = args10
#position of the target site of selection (bp)
selpos = 1


#generate empty dataframe
createEmptyDf = function( nrow, ncol, colnames = c() ){
  data.frame( matrix( vector(), nrow, ncol, dimnames = list( c(), colnames ) ) )
}
EHH_data= createEmptyDf( nrep, 7,
                         colnames = c( "EHH_A", "EHH_D","rEHH",
                         "iHH_A","iHH_D","iHS","f_current") )

for (i in 1:nrep){
  hap_file <- paste(ms_data_dir, '/mbs_tmutation', t_mutation_in_year, '_s',
                    s,'_bage',b_age,'_bduration',b_duration, '_bstrength',b_strength,'.txt', sep = "")

  hh <- data2haplohh(hap_file = hap_file,
                     chr.name = i,
                     allele_coding = '01',
                     position_scaling_factor = lsites)
  #calc ehh
  res <- calc_ehh(hh,
                  mrk = selpos,
                  include_nhaplo = TRUE,
                  discard_integration_at_border = FALSE)

  res1 <- tail(res$ehh[res$ehh[1] < distance_in_bp, ], n = 1)

  #calc rEHH
  #get EHH_A, EHH_D, rEHH
  EHH_A <- res1[2]
  EHH_D <- res1[3]
  rEHH <- res1[3]/res1[2]

  #calc unstandardized iHS
  res2 <- res$ihh
  IHH_A <- res2[1]
  IHH_D <- res2[2]
  iHS <- log(res2[1]/res2[2])

  #get trajectory file
  number_of_trajectory = i - 1
  traj_file <- paste(ms_data_dir, '/traj_tmutation', t_mutation_in_year,'_s', s ,
                     '_bage',b_age,
                     '_bduration', b_duration,
                     '_bstrength', b_strength,
                     '_',number_of_trajectory,
                     '.dat', sep = "")
  dt <- read.table(traj_file, header = FALSE)
  #get derived allele frequency
  f_current <- dt[1,4]

  temporary_list <- c(EHH_A,EHH_D,rEHH,IHH_A,IHH_D,iHS,f_current)
  EHH_data[ i, ] = temporary_list
}

file_name <- paste(ehh_data_dir, '/EHH_data_tmutation', t_mutation_in_year, '_s', s,'_bage',b_age,'_bduration',b_duration,'_bstrength',b_strength,
'_traj',traj,'.csv', sep = "")
write.csv(EHH_data, file_name , row.names = FALSE)

