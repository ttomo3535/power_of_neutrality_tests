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

#time selection started in year
t0_in_year = args1
# initial frequency
p0 = args2
#number of trajectories
nrep = as.integer(args3)
#length of simulated region (bp)
lsites = as.integer(args4)
#selection coefficients
s = args5
#position of the target site of selection (bp)
selpos = 1
#position at which ehh is calculated
distance = as.integer(args6)
#data directory
data_dir = args7
#save directory
save_dir = args8

#create empty dataframe
createEmptyDf = function( nrow, ncol, colnames = c() ){
  data.frame( matrix( vector(), nrow, ncol, dimnames = list( c(), colnames ) ) )
}
EHH_data= createEmptyDf( nrep, 7,
                         colnames = c( "EHH_A", "EHH_D","rEHH",
                         "iHH_A","iHH_D","iHS","f_current") )

for (i in 1:nrep){
  hap_file <- paste(data_dir, '/mbs_t0', t0_in_year, '_p0', p0, '_s', s,'.txt', sep = "")

  hh <- data2haplohh(hap_file = hap_file,
                     chr.name = i,
                     allele_coding = '01',
                     position_scaling_factor = lsites)
  # calculte ehh
  res <- calc_ehh(hh,
                  mrk = selpos,
                  include_nhaplo = TRUE,
                  discard_integration_at_border = FALSE)

  # get EHH
  res1 <- tail(res$ehh[res$ehh[1] < distance, ], n = 1)

  # calc rEHH
  EHH_A <- res1[2]
  EHH_D <- res1[3]
  rEHH <- res1[3]/res1[2]

  # calc unstandardized iHS
  res2 <- res$ihh
  IHH_A <- res2[1]
  IHH_D <- res2[2]
  iHS <- log(res2[1]/res2[2])

  # get trajectory file
  number_of_trajectory = i - 1
  traj_file <- paste('results/traj_t0', t0_in_year, '_p0', p0, '_s', s,
                     '_',number_of_trajectory,
                     '.dat', sep = "")

  dt <- read.table(traj_file, header = FALSE)
  # get current frequency of derived allele
  f_current <- dt[1,4]
  # add data
  temporary_list <- c(EHH_A,EHH_D,rEHH,IHH_A,IHH_D,iHS,f_current)
  EHH_data[ i, ] = temporary_list
}

file_name <- paste(save_dir, '/EHH_data_t0', t0_in_year, '_p0', p0, '_s', s, '.csv', sep = "")
write.csv(EHH_data, file_name , row.names = FALSE)

