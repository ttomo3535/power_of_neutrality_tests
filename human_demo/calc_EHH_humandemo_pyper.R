#import packages
library(rehh)
library(gap)
library(dplyr)
#temporary_list <- c(1,2,3)
options(scipen=1)

#position of the target site of selection (bp)
selpos = 1
# distance in bp
distance = distance_in_bp


# generate empty dataframe
createEmptyDf = function( nrow, ncol, colnames = c() ){
  data.frame( matrix( vector(), nrow, ncol, dimnames = list( c(), colnames ) ) )
}
EHH_data= createEmptyDf( nrep, 7,
                         colnames = c( "EHH_A", "EHH_D","rEHH",
                         "iHH_A","iHH_D","iHS","f_current") )

for (i in 1:nrep){
  hap_file <- ms_file
  hh <- data2haplohh(hap_file = hap_file,
                     chr.name = i,
                     allele_coding = '01',
                     position_scaling_factor = lsites)
  #calc ehh
  res <- calc_ehh(hh,
                  mrk = selpos,
                  include_nhaplo = TRUE,
                  discard_integration_at_border = FALSE)

  # extract ehh at calculation position
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
  traj_file <- traj_file

  dt <- read.table(traj_file, header = FALSE)
  # extract current derived allele frequency
  f_current <- dt[1,4]
  #f_current = 0

  # add to dataframe
  temporary_list <- c(EHH_A,EHH_D,rEHH,IHH_A,IHH_D,iHS,f_current)
  EHH_data[ i, ] = temporary_list
}
