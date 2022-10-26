#!/bin/bash

# input file and output directory need to be specified
data_file=$1
result_dir=$2

test ! -d $result_dir && echo "$result_dir is not a directory" && exit 0

# Setting up MATLAB environment variables
MCRDIR=/usr/local/MATLAB/MATLAB_Runtime/v91
temp=${LD_LIBRARY_PATH}
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MCRDIR}/runtime/glnxa64:${MCRDIR}/bin/glnxa64:${MCRDIR}/sys/os/glnxa64
XAPPLRESDIR=${MCRDIR}/X11/app-defaults
export XAPPLRESDIR
export LD_LIBRARY_PATH

# set path of RobustClone
robustclone=/path_to_robustclone/matlab_and_R_scripts

outputPrefix=$result_dir/rep_1

current=`date "+%Y-%m-%d %H:%M:%S"`
seconds_s=`date -d "$current" +%s`

log_file=$result_dir/log_1.txt
test -e $log_file && rm $log_file

eval $robustclone/carryout_RPCA $data_file $outputPrefix >> $log_file 2>&1

LD_LIBRARY_PATH=$temp
export LD_LIBRARY_PATH

test -e $outputPrefix.clone && rm $outputPrefix.clone

test -e $result_dir/temp.R && rm $result_dir/temp.R
echo "#!/usr/bin/Rscript" >> $result_dir/temp.R

echo "setwd('$result_dir')" >> $result_dir/temp.R
echo "source('$robustclone/Clustering_EvolutionaryTree_function.R')" >> $result_dir/temp.R
echo "AA <- read.table('$result_dir/rep',head=FALSE)" >> $result_dir/temp.R
echo "robust_clone <- LJClustering(AA)" >> $result_dir/temp.R
echo "clone_gety <- subclone_GTM(AA, robust_clone, 'SNV')" >> $result_dir/temp.R

echo "num_c <- length(robust_clone)" >> $result_dir/temp.R
echo "for (i in 1:num_c) {" >> $result_dir/temp.R
echo "  write.table(robust_clone[[i]], file = \"$outputPrefix.clone\", append = TRUE)" >> $result_dir/temp.R
echo "}" >> $result_dir/temp.R
echo "write.table(clone_gety, file = \"$outputPrefix.gety\")" >> $result_dir/temp.R

echo 'q()' >> $result_dir/temp.R
chmod a+x $result_dir/temp.R
/usr/bin/R CMD $result_dir/temp.R

current=`date "+%Y-%m-%d %H:%M:%S"`
seconds_e=`date -d "$current" +%s`
let time_used=seconds_e-seconds_s

echo -e "time used: $time_used" >> $log_file

exit 0