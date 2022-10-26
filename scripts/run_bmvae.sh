#!/bin/bash

# input file and output directory need to be specified
data_file=$1
result_dir=$2

test ! -d $result_dir && echo "$result_dir is not a directory" && exit 0

# set path of bmvae
bmvae=/path_to_bmvae/bmvae.py

# run five times
for i in `seq 1 5`
do
	echo "run $i..."
	
	output_dir=$result_dir/rep_$i
	test ! -d $output_dir && mkdir $output_dir
	
	current=`date "+%Y-%m-%d %H:%M:%S"`
	seconds_s=`date -d "$current" +%s`
	
	log_file=$result_dir/log_$i.txt
	test -e $log_file && rm $log_file
	
	python $bmvae --input $data_file --output $output_dir --seed $i >> $log_file 2>&1

	current=`date "+%Y-%m-%d %H:%M:%S"`
	seconds_e=`date -d "$current" +%s`
	let time_used=seconds_e-seconds_s

	echo -e "time used: $time_used" >> $log_file

done

exit 0
