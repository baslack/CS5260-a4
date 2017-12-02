#!/bin/sh -login
for iter in 1 2 3 4 5;
do
	#for data in 16384;
	for data in 512 1024 2048 4096 8192 16384;
	do
		#for block in 64 128;
		for block in 64 128 256 512 1024;
		do
			#for threads in 1 2 4 8;
			#do
				#qsub -F "$data $data $block $block $threads $threads" a4.pbs;
			#done
			qsub -F "$data $data $block $block 8 4" a4.pbs;
			sleep 60;
			#qsub -F "$data $data $block $block 4 2" a4.pbs;
		done
	done
	sleep 10;
done
