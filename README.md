Benjamin A. Slack
CS5260
Assignment #4
CUDA 2D Matrix Addition
12.02.2017

Description:
Implements two dimensional matrix addition
via CUDA. Matrices are initialized on the device 
to a default specification for the assignment. 
They are then added and retrieved. The results are 
checked to a CPU version. Timing data is recored for 
the GPU and CPU computation times. This data is returned
via stdout.

Package Contents:

msvc:
This directory contains the Microsoft Visual C++ 2017 
project files. It was compiled on my system against the 
CUDA 9.0 toolkit.

gcc:
This directory contains a standard makefile project 
which I used for compiling on the THOR cluster 
(thor.cs.wmich.edu) as well as the accompanying bash 
and torque scripts for running and managing the test 
batches. The CUDA module 6.5 was used, and a small change 
required in the project syntax for the initialization of 
the grid and thread dimensions. The gcc folder also 
contains the bash and torque scripts used to generate 
the data set. Sample result.csv files are included 
though these don't represent raw output and have been 
massaged to allow for easy import into Google Sheets 
for analysis.

Usage:

./a4 -m <int: matrix size M> -n <int: matrix size N> 
     -x <int: grid dimension X> -y <int: grid dimension Y> 
	 -tx <int: thread dim X> -ty <int: thread dim Y>
	 
Example:
 
 ./a4 -m 512 -n 512 -x 64 -y 64 -tx 1 -ty 1
 
Output:
We did it! 
m: 512, n: 512, grid: 64 x 64, threads: 1, elapsed gpu time: 0.022268
m: 512, n: 512:, grid: 64 x 64, threads: 1, elapsed cpu time: 0.009210
	 
