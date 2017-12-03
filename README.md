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

Usage:
./a4 -m <int: matrix size M> -n <int: matrix size N> 
     -x <int: grid dimension X> -y <int: grid dimension Y> 
	 -tx <int: thread dim X> -ty <int: thread dim Y>
	 
 