/*
Benjamin A. Slack
CS5260
CUDA Addition of 2D Matricies
11.30.17
*/

// Developing this on Windows so that I have access to 
// my GPU for testing. However, time.h and sys/time.h
// are not available in Windows for reasons of M$ infinite
// wisdom. Therefore using this snippet from Stack Overflow
// to make a platform indepedent timing function. Should 
// make it so I can compile on windows or Thor as needed.
//
// URL:
// https://stackoverflow.com/questions/17432502/how-can-i-measure-cpu-time-and-wall-clock-time-on-both-linux-windows

//  Windows
#ifdef _WIN32
#include <Windows.h>
double get_wall_time() {
	LARGE_INTEGER time, freq;
	if (!QueryPerformanceFrequency(&freq)) {
		//  Handle error
		return 0;
	}
	if (!QueryPerformanceCounter(&time)) {
		//  Handle error
		return 0;
	}
	return (double)time.QuadPart / freq.QuadPart;
}
double get_cpu_time() {
	FILETIME a, b, c, d;
	if (GetProcessTimes(GetCurrentProcess(), &a, &b, &c, &d) != 0) {
		//  Returns total user time.
		//  Can be tweaked to include kernel times as well.
		return
			(double)(d.dwLowDateTime |
			((unsigned long long)d.dwHighDateTime << 32)) * 0.0000001;
	}
	else {
		//  Handle error
		return 0;
	}
}

//  Posix/Linux
#else
#include <time.h>
#include <sys/time.h>
double get_wall_time() {
	struct timeval time;
	if (gettimeofday(&time, NULL)) {
		//  Handle error
		return 0;
	}
	return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
double get_cpu_time() {
	return (double)clock() / CLOCKS_PER_SEC;
}
#endif

#include "cuda_runtime.h"
#include "cuda.h"
//#include "driver_types.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <args.h>
//#include <errno.h>
#include <stdbool.h>
//#include <book.h>
//#include <math.h>

#define TOK_MATRIXHEIGHT "-m"
#define TOK_MATRIXWIDTH "-n"
#define TOK_ONCPU "-c"
#define TOK_BLOCKSHEIGHT "-y"
#define TOK_BLOCKSWIDTH "-x"
#define TOK_THREADSX "-tx"
#define TOK_THREADSY "-ty"
#define DEFAULT_BLOCKSDIM 16
#define DEFAULT_THREADSX 8
#define DEFAULT_THREADSY 4
#define DEFAULT_M 128
#define DEFAULT_N 128
#define DEFAULT_GPU true;

enum Location {HOST, DEVICE};

typedef struct matrix {
	//Location loc;
	int m;
	int n;
	float *contents;
}matrix_t;

#define calc_matrix_offset(_PTR, _M, _N) (_M + _N*_PTR->m)

// Borrowed this macro from Stack Overflow for generating
// custom cuda error messages.
// https://stackoverflow.com/questions/16282136/is-there-a-cuda-equivalent-of-perror
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

matrix_t *alloc_matrix(Location, int, int);

//__host__ __device__ int calc_matrix_offset(matrix_t *, int, int);

__host__ __device__ float get_matrix_item(matrix_t *, int, int);

__host__ __device__ void set_matrix_item(matrix_t *, int, int, float);

void free_matrix(matrix_t *, Location);

void copy_matrix(matrix_t *, matrix_t *, Location);

__host__ __device__ void init_A(matrix_t *, int, int);

__host__ __device__ void init_B(matrix_t *, int, int);

__global__ void setupA(matrix_t *);

__global__ void setupB(matrix_t *);

__global__ void add(matrix_t *, matrix_t *, matrix_t *);

int main(int argc, char **argv) {
	//init
	int m = DEFAULT_M;
	int n = DEFAULT_N;
	int x = DEFAULT_BLOCKSDIM;
	int y = DEFAULT_BLOCKSDIM;
	int tx = DEFAULT_THREADSX;
	int ty = DEFAULT_THREADSY;
	bool run_on_gpu = DEFAULT_GPU;
	matrix_t *a = NULL;
	matrix_t *b = NULL;
	matrix_t *c = NULL;
	matrix_t *dev_a = NULL;
	matrix_t *dev_b = NULL;
	matrix_t *dev_c = NULL;

	//parse command line
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], TOK_MATRIXHEIGHT) == 0) {
			m = atoi(argv[i + 1]);
		}
		if (strcmp(argv[i], TOK_MATRIXWIDTH) == 0) {
			n = atoi(argv[i + 1]);
		}
		if (strcmp(argv[i], TOK_ONCPU) == 0) {
			run_on_gpu = false;
		}
		if (strcmp(argv[i], TOK_BLOCKSWIDTH) == 0) {
			x = atoi(argv[i + 1]);
		}
		if (strcmp(argv[i], TOK_BLOCKSHEIGHT) == 0) {
			y = atoi(argv[i + 1]);
		}
		if (strcmp(argv[i], TOK_THREADSX) == 0) {
			tx = atoi(argv[i + 1]);
		}
		if (strcmp(argv[i], TOK_THREADSY) == 0) {
			ty = atoi(argv[i + 1]);
		}
	}
	dim3 grid = { (unsigned int)x, (unsigned int)y };
	dim3 threads = { (unsigned int)tx, (unsigned int)ty };

	//create matricies
	if (run_on_gpu) {
		// might as well init matricies on the GPU
		dev_a = alloc_matrix(DEVICE, m, n);
		dev_b = alloc_matrix(DEVICE, m, n);
		dev_c = alloc_matrix(DEVICE, m, n);
		setupA KERNEL_ARGS2(grid, threads) (dev_a);
		cudaCheckErrors("main: setupA");
		setupB KERNEL_ARGS2(grid, threads) (dev_b);
		cudaCheckErrors("main: setupB");
	}

	// we need matricies on the host regardless
	a = alloc_matrix(HOST, m, n);
	b = alloc_matrix(HOST, m, n);
	c = alloc_matrix(HOST, m, n);
	// init the host matricies
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			init_A(a, i, j);
			init_B(b, i, j);
		}
	}
	
	//if we're running on the GPU, do the additions
	double gpu_start = 0.0;
	double gpu_end = 0.0;
	
	if (run_on_gpu) {
		//start timing
		gpu_start = get_wall_time();
		//add matricies
		add KERNEL_ARGS2(grid, threads) (dev_a, dev_b, dev_c);
		cudaCheckErrors("main: add");
		//get matricies from GPU
		//copy_matrix(a, dev_a, DEVICE);
		//copy_matrix(b, dev_b, DEVICE);
		copy_matrix(c, dev_c, DEVICE);
		//stop timing
		gpu_end = get_wall_time();
	}

	//need to make a matching matrix on CPU for checking
	matrix_t *check = alloc_matrix(HOST, m, n);

	// will need timing data for cpu, regardless
	//start timing
	double cpu_start = 0.0;
	double cpu_end = 0.0;
	cpu_start = get_wall_time();
	
	//add matricies on CPU
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			float temp = get_matrix_item(a, i, j) + get_matrix_item(b, i, j);
			set_matrix_item(check, i, j, temp);
		}
	}
	//stop timing
	cpu_end = get_wall_time();

	//if we're running on the GPU we check the returns
	if (run_on_gpu) {
		//check matrix addition
		bool checked = true;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				checked &= (get_matrix_item(c, i, j) == get_matrix_item(check, i, j));
			}
		}
		if (checked) {
			printf("We did it!\n\r");
		}
		else {
			printf("Something borked!\n\r");
		}
	}

	//if we're running on the gpu, output the elapsed gpu and cpu time
	if (run_on_gpu) {
		printf("m: %d, n: %d, grid: %d x %d, threads: %d, elapsed gpu time: %f\n\r", \
			m, n, x, y, (int)(tx * ty), gpu_end - gpu_start);
	}
	printf("m: %d, n: %d, grid: %d x %d, threads: %d, elapsed cpu time: %f\n\r", \
		m, n, x, y, (int)(tx * ty), cpu_end - cpu_start);

	free_matrix(a, HOST);
	free_matrix(b, HOST);
	free_matrix(c, HOST);
	free_matrix(dev_a, DEVICE);
	free_matrix(dev_b, DEVICE);
	free_matrix(dev_c, DEVICE);
	free_matrix(check, HOST);

	exit(0);
}

matrix_t *alloc_matrix(Location loc, int m, int n) {
	matrix_t *ret_ptr = NULL;
	if (loc == HOST) {
		ret_ptr = (matrix_t *)calloc(1, sizeof(matrix_t));
		//ret_ptr->loc = loc;
		ret_ptr->m = m;
		ret_ptr->n = n;
		ret_ptr->contents = (float *)calloc(m*n, sizeof(float));
	}
	else {
		matrix_t *temp = NULL;
		temp = (matrix_t *)calloc(1, sizeof(matrix_t));
		//temp->loc = loc;
		temp->m = m;
		temp->n = n;
		cudaMalloc((void **)&temp->contents, sizeof(float)*m*n);
		cudaCheckErrors("alloc_matrix: contents mem");
		cudaMalloc((void **)&ret_ptr, sizeof(matrix_t));
		cudaCheckErrors("alloc_matrix: matrix mem");
		cudaMemcpy((void *)ret_ptr, (void *)temp, sizeof(matrix_t), cudaMemcpyHostToDevice);
		cudaCheckErrors("alloc_matrix: matrix copy");
		free(temp);
	}
	return ret_ptr;
}

/*
__host__ __device__ int calc_matrix_offset(matrix_t *mat, int m, int n)
{
	return m + n*(mat->m);
}
*/

__host__ __device__ float get_matrix_item(matrix_t * mat, int  m, int n)
{
	return mat->contents[calc_matrix_offset(mat, m, n)];
}

__host__ __device__ void set_matrix_item(matrix_t * mat, int m, int n, float f)
{
	mat->contents[calc_matrix_offset(mat, m, n)] = f;
}

void free_matrix(matrix_t *mat, Location loc)
{
	if (loc == HOST){
		free(mat->contents);
		free(mat);
	}
	else {
		matrix_t *temp = NULL;
		temp = (matrix_t *)calloc(1, sizeof(matrix_t));
		cudaMemcpy((void *)temp, (void *)mat, sizeof(matrix_t), cudaMemcpyDeviceToHost);
		cudaCheckErrors("free_matrix: matrix copy");
		cudaFree((void *)temp->contents);
		cudaCheckErrors("free_matrix: free contents");
		cudaFree((void *)mat);
		cudaCheckErrors("free_matrix: matrix free");
		free(temp);
	}
}

void copy_matrix(matrix_t *dest, matrix_t *source, Location source_loc)
{
	if (source_loc == HOST) {
		// ok, dest already exists, so to get access to its contents ptr
		// we need to bring it back to the host
		matrix_t *temp_mat = (matrix_t *)calloc(1, sizeof(matrix_t));
		cudaMemcpy((void *)temp_mat, (void *)dest, sizeof(matrix_t), cudaMemcpyDeviceToHost);
		cudaCheckErrors("copy_matrix: temp_mat back copy");
		float *dev_mat_contents = temp_mat->contents;
		// now that we have the device pointer for its contents, we can copy
		// the contents from the source to the device
		cudaMemcpy((void *)dev_mat_contents, (void *)source->contents, \
			sizeof(float)*source->m*source->n, cudaMemcpyHostToDevice);
		cudaCheckErrors("copy_matrix: copy source contents to dest");
		// likely unnecessary, but just to make sure set the values from the source
		temp_mat->m = source->m;
		temp_mat->n = source->n;
		// ok now we can put it back, not, if we assume that m and n don't change
		// there's no reason to do this last copy
		cudaMemcpy((void *)dest, (void *)temp_mat, sizeof(matrix_t), cudaMemcpyHostToDevice);
		cudaCheckErrors("copy_matrix: update dest mat from temp");
		free(temp_mat);
	}
	else {
		//ok, dealing with a device matrix
		// we need to bring back the mat struct
		matrix_t *temp_mat = (matrix_t *)calloc(1, sizeof(matrix_t));
		cudaMemcpy((void *)temp_mat, (void *)source, sizeof(matrix_t), cudaMemcpyDeviceToHost);
		cudaCheckErrors("copy_matrix: get temp from source");
		// then copy the contents
		cudaMemcpy((void *)dest->contents, (void *)temp_mat->contents, \
			sizeof(float)*temp_mat->m*temp_mat->n, cudaMemcpyDeviceToHost);
		cudaCheckErrors("copy_matrix: copy contents from temp to dest");
		// and set the variables to be safe
		dest->m = temp_mat->m;
		dest->n = temp_mat->n;
		free(temp_mat);
	}
}

__host__ __device__ void init_A(matrix_t *mat, int i, int j)
{
	mat->contents[calc_matrix_offset(mat, i, j)] = 2.0 * i + j + 1.0;
}

__host__ __device__ void init_B(matrix_t *mat, int i, int j)
{
	mat->contents[calc_matrix_offset(mat, i, j)] = i + 4.0 * j + 2.0;
}


__global__ void setupA(matrix_t *mat) {
	const int m = mat->m;
	const int n = mat->n;
	//initial location for the thread
	int dev_m = blockIdx.x * blockDim.x + threadIdx.x;
	int dev_n = blockIdx.y * blockDim.y + threadIdx.y;
	//thread's done when its index goes off the bottom 
	while (dev_n < n){
		// if we're within the range of the array
		// do the init of A
		if ((dev_m < m) && (dev_n < n)) {
			//init_A(mat, dev_m, dev_n);
			mat->contents[calc_matrix_offset(mat, dev_m, dev_n)] = 2.0 * dev_m + dev_n + 1.0;
		}
		// shift the index forward
		dev_m += gridDim.x*blockDim.x;
		// if the new index is outside of the range
		// reset to the start with a mod
		// increment the n dimension i.e. next row
		if (!(dev_m < m)) {
			dev_m = dev_m % m;
			dev_n = dev_n + gridDim.y*blockDim.y;
		}
	}
}

__global__ void setupB(matrix_t *mat) {
	const int m = mat->m;
	const int n = mat->n;
	int dev_m = blockIdx.x * blockDim.x + threadIdx.x;
	int dev_n = blockIdx.y * blockDim.y + threadIdx.y;
	while (dev_n < n) {
		if ((dev_m < m) && (dev_n < n)) {
			//init_B(mat, dev_m, dev_n);
			mat->contents[calc_matrix_offset(mat, dev_m, dev_n)] = dev_m + 4.0 * dev_n + 2.0;
		}
		dev_m += gridDim.x*blockDim.x;
		if (!(dev_m < m)) {
			dev_m = dev_m % m;
			dev_n = dev_n + gridDim.y*blockDim.y;
		}
	}
}

__global__ void add(matrix_t *a, matrix_t *b, matrix_t *c)
{
	const int m = a->m;
	const int n = a->n;
	int dev_m = blockIdx.x * blockDim.x + threadIdx.x; 
	int dev_n = blockIdx.y * blockDim.y + threadIdx.y;
	float item_a = 0;
	float item_b = 0;
	float item_c = 0;
	while (dev_n < n) {
		if ((dev_m < m) && (dev_n < n)) {
			//float temp = get_matrix_item(a, dev_m, dev_n) + get_matrix_item(b, dev_m, dev_n);
			//set_matrix_item(c, dev_m, dev_n, temp);
			item_a = a->contents[calc_matrix_offset(a, dev_m, dev_n)];
			item_b = b->contents[calc_matrix_offset(b, dev_m, dev_n)];
			item_c = item_a + item_b;
			c->contents[calc_matrix_offset(c, dev_m, dev_n)] = item_c;
		}
		dev_m += gridDim.x*blockDim.x;
		if (!(dev_m < m)) {
			dev_m = dev_m % m;
			dev_n = dev_n + gridDim.y*blockDim.y;
		}
	}
}