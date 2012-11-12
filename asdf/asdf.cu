#include <cuda.h>
#include <stdio.h>
#include "HandleError.h"

#define N 32768

/*-------------------------------------------------------------------
* Wrapper for stdlib malloc function
*-----------------------------------------------------------------*/
void *MyMalloc(size_t size, const int line, const char *file)
{
	void *ptr = malloc(size);
	if (ptr == NULL) {
		fprintf(stderr, "[%d@%s][ERROR] can't malloc %u bytes\n",
			line, file, size);
		exit(EXIT_FAILURE);
	}
	return ptr;
}

/*-------------------------------------------------------------------* MY_MALLOC macro.
* Wrapping macro for MyMalloc function (provides "file" and "line" 
* parameters).
*-----------------------------------------------------------------*/
#define MY_MALLOC(size) MyMalloc((size),__LINE__,__FILE__)

/*-------------------------------------------------------------------
* Wrapper for stdlib free function
* @param ptr [IN] ptr to free
* @param line [IN] line of source code where function is called
* @param file [IN] name of source file where function is called
* @return NULL
*-----------------------------------------------------------------*/
void *MyFree(void *ptr, const int line, const char *file)
{
	if (ptr != NULL) {
		free(ptr);
	} else {
		fprintf(stderr, "[%d@%s][ERROR] can't free NULL pointer\n",
			line, file);
		exit(EXIT_FAILURE);
	}
	return NULL;
}

/*-------------------------------------------------------------------
* MY_FREE macro.
* Wrapping macro for MyFree function (provides "file" and "line" 
* parameters).
* @param ptr [IN] ptr to free
* @return on error (NULL ptr to free), the function terminates 
* the process.
*-----------------------------------------------------------------*/
#define MY_FREE(ptr) MyFree((ptr),__LINE__,__FILE__)

float VectorSum(float *vector, int n)
{

	float total = 0;

	for (int i = 0; i < N; i++) {
		total += vector[i];
	}
	return total;
}

__global__ void DefaultKernel(float *a_dev, float *b_dev, float *c_dev, int n)
{
	int stride =
	    (blockDim.x * blockDim.y * blockDim.z) * (gridDim.x * gridDim.y *
						      gridDim.z);

	int tid =
	    blockIdx.z * gridDim.x * gridDim.y * blockDim.x * blockDim.y *
	    blockDim.z +
	    blockIdx.y * gridDim.x * blockDim.x * blockDim.y * blockDim.z +
	    blockIdx.x * blockDim.x * blockDim.y * blockDim.z +
	    threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
	    threadIdx.x;

	if (tid == 0) {
		printf("gridDim.x = %d\n", gridDim.x);
		printf("gridDim.y = %d\n", gridDim.y);
		printf("gridDim.z = %d\n", gridDim.z);
		printf("blockDim.x = %d\n", blockDim.x);
		printf("blockDim.y = %d\n", blockDim.y);
		printf("blockDim.z = %d\n", blockDim.z);
	}

	while (tid < n) {
		c_dev[tid] = a_dev[tid] + b_dev[tid];
		tid += stride;
	}
}

int main(void)
{


	float *a_host, *b_host, *c_host;
	float *a_dev, *b_dev, *c_dev;

	/* Host input data allocation */
	a_host = (float *)MY_MALLOC(N * sizeof(float));
	b_host = (float *)MY_MALLOC(N * sizeof(float));
	/* Host output data allocation */
	c_host = (float *)MY_MALLOC(N * sizeof(float));

	/* Device input data allocation */
	HANDLE_ERROR(cudaMalloc((void **)&a_dev, N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void **)&b_dev, N * sizeof(float)));
	/* Device output data allocation */
	HANDLE_ERROR(cudaMalloc((void **)&c_dev, N * sizeof(float)));

	/* Initialize host input data */
	for (int i = 0; i < N; i++) {
		a_host[i] = b_host[i] = i + 1;
		c_host[i] = 0;
	}
	/* Copy data from Host to Device */
	HANDLE_ERROR(cudaMemcpy
		     (a_dev, a_host, N * sizeof(float),
		      cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy
		     (b_dev, b_host, N * sizeof(float),
		      cudaMemcpyHostToDevice));
	/* cleans the memory of the previows execution */
	HANDLE_ERROR(cudaMemcpy
		     (c_dev, c_host, N * sizeof(float),
		      cudaMemcpyHostToDevice));

	//dim3 NumBlocks (65535, 65535, 65535);
	dim3 NumBlocks(1);

	//for 1.x x<=512, y<=512, z<=64, x*y*z<=512
	//for 2.x and 3.0 x<=1024, y<=1024, z<=64, x*y*z<=1024
	dim3 ThreadsPerBlock(1);


	
	DefaultKernel <<< NumBlocks, ThreadsPerBlock >>> (a_dev, b_dev, c_dev, N);


	

	HANDLE_ERROR(cudaMemcpy
		     (c_host, c_dev, N * sizeof(float),
		      cudaMemcpyDeviceToHost));

	int nErrors = 0;
	for (int i = 0; i < N; i++) {
		//printf("%f + %f = %f\n", a_host[i], b_host[i], c_host[i]);
		if (a_host[i] + b_host[i] != c_host[i]) {
			nErrors++;
		}
	}
	if (nErrors == 0) {
		printf("success\n");
	} else {
		printf("%d errors occoured\n", nErrors);
	}

	/* Free allocated resources */

	MY_FREE(a_host);
	MY_FREE(b_host);
	MY_FREE(c_host);
	HANDLE_ERROR(cudaFree(a_dev));
	HANDLE_ERROR(cudaFree(b_dev));
	HANDLE_ERROR(cudaFree(c_dev));
	return 0;
}
