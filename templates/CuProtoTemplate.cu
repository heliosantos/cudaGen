/**
* @file: $!FILENAME!$.cu
* @author: *** insert name **
* @created: *** $!C_DATE!$ ***
* @comment 
*/

#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>

#include "$!FILENAME!$.h"

__global__ void $!KERNEL_PROTO!$
{
/*
* gridDim.x or .y or .z: size, in blocks, of calling grid
* blockDim.x or .y or .z: size, in number of threads, of calling block
* blockIdx.x or .y or .z: index of current block
* threadIdx.x or .y or .z: index of current thread
*/

/* Compute total number of threads per grid */
	int TotalThreads = (blockDim.x * blockDim.y) * (gridDim.x * gridDim.y);
/* tid is the thread working index */
	int tid = blockIdx.y * gridDim.x * blockDim.x * blockDim.y	/*complete rows */
	    + blockIdx.x * blockDim.x * blockDim.y	/* complete blocks */
	    + threadIdx.y * blockDim.x + threadIdx.x;
/* INSERT CODE HERE */
}

int main(int argc, char *argv[])
{
	/* Variable declaration */

	/* Disable warnings */
	(void)argc;
	(void)argv;


	//dim3 NumBlocks(1, 1);
	//dim3 ThreadsPerBlock(1, 1);

		
	//$!KERNEL_PROTO!$;
		
	/*Insert code here */

	return 0;
}

