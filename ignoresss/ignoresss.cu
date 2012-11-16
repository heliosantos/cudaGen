 /* template copied from:
 * High Performance Computing / Computação de Alto Desempenho – Practical CUDA 
 *	 Exercise sheet #2 – CUDA
 *	 http://www.estg.ipleiria.pt
 */


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

/*-------------------------------------------------------------------
* MY_MALLOC macro.
* Wrapping macro for MyMalloc function (provides "file" and "line" 
* parameters).
*-----------------------------------------------------------------*/
#define MY_MALLOC(size) MyMalloc((size),__LINE__,__FILE__)

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;
    
    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
        { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
        { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
        { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        {   -1, -1 }
    };
    
    int index = 0;
    
    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }
        
        index++;
    }
    
    // If we don't find the values, we default use the previous one to run properly
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[7].Cores);
    return nGpuArchCoresPerSM[7].Cores;
}
// end of GPU Architecture definitions



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

__global__ void Kernel(float *a_dev, float *b_dev, float *c_dev, int n)
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


	while (tid < n) {
		c_dev[tid] = a_dev[tid] + b_dev[tid];
		tid += stride;
	}
}

int main(void)
{


	float *a_host, *b_host, *c_host;
	float *a_dev, *b_dev, *c_dev;


	// Print hardware
	cudaDeviceProp prop;
	  int count;
	  cudaGetDeviceCount (&count);
	  for (int i = 0; i < count; i++)
	    {
	      cudaGetDeviceProperties (&prop, i);
	      printf (" --- Information for device %d ---\n", i);
	      printf ("Name: %s\n", prop.name);
	      printf ("Compute capability: %d.%d\n", prop.major, prop.minor);
	      printf ("Clock rate: %d\n", prop.clockRate);
	      printf ("Device copy overlap: ");
	      if (prop.deviceOverlap)
		printf ("Enabled\n");
	      else
		printf ("Disabled\n");
	      printf ("Kernel execution timeout : ");
	      if (prop.kernelExecTimeoutEnabled)
		printf ("Enabled\n");
	      else
		printf ("Disabled\n");
	      printf (" --- Memory Information for device %d ---\n", i);
	      printf ("Total global mem: %ld\n", prop.totalGlobalMem);
	      printf ("Total constant Mem: %ld\n", prop.totalConstMem);
	      printf ("Max mem pitch: %ld\n", prop.memPitch);
	      printf ("Texture Alignment: %ld\n", prop.textureAlignment);
	      printf (" --- MP Information for device %d ---\n", i);
	      printf ("Multiprocessor count: %d\n", prop.multiProcessorCount);
	      printf ("(%2d) Multiprocessors x (%3d) CUDA Cores/MP:    %d CUDA Cores\n",
		      prop.multiProcessorCount,
		      _ConvertSMVer2Cores(prop.major, prop.minor),
		      _ConvertSMVer2Cores(prop.major, prop.minor) * prop.multiProcessorCount);
	      printf ("Shared mem per mp: %ld\n", prop.sharedMemPerBlock);
	      printf ("Registers per mp: %d\n", prop.regsPerBlock);
	      printf ("Threads in warp: %d\n", prop.warpSize);
	      printf ("Max threads per block: %d\n", prop.maxThreadsPerBlock);
	      printf ("Max thread dimensions: (%d, %d, %d)\n",
		      prop.maxThreadsDim[0], prop.maxThreadsDim[1],
		      prop.maxThreadsDim[2]);
	      printf ("Max grid dimensions: (%d, %d, %d)\n",
		      prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	      printf ("\n");
	    }


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
	dim3 NumBlocks(1,1,2);

	//for 1.x x<=512, y<=512, z<=64, x*y*z<=512
	//for 2.x and 3.0 x<=1024, y<=1024, z<=64, x*y*z<=1024
	dim3 ThreadsPerBlock(1,2,3);


	
	Kernel <<< NumBlocks, ThreadsPerBlock >>> (a_dev, b_dev, c_dev, N);


	

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


