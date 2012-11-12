/**
 * @file utils.c
 * @brief A set of generic memory handle functions
 * 
 * @author 2120916@my.ipleiria.pt
 * @author 2120912@my.ipleiria.pt
 * @author 2120024@my.ipleiria.pt
 *
 * @date 07/11/2012
 * @version 1 
 *
 * functions and macros copied from:
 * High Performance Computing / Computação de Alto Desempenho – Practical CUDA 
 *	 Exercise sheet #2 – CUDA
 *	 http://www.estg.ipleiria.pt
 *
 */

/*-------------------------------------------------------------------
* Wrapper for stdlib malloc function
*
* function copied from:
* 	 High Performance Computing / Computação de Alto Desempenho – Practical CUDA 
*	 Exercise sheet #2 – CUDA
*	 http://www.estg.ipleiria.pt
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
*
* macro copied from:
* 	 High Performance Computing / Computação de Alto Desempenho – Practical CUDA 
*	 Exercise sheet #2 – CUDA
*	 http://www.estg.ipleiria.pt
*-----------------------------------------------------------------*/
#define MY_MALLOC(size) MyMalloc((size),__LINE__,__FILE__)

/*-------------------------------------------------------------------
* Wrapper for stdlib free function
* @param ptr [IN] ptr to free
* @param line [IN] line of source code where function is called
* @param file [IN] name of source file where function is called
* @return NULL
*
* function copied from:
* 	 High Performance Computing / Computação de Alto Desempenho – Practical CUDA 
*	 Exercise sheet #2 – CUDA
*	 http://www.estg.ipleiria.pt
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
*
*
* macro copied from:
* 	 High Performance Computing / Computação de Alto Desempenho – Practical CUDA 
*	 Exercise sheet #2 – CUDA
*	 http://www.estg.ipleiria.pt
*-----------------------------------------------------------------*/
#define MY_FREE(ptr) MyFree((ptr),__LINE__,__FILE__)


