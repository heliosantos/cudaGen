/**
 * @file utils.h
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


#ifndef __MEMORY_H
#define __MEMORY_H

void *MyMalloc(size_t size, const int line, const char *file);

/*-------------------------------------------------------------------
* MY_MALLOC macro.
* Wrapping macro for MyMalloc function (provides "file" and "line" 
* parameters).
*-----------------------------------------------------------------*/
#define MY_MALLOC(size) MyMalloc((size),__LINE__,__FILE__)

void *MyFree(void *ptr, const int line, const char *file);

/*-------------------------------------------------------------------
* MY_FREE macro.
* Wrapping macro for MyFree function (provides "file" and "line" 
* parameters).
* @param ptr [IN] ptr to free
* @return on error (NULL ptr to free), the function terminates 
* the process.
*-----------------------------------------------------------------*/
#define MY_FREE(ptr) MyFree((ptr),__LINE__,__FILE__)


	
#endif

