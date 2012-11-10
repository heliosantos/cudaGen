/**
 * @file chambel.c
 * @brief Code file with functions created by chambel to cudaGen main program.
 *
 * Functions to handle -p, -r, -d and -F options
 *
 * @todo code review
 * 
 * @author 2120912@my.ipleiria.pt
 * @date 07/11/2012
 * @version 1 
 * @par Nothing done yet
 * 
 * Separated code by author
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <time.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>

#include "templateGen.h"
#include "cudaGen.h"
#include "dirutils.h"

#include "debug.h"


/**
 * Function to create template directory and template header file
 *
 * @param force user option to create folder
 * @param dirname directory name to be created
 * @param path path of directory
 * @return result (TRUE or FALSE)
 * @see 
 */
void createDirectoryAndHeaderFile(int force,char *dirname, char **path)
{
  
  char timestamp[22];
    
  if(!force && directoryExists(dirname)){
    
    time_t now = time(NULL);
    
    struct tm *t;
    t = localtime(&now);
    
    sprintf(timestamp, "%04d.%02d.%02d---%02dh%02dm%02ds", t->tm_year+1900,t->tm_mon+1,t->tm_mday,t->tm_hour,t->tm_min,t->tm_sec);
    
    if((*path = (char*)malloc((strlen(dirname)+strlen(timestamp) + 1)*sizeof(char) + 1)) == NULL)
      ERROR(5,"Can't malloc");
    
  }else{
    if((*path = (char*)malloc((strlen(dirname))*sizeof(char) + 1)) == NULL)
      ERROR(5,"Can't malloc");
  }
  
  strcpy(*path, dirname);
  
  if(!force && directoryExists(dirname)){ 
    strcpy(*path + strlen(dirname),timestamp);
  }else if(force && directoryExists(dirname)){
    remove_directory(*path);
  }
  
  mkdir(*path, 0755);
  
  createHeaderTemplate(dirname, *path);
  
}



/**
 * Function to create header file template
 * @param dirname directory name to create header file name
 * @param path path of directory where user wants to create files
 * @return result (TRUE or FALSE)
 * @see 
 */
void createHeaderTemplate(char *dirname, char *path)
{
  
  char *filename;
  
  if((filename=(char*)malloc((strlen(path)+strlen(dirname)+4)*sizeof(char))) == NULL)
    ERROR(5,"Can't malloc");
  
  char *symbol;
  
  if((symbol=(char*)malloc((strlen(dirname)+5)*sizeof(char))) == NULL)
    ERROR(5,"Can't malloc");

  sprintf(filename,"%s/%s.h", path, dirname);  
  sprintf(symbol,"__%s_H",dirname);
  
  FILE *fptr=NULL;
  DIR *dptr;
  
  if ((dptr = opendir(path)) == NULL)
    ERROR(4,"Can't open dir to write");
  
  rewinddir(dptr);
 
  if((fptr=fopen(filename,"w"))==NULL)
    ERROR(3,"Can't open file to write");
  
  
  fprintf(fptr,"#ifndef %s\n",symbol);
  fprintf(fptr,"#define %s\n\n",symbol);
  fprintf(fptr,"#endif\n");
  
  fclose(fptr);
  
  fptr=NULL;
  
  closedir(dptr);
  
}

/**
 * Function to create code files to static templates
 * @param dirname directory name to create code files names
 * @param path path of directory where user wants to create files
 * @param cudaTemplateByDefault handle user request for regular C template or custom kernel prototype template
 * @param kernelProto custom kernel prototype to create template
 * @return result (TRUE or FALSE)
 * @see 
 */
void generateStaticTemplate(char *dirname, char *path, int cudaTemplateByDefault, char *kernelProto)
{
  
  char *ext;
  
  if(cudaTemplateByDefault)
  {
    if(!createErrorFile(path)){
      ERROR(4,"Can't open dir to write");
    }
    ext = ".cu";
  }
  else
  {
    ext = ".c";
  }
  
  char *filename;
  if((filename = (char*)malloc((strlen(path)+strlen(dirname)+strlen(ext)+2)*sizeof(char))) == NULL)
    ERROR(5,"Can't malloc");

  
  sprintf(filename,"%s/%s%s", path, dirname, ext);
  
  FILE *fptr=NULL;
  DIR *dptr;
  
 if ((dptr = opendir(path)) == NULL)
    ERROR(4,"Can't open dir to write");
 
  rewinddir(dptr);
  
  if((fptr=fopen(filename,"w"))==NULL)
    ERROR(3,"Can't open file to write");
  
  fprintf(fptr,"/**\n");
  fprintf(fptr,"* @file: %s%s\n",dirname,ext);
  fprintf(fptr,"* @author: *** insert name **\n");
  fprintf(fptr,"* @created: *** insert date ***\n");
  fprintf(fptr,"* @comment \n");
  fprintf(fptr,"*/\n\n");
  fprintf(fptr,"#include <stdio.h>\n");
  fprintf(fptr,"#include <stdlib.h>\n\n");
  
  if(cudaTemplateByDefault)
  {
    fprintf(fptr,"#include <cuda.h>\n\n");
  }
  
  fprintf(fptr,"#include \"%s.h\"\n\n",dirname);
  
  if(cudaTemplateByDefault)
  {
  
    fprintf(fptr,"__global__ void %s\n",kernelProto);
    fprintf(fptr,"{\n");
    fprintf(fptr,"/*\n");
    fprintf(fptr,"* gridDim.x or .y or .z: size, in blocks, of calling grid\n");
    fprintf(fptr,"* blockDim.x or .y or .z: size, in number of threads, of calling block\n");
    fprintf(fptr,"* blockIdx.x or .y or .z: index of current block\n");
    fprintf(fptr,"* threadIdx.x or .y or .z: index of current thread\n");
    fprintf(fptr,"*/\n\n");
    fprintf(fptr,"/* Compute total number of threads per grid */\n");
    fprintf(fptr,"int TotalThreads = (blockDim.x*blockDim.y) * (gridDim.x*gridDim.y);\n");
    fprintf(fptr,"/* tid is the thread working index */\n");
    fprintf(fptr,"int tid = blockIdx.y*gridDim.x*blockDim.x*blockDim.y /*complete rows */\n");
    fprintf(fptr,"+ blockIdx.x * blockDim.x*blockDim.y /* complete blocks */\n");
    fprintf(fptr,"+ threadIdx.y * blockDim.x + threadIdx.x;\n");
    fprintf(fptr,"/* INSERT CODE HERE */\n");
    fprintf(fptr,"}\n\n\n");
  }
    
  fprintf(fptr,"int main(int argc, char *argv[])\n{\n");
  
  fprintf(fptr,"\t/* Variable declaration */\n\n");
  fprintf(fptr,"\t/* Disable warnings */\n");
  fprintf(fptr,"\t(void)argc; (void)argv;\n\n");
  fprintf(fptr,"\t/*Insert code here*/\n\treturn 0;\n}\n");
  
  
  
  fclose(fptr);
  fptr=NULL;
  
  closedir(dptr);
  
}

/**
 * Function to create HandleError.h file to cuda templates
 * @param path path of directory where user wants to create files
 * @return result (TRUE or FALSE)
 * @see 
 */
int createErrorFile(char *path)
{
  char *error = "HandleError.h";
  char *filename;
  
  if((filename=malloc((strlen(path)+strlen(error)+2)*sizeof(char))) == NULL)
    return FALSE;
  
  sprintf(filename,"%s/%s", path, error);
    
  FILE *fptr=NULL;
  DIR *dptr;
  
  if ((dptr = opendir(path)) == NULL)
    return FALSE;
  
  rewinddir(dptr);
  
  if((fptr=fopen(filename,"w"))==NULL)
    return FALSE;
  
  fprintf(fptr,"#ifndef __HANDLE_ERROR_H__\n");
  fprintf(fptr,"#define __HANDLE_ERROR_H__\n");
  fprintf(fptr,"/*-------------------------------------------------------------------\n");
  fprintf(fptr,"* Function to process CUDA errors\n");
  fprintf(fptr,"* @param err [IN] CUDA error to process (usually the code returned\n");
  fprintf(fptr,"*	 by the cuda function)\n");
  fprintf(fptr,"* @param line [IN] line of source code where function is called\n");
  fprintf(fptr,"* @param file [IN] name of source file where function is called\n");
  fprintf(fptr,"* @return on error, the function terminates the process with \n");
  fprintf(fptr,"* 		EXIT_FAILURE code.\n");
  fprintf(fptr,"* source: \"CUDA by Example: An Introduction to General-Purpose \"\n");
  fprintf(fptr,"* GPU Programming\", Jason Sanders, Edward Kandrot, NVIDIA, July 2010\n");
  fprintf(fptr,"* @note: the function should be called through the \n");
  fprintf(fptr,"* 	macro 'HANDLE_ERROR'\n");
  fprintf(fptr,"*------------------------------------------------------------------*/\n");
  fprintf(fptr,"static void HandleError( cudaError_t err,\n");
  fprintf(fptr,"                         const char *file,\n");
  fprintf(fptr,"                         int line ) {\n");
  fprintf(fptr,"    if (err != cudaSuccess) \n");
  fprintf(fptr,"    {\n");
  fprintf(fptr,"        printf( \"[ERROR] '%s' (%s) in '%s' at line '%s'\n\",\n","%s" ,"%d" ,"%s","%d");
  fprintf(fptr,"		cudaGetErrorString(err),err,file,line);\n");
  fprintf(fptr,"        exit( EXIT_FAILURE );\n");
  fprintf(fptr,"    }\n");
  fprintf(fptr,"}\n");
  fprintf(fptr,"/* The HANDLE_ERROR macro */\n");
  fprintf(fptr,"#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__ ))\n");
  fprintf(fptr,"\n");
  fprintf(fptr,"#endif\n");
    
  fclose(fptr);
  fptr=NULL;
  
  closedir(dptr);
  
  return TRUE;
}
