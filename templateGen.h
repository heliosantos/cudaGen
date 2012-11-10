/**
* @file chambel.h
* @brief Header file with prototypes created by chambel to cudaGen main program.
* @date 07-11-2012
* @author 2120912@my.ipleiria.pt
*/

#ifndef __TEMPLATEGEN_H
#define __TEMPLATEGEN_H

#include "3rdParty/hashtables.h"

typedef struct Coords3D_s {
	int x;
	int y;
	int z;
	char sx[6];
	char sy[6];
	char sz[6];
} Coords3D;


void fill_cu_main_template_hashtable(HASHTABLE_T *tabela, Coords3D *grid_dim, Coords3D *block_dim);

void fill_cu_proto_template_hashtable(HASHTABLE_T * tabela, char *kernelName, char *filename, char *currentDate);

void fill_header_template_hashtable(HASHTABLE_T * tabela, char *filename, char *capitalFilename, char *currentDate);

void fill_c_main_template_hashtable(HASHTABLE_T * tabela, char *filename, char *currentDate);

char *replace_string_with_template_variables(char *template, HASHTABLE_T * tabela);

#endif
