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
	char csvString[20];
} Coords3D;

void fill_system_vars_hashtable(HASHTABLE_T *table, char *currentDate, Coords3D *grid_dim, Coords3D *block_dim, char *filename, char *capitalFilename, char *kernelProto, char *userName);

void fill_file_vars_hashtable(HASHTABLE_T *table, char *unparsedVars);

char *replace_string_with_hashtable_variables(char *template, HASHTABLE_T *table);

void free_matched_vars_from_hashtable(HASHTABLE_T *table, LISTA_GENERICA_T *var_list);

int list_compare_elements(char *str1, char *str2);

#endif
