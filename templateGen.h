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

typedef struct MultilineString_s{
	char **line;
	int numberOfLines;
} MultilineString;


void freeMultiLineString(MultilineString *multilineString);

void fill_cu_main_template_hashtable(HASHTABLE_T *table, Coords3D *grid_dim, Coords3D *block_dim);

void fill_cu_proto_template_hashtable(HASHTABLE_T * table, char *kernelName, char *filename, char *currentDate);

void fill_header_template_hashtable(HASHTABLE_T *table, char *filename, char *capitalFilename, char *currentDate);

void fill_c_main_template_hashtable(HASHTABLE_T *table, char *filename, char *currentDate);

void fill_system_vars_hashtable(HASHTABLE_T *table, char *currentDate, Coords3D *grid_dim, Coords3D *block_dim, char *filename, char *capitalFilename, char *kernelProto, char *userName);

void fill_user_vars_hashtable(HASHTABLE_T *table, char *unparsedVars);

char *replace_string_with_template_variables(char *template, HASHTABLE_T *table);

char *replace_string_with_template_multiline_variables(char *template, HASHTABLE_T * table);

#endif
