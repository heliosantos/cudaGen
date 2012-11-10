/** 
 *  @file main.h
 *  @brief O ficheiro header para a unidade principal do Cudagen
 *  @author 2120916@my.ipleiria.pt
 *  @author 
 *  @author 
 */

#ifndef MAIN_H
#define MAIN_H

#include "cmdline.h"
#include "3rdParty/hashtables.h"

typedef struct Coords3D_s {
	int x;
	int y;
	int z;
	char sx[6];
	char sy[6];
	char sz[6];
} Coords3D;

int createKernel(char *outputDir, char *kernelName, int geometry);

char *fileToString(char *fileName);

char *str_replace(const char *s, const char *old, const char *new);

int fill_grid_dim(Coords3D *grid_dim, struct gengetopt_args_info *args_info);

int fill_block_dim(Coords3D *block_dim, struct gengetopt_args_info *args_info);

void fill_default_template_list(HASHTABLE_T *tabela, Coords3D *grid_dim, Coords3D *block_dim);

#endif
