/** 
 *  @file main.h
 *  @brief O ficheiro header para a unidade principal do Cudagen
 *  @author 2120916@my.ipleiria.pt
 *  @author 
 *  @author 
 */

#ifndef MAIN_H
#define MAIN_H

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

#endif
