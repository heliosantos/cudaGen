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
	int x = 1;
	int y = 1;
	int z = 1;
} Coords3D;

int createKernel(char *outputDir, char *kernelName, int geometry);

static void iter(const char *key, const char *value, const void *obj);

char *fileToString(char *fileName);

char *str_replace(const char *s, const char *old, const char *new);

void freeElement(char *element);

#endif
