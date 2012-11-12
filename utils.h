/**
* @file: utils.h
* @author: *** insert name **
* @created: *** 2012.11.11---16h11m16s ***
* @comment 
*/


#ifndef __UTILS_H
#define __UTILS_H
	
char *string_clone(char *str);

char *string_join(int n_args, ...);

char *string_join(int i, ...);

char *fileToString(char *fileName);

void stringToFile(char *filename, char *string);

char *str_replace(const char *s, const char *old, const char *new);

	
#endif
