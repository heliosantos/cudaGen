/**
 * @file utils.h
 * @brief A set of generic utility functions that didn't fit anywhere else
 * 
 * @author 2120916@my.ipleiria.pt
 * @author 2120912@my.ipleiria.pt
 * @author 2120024@my.ipleiria.pt
 *
 * @date 07/11/2012
 * @version 1 
 * 
 */


#ifndef __UTILS_H
#define __UTILS_H
	
char *string_clone(char *str);

char *string_join(int n_args, ...);

char *fileToString(char *fileName);

void stringToFile(char *filename, char *string);

char *str_replace(const char *s, const char *old, const char *new);

char *getDateTime(void);
	
#endif

