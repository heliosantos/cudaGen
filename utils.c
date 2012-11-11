/**
* @file: utils.cu
* @author: *** insert name **
* @created: *** 2012.11.11---16h11m16s ***
* @comment 
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "3rdParty/debug.h"

#include "utils.h"

char *string_clone(char *str){
	if(str == NULL){
		return NULL;
	}
	char *clone = (char*)malloc(strlen(str) + 1);
	strcpy(clone, str);
	return clone;
}


/**
*
* Stores a string in a file
*
*/
void stringToFile(char *filename, char *string){
	FILE *fptr = NULL;
	if ((fptr = fopen(filename, "w")) == NULL){
		ERROR(3, "Can't open file to write");
	}
	fprintf(fptr, "%s", string);
	fclose(fptr);
}

/**
*
* Stores a file in a string
*
*http://stackoverflow.com/questions/1285097/how-to-copy-text-file-to-string-in-c
*
*/
char *fileToString(char *fileName)
{
	long f_size;
	char *code;
	size_t code_s;
	FILE *fp = NULL;
	
	if((fp = fopen(fileName, "r")) == NULL){
		ERROR(3,"Can't open file to read");
	}
	
	fseek(fp, 0, SEEK_END);
	f_size = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	code_s = sizeof(char) * f_size + 1;
	code = malloc(code_s);
	code[code_s - 1] = 0;
	fread(code, 1, f_size, fp);
	fclose(fp);
	return code;
}


/**
*
* Replaces all occurences of string A by string B on string C
*
*http://stackoverflow.com/questions/3659694/how-to-replace-substring-in-c
*
*/
char *str_replace(const char *s, const char *old, const char *new)
{
	size_t slen = strlen(s) + 1;
	char *cout = malloc(slen), *p = cout;
	if (!p)
		return 0;
	while (*s)
		if (!strncmp(s, old, strlen(old))) {
			p = (char *)(p - cout);
			cout = realloc(cout, slen += strlen(new) - strlen(old));
			p = cout + (long)p;
			p += strlen(strcpy(p, new));
			s += strlen(old);
		} else
			*p++ = *s++;
	*p = 0;
	return cout;
}


