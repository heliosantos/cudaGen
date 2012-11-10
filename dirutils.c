/** 
 *  @file dirutils.c
 *  @brief Funcoes utilitarias para diretorias e ficheiros
 *  @author 2120916@my.ipleiria.pt
 *  @author 
 *  @author 
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <string.h>

#include "dirutils.h"

bool directoryExists(char *directory)
{
	struct stat st;
	
	//memset(&st, 0, sizeof(st));

	if (stat(directory, &st) == -1)
		return FALSE
		else
		return TRUE;
}


bool createDirectory(char *directory)
{
	if (!directoryExists(directory)) {
		mkdir(directory, 0700);

		return TRUE;
	}

	return FALSE;
}


 char *getDateTime(void)
{
	time_t now;
	struct tm *t;
	char *str;

	str = malloc(25);
	memset(str, '\0', 25);
	now = time(NULL);
	t = localtime(&now);
	strftime(str, 25, "%Y.%m.%d---%Hh%Mm%Ss", t);
	// YYYY.MM.DD---HHhMM.SSs
	
	return str;
}/*
char *getDateTime()
{
 char timestamp[22];
time_t now = time(NULL);
    
    struct tm *t;
    t = localtime(&now);
    
    sprintf(timestamp, "%04d.%02d.%02d---%02dh%02dm%02ds", t->tm_year+1900,t->tm_mon+1,t->tm_mday,t->tm_hour,t->tm_min,t->tm_sec);
   return timestamp; }*/
