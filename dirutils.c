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
#include <sys/time.h>

#include "dirutils.h"

bool directoryExists(char *directory)
{
	struct stat st = { 0 };

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

char *getDateTime()
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
}
