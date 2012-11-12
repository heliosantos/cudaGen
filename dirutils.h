/**
 * @file dirutils.h
 * @brief A set of functions that help using the directories
 * 
 * @author 2120916@my.ipleiria.pt
 * @author 2120912@my.ipleiria.pt
 * @author 2120024@my.ipleiria.pt
 *
 * @date 07/11/2012
 * @version 1 
 * 
 */

#ifndef DIRUTILS_H
#define DIRUTILS_H

typedef int bool;

#define TRUE 1;
#define FALSE 0;


int remove_directory(const char *path);

bool directoryExists(char *directory);

void rmkdir(char *directory);

bool createDirectory(char *directory);

void parseGivenName(char *givenName);

void getFilenameFromPath(char *path, char *filename);

#endif

