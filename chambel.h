/**
* @file chambel.h
* @brief Header file with prototypes created by chambel to cudaGen main program.
* @date 07-11-2012
* @author 2120912@my.ipleiria.pt
*/

#ifndef __chambel_H
#define __chambel_H

char *parseGivenName(char *givenName);
void createDirectoryAndHeaderFile(int force, char *dirname, char **path);
int dirExists(char *dirname);
int remove_directory(const char *path);
void createHeaderTemplate(char *dirname, char *path);
void generateStaticTemplate(char *dirname, char *path, int cudaTemplateByDefault, char *kernelProto);
int createErrorFile(char *path);

#endif
