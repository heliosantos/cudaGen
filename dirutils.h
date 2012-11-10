/** 
 *  @file dirutils.h
 *  @brief O ficheiro header para a unidade de ferramentas de diretorias e ficheiros
 *  @author 2120916@my.ipleiria.pt
 *  @author 
 *  @author 
 */



#ifndef DIRUTILS_H
#define DIRUTILS_H

typedef int bool;

#define TRUE 1;
#define FALSE 0;

bool directoryExists(char *directory);
bool createDirectory(char *directory);
char *getDateTime(void);

#endif
