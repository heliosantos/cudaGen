/**
* @file: str.cu
* @author: *** insert name **
* @created: *** 2012.11.12---09h51m22s ***
* @comment 
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "str.h"


int main(int argc, char *argv[])
{
	/* Variable declaration */

	/* Disable warnings */
	(void)argc;
	(void)argv;

	/*Insert code here */
	
	char * str = string_join(3, "um ", "dois ", "tres");
	
	printf("_%s_\n", str);

	return 0;
}

char *string_join(int n_args, ...){
	register int i;
        char *result = "";
        va_list ap;
        int length = 0;

        va_start(ap, n_args);
               
        for(i = 1; i <= n_args; i++) {
        	char *current = va_arg(ap, char*);
        	length +=  strlen(current);
        } 
        va_end(ap);
        
        va_start(ap, n_args);       
        result = (char*)malloc(length + 1);
        result[0] = 0;        
        for(i = 1; i <= n_args; i++) {
        	char *current = va_arg(ap, char*);
        	//strcat(result, current);
        	sprintf(result, "%s%s", result, current);
        }

        va_end(ap);
        return result;
}

