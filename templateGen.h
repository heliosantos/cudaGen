/**
* @file chambel.h
* @brief Header file with prototypes created by chambel to cudaGen main program.
* @date 07-11-2012
* @author 2120912@my.ipleiria.pt
*/

#ifndef __TEMPLATEGEN_H
#define __TEMPLATEGEN_H

#include "3rdParty/hashtables.h"

void fill_file_vars_hashtable(HASHTABLE_T *table, char *unparsedVars);

char *replace_string_with_hashtable_variables(char *template, HASHTABLE_T *table);

void free_matched_vars_from_hashtable(HASHTABLE_T *table, LISTA_GENERICA_T *var_list);

int list_compare_elements(char *str1, char *str2);

#endif
