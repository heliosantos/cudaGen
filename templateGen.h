/**
 * @file templateGen.h
 * @brief A set of functions that help using the templates
 * 
 * @author 2120916@my.ipleiria.pt
 * @author 2120912@my.ipleiria.pt
 * @author 2120024@my.ipleiria.pt
 *
 * @date 07/11/2012
 * @version 1 
 * 
 */

#ifndef __TEMPLATEGEN_H
#define __TEMPLATEGEN_H

#include "3rdParty/hashtables.h"

void store_file_vars(HASHTABLE_T *table, char *unparsedVars);

char *replace_string_with_hashtable_variables(char *template, HASHTABLE_T *table);

void free_matched_vars_from_hashtable(HASHTABLE_T *table, LISTA_GENERICA_T *var_list);

int list_compare_elements(char *str1, char *str2);

#endif
