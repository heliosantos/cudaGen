/**
 * @file templateGen.c
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "templateGen.h"
#include "main.h"
#include "3rdParty/debug.h"
#include "utils.h"

void store_file_vars(HASHTABLE_T * table, char *unparsedVars){
	char *split = NULL;
	char *key = NULL;	
	
	char *value = NULL;
		
	split = strtok(unparsedVars, "\n");
		
	while(split != NULL){
		int len = strlen(split);	
		if(split[0] == '$' && split[1] == '!' && split[len - 2] == '!' && split[len - 1] == '$'){
				
			if(key != NULL && value != NULL){			
								
				tabela_inserir(table, key, value);
				key = NULL;
				value = NULL;
			}
			
			 key = split;
			 	
		}else{
			if(value == NULL){
				value = (char*)malloc(len + 2);
				value[0] = 0;
			}else{
				value = (char*)realloc(value, strlen(value) + len + 3);
			}
			sprintf(value, "%s\n%s", value, split);
		}
		
		split = strtok(NULL, "\n");
	}
	
	if(key != NULL && value != NULL){
		tabela_inserir(table, key, value);	
	}
}

char *replace_string_with_hashtable_variables(char *template, HASHTABLE_T * table){

	LISTA_GENERICA_T *keys;
	ITERADOR_T *iterador;

	//a list containing all keys of hashtable
	keys = tabela_criar_lista_chaves(table);

	//iterator for the list of keys
	iterador = lista_criar_iterador(keys);

	//for each key, replaces the key in the template for its value
	char *it;
	while ((it = (char *)iterador_proximo_elemento(iterador)) != NULL) {
		char *temp;
		temp = str_replace(template, it, (char *)
				   tabela_consultar(table, it));
		free(template);
		template = temp;
	}	
	
	iterador_destruir(&iterador);
	lista_destruir(&keys);
	return template;
}

void free_matched_vars_from_hashtable(HASHTABLE_T *table, LISTA_GENERICA_T *var_list){
	LISTA_GENERICA_T *keys;
	ITERADOR_T *iterador;

	//a list containing all keys of hashtable
	keys = tabela_criar_lista_chaves(table);

	//iterator for the list of keys
	iterador = lista_criar_iterador(keys);

	//for each key, replaces the mathched value by a empty string
	char *it;
	while ((it = (char *)iterador_proximo_elemento(iterador)) != NULL) {
		
		char *found = (char*)lista_pesquisar(var_list, it, (COMPARAR_FUNC) list_compare_elements);
		if(found != NULL){
			char *value = (char *) tabela_consultar(table, it);			
			value[0] = 0;
		}		
	}
	iterador_destruir(&iterador);
	lista_destruir(&keys);	
}

int list_compare_elements(char *str1, char *str2){
	if(strstr(str1, str2) != NULL){
		return 0;				
	}
	return 1;
}

