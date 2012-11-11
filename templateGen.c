/**
 * @file chambel.c
 * @brief Code file with functions created by chambel to cudaGen main program.
 *
 * Functions to handle -p, -r, -d and -F options
 *
 * @todo code review
 * 
 * @author 2120912@my.ipleiria.pt
 * @date 07/11/2012
 * @version 1 
 * @par Nothing done yet
 * 
 * Separated code by author
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "templateGen.h"
#include "main.h"
#include "3rdParty/debug.h"


void freeMultiLineString(MultilineString *multilineString){
	int i = 0;
	for(i = 0; i < multilineString->numberOfLines; i++){
		free(multilineString->line[i]);
	}
	free(multilineString->line);
	free(multilineString);
}

void fill_user_vars_hashtable(HASHTABLE_T * table, char *unparsedVars){
	char *split = NULL;
	char *key = NULL;	
	
	MultilineString *value = NULL;
		
	split = strtok(unparsedVars, "\n");
		
	while(split != NULL){
		int len = strlen(split);	
		if(split[0] == '$' && split[1] == '!' && split[len - 2] == '!' && split[len - 1] == '$'){
				
			if(key != NULL && value != NULL){			
								
				tabela_inserir(table, key, value);
				key = NULL;
				value = NULL;
			}
			
			 if((value = (MultilineString *)malloc(sizeof(MultilineString))) == NULL){
			 	ERROR(4, "Failed Malloc");
			 }
			 value->numberOfLines = 0;
			 key = split;
			 	
		}else{
			if(value->numberOfLines == 0){
				value->numberOfLines++;
				value->line = (char**)malloc(sizeof(char*) * value->numberOfLines);
			}else{
				value->numberOfLines++;
				value->line = realloc(value->line, sizeof(char*) * value->numberOfLines);
			}
						
			value->line[value->numberOfLines - 1] = (char*)malloc(len + 1);
			strcpy(value->line[value->numberOfLines - 1], split);
		}
		
		split = strtok(NULL, "\n");
	}
	
	if(key != NULL && value != NULL){
		tabela_inserir(table, key, value);
	}
}


void fill_system_vars_hashtable(HASHTABLE_T * table, char *currentDate, Coords3D *grid_dim, Coords3D *block_dim, char *filename, char *capitalFilename, char *kernelProto, char *userName){
	tabela_inserir(table, "$!FILENAME!$", filename);
	tabela_inserir(table, "$!CAPITAL_FILENAME!$", capitalFilename);
	tabela_inserir(table, "$!C_DATE!$", currentDate);	
	tabela_inserir(table, "$!BX!$", grid_dim->sx);
	tabela_inserir(table, "$!BY!$", grid_dim->sy);
	tabela_inserir(table, "$!BZ!$", grid_dim->sz);
	tabela_inserir(table, "$!TX!$", block_dim->sx);
	tabela_inserir(table, "$!TY!$", block_dim->sy);
	tabela_inserir(table, "$!TZ!$", block_dim->sz);
	tabela_inserir(table, "$!KERNEL_PROTO!$", kernelProto);
	tabela_inserir(table, "$!USER_NAME!$", userName);
}

char *replace_string_with_template_variables(char *template, HASHTABLE_T * table){

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

char *replace_string_with_template_multiline_variables(char *template, HASHTABLE_T * table){
	int i = 0;
	int len = 0;
	
	LISTA_GENERICA_T *keys;
	ITERADOR_T *iterador;

	//a list containing all keys of hashtable
	keys = tabela_criar_lista_chaves(table);

	//iterator for the list of keys
	iterador = lista_criar_iterador(keys);

	//for each key, replaces the key in the template for its value
	char *it;
	while ((it = (char *)iterador_proximo_elemento(iterador)) != NULL) {
		
		MultilineString *value = (MultilineString*)tabela_consultar(table, it);	
		char *toReplace = NULL;
		
		for(i = 0; i < value->numberOfLines; i++){
			len = strlen(value->line[i]);
			if(toReplace == NULL){
				toReplace = malloc(len + 2);
				toReplace[0] = 0;
			}else{
			
				if((toReplace = realloc(toReplace, strlen(toReplace) + len + 2)) == NULL){
					ERROR(4, "Can't alloc");
				}			
			}
			
			sprintf(toReplace, "%s%s\n", toReplace, value->line[i]);
		}		
		
		char *temp;
		temp = str_replace(template, it, toReplace);
		if(toReplace != NULL){
			free(toReplace);
			toReplace = NULL;
		}
		free(template);
		template = temp;
	}		
	
	iterador_destruir(&iterador);
	lista_destruir(&keys);
	return template;
}
