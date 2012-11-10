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

#include "templateGen.h"
#include "main.h"



void fill_cu_main_template_hashtable(HASHTABLE_T * tabela, Coords3D * grid_dim,
				Coords3D * block_dim)
{
//insert key-value into hashtable
	tabela_inserir
	    (tabela,
	     "$DECLARE_TIMER$",
	     "cudaEvent_t start, stop;\n" "\tfloat elapsedTime;");
	tabela_inserir(tabela, "$CREATE_TIMER$",
		       "/* create the timers */\n"
		       "\tHANDLE_ERROR(cudaEventCreate(&start));\n"
		       "\tHANDLE_ERROR(cudaEventCreate(&stop));\n"
		       "\t/* start the timer */\n"
		       "\tHANDLE_ERROR(cudaEventRecord(start, 0));");
	tabela_inserir(tabela, "$TERMINATE_TIMER$",
		       "HANDLE_ERROR(cudaEventRecord(stop, 0));\n"
		       "\tcudaEventSynchronize(stop);\n"
		       "\tHANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));\n"
		       "\tprintf(\"execution took %3.6f miliseconds\", elapsedTime);");
	tabela_inserir(tabela, "$FREE_TIMER$",
		       "HANDLE_ERROR(cudaEventDestroy(start));\n"
		       "\tHANDLE_ERROR(cudaEventDestroy(stop));");

	tabela_inserir(tabela, "$BX$", grid_dim->sx);

	tabela_inserir(tabela, "$BY$", grid_dim->sy);

	tabela_inserir(tabela, "$BZ$", grid_dim->sz);

	tabela_inserir(tabela, "$TX$", block_dim->sx);

	tabela_inserir(tabela, "$TY$", block_dim->sy);

	tabela_inserir(tabela, "$TZ$", block_dim->sz);

	//end insert key-value into hashtable
}

void fill_cu_proto_template_hashtable(HASHTABLE_T * tabela, char *kernelName, char *filename, char *currentDate)
{
	tabela_inserir(tabela, "$FILENAME$", filename);
	tabela_inserir(tabela, "$KERNEL_NAME$", kernelName);
	tabela_inserir(tabela, "$C_DATE$", currentDate);
}


void fill_header_template_hashtable(HASHTABLE_T * tabela, char *filename, char *capitalFilename, char *currentDate)
{
	tabela_inserir(tabela, "$FILENAME$", filename);
	tabela_inserir(tabela, "$CAPITAL_FILENAME$", capitalFilename);
	tabela_inserir(tabela, "$C_DATE$", currentDate);
}

void fill_c_main_template_hashtable(HASHTABLE_T * tabela, char *filename, char *currentDate)
{
	tabela_inserir(tabela, "$FILENAME$", filename);
	tabela_inserir(tabela, "$C_DATE$", currentDate);
}

char *replace_string_with_template_variables(char *template, HASHTABLE_T * tabela){

	LISTA_GENERICA_T *keys;
	ITERADOR_T *iterador;

	//a list containing all keys of hashtable
	keys = tabela_criar_lista_chaves(tabela);

	//iterator for the list of keys
	iterador = lista_criar_iterador(keys);

	//for each key, replaces the key in the template for its value
	char *it;
	while ((it = (char *)iterador_proximo_elemento(iterador)) != NULL) {
		char *temp;
		temp = str_replace(template, it, (char *)
				   tabela_consultar(tabela, it));
		free(template);
		template = temp;
	}	
	
	iterador_destruir(&iterador);
	lista_destruir(&keys);
	return template;
}

