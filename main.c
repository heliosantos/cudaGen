/**
 * @file main.c
 * @brief Ficheiro principal
 * @date 2012-10-28
 * @author 2120916@my.ipleiria.pt
 * @author
 * @author
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "main.h"
#include "cmdline.h"
#include "hashtables.h"

#define MAIN_FILE "main.cu"
#define TEMPLATE1 "templates/CudaTemplate.cu"

int main(int argc, char **argv)
{
	struct gengetopt_args_info args_info;
	//char *str;
	Coords3D blocks;
	Coords3D threads;

	int numOfBlocks = 0, numOfThreads = 0;
	char *outputDir = ".";
	char *kernelName = "DefaultKernel";

	HASHTABLE_T *tabela;
	LISTA_GENERICA_T *keys;
	ITERADOR_T *iterador;

	char *template;
	char dimension[6] = "\0";
	int isValidFlag = 0;
    /**
     * captura e processa os argumento de  entrada
     */

	if (cmdline_parser(argc, argv, &args_info) != 0)
		exit(1);
	
	//validates grid dim x
	isValidFlag = 0;
	if (args_info.blocks_given >= 1) {
		isValidFlag = args_info.blocks_arg[0] >= 1 && args_info.blocks_arg[0] <= 65535;
		
	}
	
	if(isValidFlag){
		blocks.x = args_info.blocks_arg[0];
		strcpy(blocks.sx, args_info.blocks_orig[0]);
	}else{
		printf("Warning: Invalid grid x dimension (it must me larger than 0 and lower then 65536)\n\n");
	}
	
	//validates grid dim y
	isValidFlag = 0;
	if (args_info.blocks_given >= 2) {
		isValidFlag = args_info.blocks_arg[1] >= 1 && args_info.blocks_arg[1] <= 65535;
		
	}
	if(isValidFlag){
		blocks.y = args_info.blocks_arg[1];
		strcpy(blocks.sy, args_info.blocks_orig[1]);
	}else{
		printf("Warning: Invalid grid y dimension (it must me larger than 0 and lower then 65536)\n\n");
	}
	
	
	//validates grid dim z
	isValidFlag = 0;
	if (args_info.blocks_given >= 3) {
		isValidFlag = args_info.blocks_arg[2] >= 1 && args_info.blocks_arg[2] <= 65535;
		
	}
	if(isValidFlag){
		blocks.z = args_info.blocks_arg[2];
		strcpy(blocks.sz, args_info.blocks_orig[2]);
	}else{
		printf("Warning: Invalid grid z dimension (it must me larger than 0 and lower then 65536)\n\n");
	}
	numOfBlocks = blocks.x * blocks.y * blocks.z;

	//validates blocks dim x
	isValidFlag = 0;
	if (args_info.threads_given >= 1) {
		isValidFlag = args_info.threads_arg[0] >= 1 && args_info.threads_arg[0] <= 65535;
		
	}
	
	if(isValidFlag){
		threads.x = args_info.threads_arg[0];
		strcpy(threads.sx, args_info.threads_orig[0]);
	}else{
		printf("Warning: Invalid block x dimension (it must me larger than 0 and lower then 65536)\n\n");
	}
	
	//validates blocks dim y
	isValidFlag = 0;
	if (args_info.threads_given >= 2) {
		isValidFlag = args_info.threads_arg[1] >= 1 && args_info.threads_arg[1] <= 65535;
		
	}
	if(isValidFlag){
		threads.y = args_info.threads_arg[1];
		strcpy(threads.sy, args_info.threads_orig[1]);
	}else{
		printf("Warning: Invalid block y dimension (it must me larger than 0 and lower then 65536)\n\n");
	}
	
	
	//validates grid dim z
	isValidFlag = 0;
	if (args_info.threads_given >= 3) {
		isValidFlag = args_info.threads_arg[2] >= 1 && args_info.threads_arg[2] <= 65535;
		
	}
	if(isValidFlag){
		threads.z = args_info.threads_arg[2];
		strcpy(threads.sz, args_info.threads_orig[2]);
	}else{
		printf("Warning: Invalid block z dimension (it must me larger than 0 and lower then 65536)\n\n");
	}
	numOfThreads = threads.x * threads.y * threads.z;
	
	//creates hashtable where the key is a template tag to be replace by the key's value
	tabela = tabela_criar(10, (LIBERTAR_FUNC)
			      freeElement);
	//insert key-value into hashtable
	tabela_inserir
	    (tabela,
	     "$DECLARE_TIMER$",
	     "cudaEvent_t start, stop;\n" "\tfloat elapsedTime;");
	tabela_inserir
	    (tabela,
	     "$CREATE_TIMER$",
	     "/* create the timers */\n"
	     "\tHANDLE_ERROR(cudaEventCreate(&start));\n"
	     "\tHANDLE_ERROR(cudaEventCreate(&stop));\n"
	     "\t/* start the timer */\n"
	     "\tHANDLE_ERROR(cudaEventRecord(start, 0));");
	tabela_inserir
	    (tabela,
	     "$TERMINATE_TIMER$",
	     "HANDLE_ERROR(cudaEventRecord(stop, 0));\n"
	     "\tcudaEventSynchronize(stop);\n"
	     "\tHANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));\n"
	     "\tprintf(\"execution took %3.6f miliseconds\", elapsedTime);");
	tabela_inserir
	    (tabela,
	     "$FREE_TIMER$",
	     "HANDLE_ERROR(cudaEventDestroy(start));\n"
	     "\tHANDLE_ERROR(cudaEventDestroy(stop));");
	
	     
	tabela_inserir(tabela, "$BX$", blocks.sx);
	
	tabela_inserir(tabela, "$BY$", blocks.sy);
	
	tabela_inserir(tabela, "$BZ$", blocks.sz);



	tabela_inserir(tabela, "$TX$", threads.sx);
	
	tabela_inserir(tabela, "$TY$", threads.sy);
	
	tabela_inserir(tabela, "$TZ$", threads.sz);
	
	//end insert key-value into hashtable
	//reads the template
	template = fileToString(TEMPLATE1);
	//a list containing all keys of hashtable
	keys = tabela_criar_lista_chaves(tabela);
	//iterater for the list of keys
	iterador = lista_criar_iterador(keys);
	//for each key, replaces the key in the template for its value
	
	char *it;
	while ((it = (char *)
		iterador_proximo_elemento(iterador))
	       != NULL) {
		char *temp;
		temp = str_replace(template, it, (char *)
				   tabela_consultar(tabela, it));
		free(template);
		template = temp;
	}

	printf("%s", template);
	iterador_destruir(&iterador);
	lista_destruir(&keys);
	tabela_destruir(&tabela);
	return 0;
}

void freeElement(char
		 *element) {

}

//http://stackoverflow.com/questions/1285097/how-to-copy-text-file-to-string-in-c
char
*fileToString(char
	      *fileName) {
	long f_size;
	char *code;
	size_t code_s, result;
	FILE *fp = fopen(fileName,
			 "r");
	fseek(fp, 0, SEEK_END);
	f_size = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	code_s = sizeof(char)
	    * f_size + 1;
	code = malloc(code_s);
	code[code_s - 1] = 0;
	result = fread(code, 1, f_size, fp);
	fclose(fp);
	return code;
}

//http://stackoverflow.com/questions/3659694/how-to-replace-substring-in-c
char
*str_replace(const char
	     *s, const char
	     *old, const char
	     *new) {
	size_t slen = strlen(s) + 1;
	char *cout = malloc(slen), *p = cout;
	if (!p)
		return 0;
	while (*s)
		if (!strncmp(s, old, strlen(old))) {
			p = (char *)(p - cout);
			cout = realloc(cout, slen += strlen(new)
				       - strlen(old));
			p = cout + (long)p;
			p += strlen(strcpy(p, new));
			s += strlen(old);
		} else
			*p++ = *s++;
	*p = 0;
	return cout;
}
