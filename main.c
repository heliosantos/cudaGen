/**
 * @file main.c
 *
 * @brief The core file of cudaGen
 * @date 2012-10-28
 * @author 2120916@my.ipleiria.pt
 * @author 2120912@my.ipleiria.pt
 * @author 2120024@my.ipleiria.pt
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <ctype.h>

#include "main.h"
#include "dirutils.h"
#include "cmdline.h"
#include "3rdParty/debug.h"
#include "3rdParty/listas.h"
#include "utils.h"

#define MAIN_FILE "main.cu"

#define C_HEADER_TEMPLATE "templates/CHeaderTemplate.h"
#define C_MAIN_TEMPLATE "templates/CMainTemplate.c"
#define C_MAKEFILE_TEMPLATE "templates/CMakefileTemplate.make"

#define CU_HEADER_TEMPLATE "templates/CuHeaderTemplate.h"
#define CU_MAIN_TEMPLATE "templates/CuMainTemplate.cu"
#define CU_MAKEFILE_TEMPLATE "templates/CuMakefileTemplate.make"
#define CU_PROTO_TEMPLATE "templates/CuProtoTemplate.cu"

#define C_HEADER_TEMPLATE_VARS "templates/CHeaderTemplateVars.h"
#define C_MAIN_TEMPLATE_VARS "templates/CMainTemplateVars.c"
#define C_MAKEFILE_TEMPLATE_VARS "templates/CMakefileTemplateVars.make"

#define CU_HEADER_TEMPLATE_VARS "templates/CuHeaderTemplateVars.h"
#define CU_MAIN_TEMPLATE_VARS "templates/CuMainTemplateVars.cu"
#define CU_MAKEFILE_TEMPLATE_VARS "templates/CuMakefileTemplateVars.make"
#define CU_PROTO_TEMPLATE_VARS "templates/CuProtoTemplateVars.cu"

#define HANDLE_ERROR_H "templates/HandleError.h"
#define HANDLE_ERROR_H_NAME "HandleError.h"

#define MAKEFILE_NAME "makefile"

#define MAX_KNAME 256

int main(int argc, char **argv)
{
	struct gengetopt_args_info args_info;

	char outputDir[PATH_MAX] = ""; // the output directory where files are to be saved
	char filename[PATH_MAX] = ""; // the name of the files (without extension) 
	char capitalFilename[PATH_MAX] = ""; // the name of the files (without extension) in capital letters
	char fullPath[PATH_MAX] = ""; // the full path of the file
	char fileType[4] = ""; // the extension of the file

	char mainTemplateName[PATH_MAX] = ""; // the path to the main template file (.c or .cu)
	char headerTemplateName[PATH_MAX] = ""; // the path to the header template file (.h)
	char makefileTemplateName[PATH_MAX] = ""; // the path to the makefile template file

	char fileVarMainTemplateName[PATH_MAX] = "";// the path to the main template variables file 
	char fileVarHeaderTemplateName[PATH_MAX] = ""; // the path to the header template variables file
	char fileVarMakefileTemplateName[PATH_MAX] = ""; // the path to the makefile template variables file
	

	HASHTABLE_T *systemVarsTable; // an hashtable containing program genereted variables
	HASHTABLE_T *fileVarsTable;
	LISTA_GENERICA_T *varsIgnoreList; // an hashtable containing file fetched variables	
	
	char userName[PATH_MAX] = "your name"; // the name of the user
	char *fileVars = NULL; // a string with the vars fetched from a variables file


	char *template; // a string with the content of a template file
	unsigned int i = 0; // a utility index
	char *currentDate = NULL; // a string representing the current date
	
	// parse input parameters
	if (cmdline_parser(argc, argv, &args_info) != 0)
		exit(1);
	
	currentDate = getDateTime();
	
	//creates an hashtable with system generated template variables
	systemVarsTable = tabela_criar(11, (LIBERTAR_FUNC) free);
	/*fill_system_vars_hashtable(systemVarsTable, currentDate, &grid_dim,
				   &block_dim, filename, capitalFilename,
				   kernelProto, userName);*/
		
	tabela_inserir(systemVarsTable, "$!C_DATE!$", string_clone(currentDate));	
	tabela_inserir(systemVarsTable, "$!USER_NAME!$", string_clone(userName));

	// --about
	if (args_info.about_given) {
		return 0;
	}
	// --proto
	if (args_info.proto_given) {
		tabela_inserir(systemVarsTable, "$!KERNEL_PROTO!$", string_clone(args_info.proto_arg));	
	} else {
		tabela_inserir(systemVarsTable, "$!KERNEL_PROTO!$", string_clone(""));	
	}
	
	// --kernel
	if (args_info.kernel_given) {
		tabela_inserir(systemVarsTable, "$!KERNEL_NAME!$", string_clone(args_info.kernel_arg));	
	} else {
		tabela_inserir(systemVarsTable, "$!KERNEL_NAME!$", string_clone("Kernel"));	
	}
	
	// --blocks
	store_grid_geometry(systemVarsTable, &args_info);
	// --threads
	store_blocks_geometry(systemVarsTable, &args_info);
	
	// --dir
	// get filename from path (the name of the last directory)
	getFilenameFromPath(args_info.dir_arg, filename);

	// the filename in capital letters
	for (i = 0; i < strlen(filename); i++) {
		capitalFilename[i] = toupper(filename[i]);
	}
	capitalFilename[i] = 0;
	
	tabela_inserir(systemVarsTable, "$!FILENAME!$", string_clone(filename));
	tabela_inserir(systemVarsTable, "$!CAPITAL_FILENAME!$", string_clone(capitalFilename));
	

	// removes the / character
	if (args_info.dir_arg[strlen(args_info.dir_arg) - 1] == '/') {
		args_info.dir_arg[strlen(args_info.dir_arg) - 1] = 0;
	}
	//creates the output directory
	if (!createDirectory(args_info.dir_arg)) {

		if (args_info.Force_given) {
			// removes the existing directoy        
			remove_directory(args_info.dir_arg);
			sprintf(outputDir, "%s", args_info.dir_arg);
		} else {
			// adds a date to the directory name
			sprintf(outputDir, "%s%s", args_info.dir_arg,
				currentDate);
		}

		createDirectory(outputDir);
	} else {
		sprintf(outputDir, "%s", args_info.dir_arg);
	}

	sprintf(outputDir, "%s/", outputDir);

	varsIgnoreList = lista_criar((LIBERTAR_FUNC) free);

	if (!args_info.measure_given) {
		char *var = malloc(strlen("MEASURE") + 1);
		strcpy(var, "MEASURE");
		lista_inserir(varsIgnoreList, var);

	}
	

	/* defines which templates to use */

	// cuda with prototype
	if (args_info.proto_given) {
		strcpy(mainTemplateName, CU_PROTO_TEMPLATE);

		strcpy(headerTemplateName, CU_HEADER_TEMPLATE);

		strcpy(makefileTemplateName, CU_MAKEFILE_TEMPLATE);

		strcpy(fileVarMainTemplateName, CU_PROTO_TEMPLATE_VARS);

		strcpy(fileVarHeaderTemplateName, CU_HEADER_TEMPLATE_VARS);

		strcpy(fileVarMakefileTemplateName, CU_MAKEFILE_TEMPLATE_VARS);

		strcat(fileType, ".cu");

		// regular c template   
	} else if (args_info.regular_code_given) {
		strcpy(mainTemplateName, C_MAIN_TEMPLATE);

		strcpy(headerTemplateName, C_HEADER_TEMPLATE);

		strcpy(makefileTemplateName, C_MAKEFILE_TEMPLATE);

		strcpy(fileVarMainTemplateName, C_MAIN_TEMPLATE_VARS);

		strcpy(fileVarHeaderTemplateName, C_HEADER_TEMPLATE_VARS);

		strcpy(fileVarMakefileTemplateName, C_MAKEFILE_TEMPLATE_VARS);

		strcat(fileType, ".c");

		// cuda default
	} else {
		strcpy(mainTemplateName, CU_MAIN_TEMPLATE);

		strcpy(headerTemplateName, CU_HEADER_TEMPLATE);

		strcpy(makefileTemplateName, CU_MAKEFILE_TEMPLATE);

		strcpy(fileVarMainTemplateName, CU_MAIN_TEMPLATE_VARS);

		strcpy(fileVarHeaderTemplateName, CU_HEADER_TEMPLATE_VARS);

		strcpy(fileVarMakefileTemplateName, CU_MAKEFILE_TEMPLATE_VARS);

		strcat(fileType, ".cu");

	}

	// create HANDLE_ERROR_H file
	if (strcmp(fileType, ".cu") == 0) {
		// reads from source file
		template = fileToString(HANDLE_ERROR_H);
		// writes to destination file
		snprintf(fullPath, PATH_MAX, "%s%s", outputDir,
			 HANDLE_ERROR_H_NAME);
		stringToFile(fullPath, template);
		free(template);
	}

	/* Create Main file */

	// get the template
	template = fileToString(mainTemplateName);
	// get the file vars for vars template
	fileVars = fileToString(fileVarMainTemplateName);
	// creates an hastable containing the file vars 
	fileVarsTable = tabela_criar(10, (LIBERTAR_FUNC) free);
	store_file_vars(fileVarsTable, fileVars);
	free(fileVars);
	// clear the vars to ignore
	free_matched_vars_from_hashtable(fileVarsTable, varsIgnoreList);

	// update the template with vars from file
	template =
	    replace_string_with_hashtable_variables(template, fileVarsTable);
	tabela_destruir(&fileVarsTable);
	// update the Main template with system variablese
	template =
	    replace_string_with_hashtable_variables(template, systemVarsTable);
	//writes to destination file
	snprintf(fullPath, PATH_MAX, "%s%s%s", outputDir, filename, fileType);
	stringToFile(fullPath, template);
	free(template);

	/* Create Header file */

	// get the header template
	template = fileToString(headerTemplateName);
	// get the file vars for template
	fileVars = fileToString(fileVarHeaderTemplateName);
	// creates an hastable containing the file vars 
	fileVarsTable = tabela_criar(10, (LIBERTAR_FUNC) free);
	store_file_vars(fileVarsTable, fileVars);
	free(fileVars);
	// clear the vars to ignore
	free_matched_vars_from_hashtable(fileVarsTable, varsIgnoreList);

	// update the template with vars from file
	template =
	    replace_string_with_hashtable_variables(template, fileVarsTable);
	tabela_destruir(&fileVarsTable);
	// update the template with system variablese
	template =
	    replace_string_with_hashtable_variables(template, systemVarsTable);
	//writes to destination file
	snprintf(fullPath, PATH_MAX, "%s%s%s", outputDir, filename, ".h");
	stringToFile(fullPath, template);
	free(template);

	/* Create Make file */

	// get the header template
	template = fileToString(makefileTemplateName);
	// get the file vars for template
	fileVars = fileToString(fileVarMakefileTemplateName);
	// creates an hastable containing the file vars 
	fileVarsTable = tabela_criar(10, (LIBERTAR_FUNC) free);
	store_file_vars(fileVarsTable, fileVars);
	free(fileVars);
	// clear the vars to ignore
	free_matched_vars_from_hashtable(fileVarsTable, varsIgnoreList);

	// update the template with vars from file
	template =
	    replace_string_with_hashtable_variables(template, fileVarsTable);
	tabela_destruir(&fileVarsTable);
	// update the template with system variablese
	template =
	    replace_string_with_hashtable_variables(template, systemVarsTable);
	//writes to destination file
	snprintf(fullPath, PATH_MAX, "%s%s", outputDir, MAKEFILE_NAME);
	stringToFile(fullPath, template);
	free(template);

	free(currentDate);
	cmdline_parser_free(&args_info);
	tabela_destruir(&systemVarsTable);
	lista_destruir(&varsIgnoreList);
	return 0;
}

void store_grid_geometry(HASHTABLE_T * table, struct gengetopt_args_info *args_info)
{
	char *x;
	char *y;
	char *z;
	char *csvString;
	
	//validates grid dim x
	if (args_info->blocks_given >= 1) {
		if (args_info->blocks_arg[0] < 1 || args_info->blocks_arg[0] > 65535) {
			printf("Warning: Invalid grid x dimension (it must be set from 1 to 65535)\n\n");			
		}
		x = string_clone(args_info->blocks_orig[0]);
	}else{
		x = string_clone("1");
	}
	
	//validates grid dim y
	if (args_info->blocks_given >= 2) {
		if (args_info->blocks_arg[1] < 1 || args_info->blocks_arg[1] > 65535) {
			printf("Warning: Invalid grid y dimension (it must be set from 1 to 65535)\n\n");			
		}
		y = string_clone(args_info->blocks_orig[1]);
	}else{
		y = string_clone("1");
	}
	
	//validates grid dim z
	if (args_info->blocks_given >= 3) {
		if (args_info->blocks_arg[2] < 1 || args_info->blocks_arg[2] > 65535) {
			printf("Warning: Invalid grid z dimension (it must be set from 1 to 65535)\n\n");			
		}
		z = string_clone(args_info->blocks_orig[2]);
	}else{
		z = string_clone("1");
	}
	
	csvString = malloc(strlen(x) + strlen(y) + strlen(z) + 3);
	csvString[0] = 0;
	
	strcpy(csvString, x);
	if (strcmp(y, "1")) {
		sprintf(csvString,"%s,%s" ,csvString, y);
		if (strcmp(z, "1")) {
			sprintf(csvString,"%s,%s" , csvString, z);
		}
	}
	
	tabela_inserir(table, "$!BX!$", x);
	tabela_inserir(table, "$!BY!$", y);
	tabela_inserir(table, "$!BZ!$", z);
	tabela_inserir(table, "$!GRID_DIM!$", csvString);
}

void store_blocks_geometry(HASHTABLE_T * table, struct gengetopt_args_info *args_info)
{
	char *x;
	char *y;
	char *z;
	char *csvString;
	
	//validates grid dim x
	if (args_info->threads_given >= 1) {
		if (args_info->threads_arg[0] < 1 || args_info->threads_arg[0] > 65535) {
			printf("Warning: Invalid block x dimension (it must be set from 1 to 1024)\n\n");		
		}
		x = string_clone(args_info->threads_orig[0]);
	}else{
		x = string_clone("1");
	}
	
	//validates grid dim y
	if (args_info->threads_given >= 2) {
		if (args_info->threads_arg[1] < 1 || args_info->threads_arg[1] > 65535) {
			printf("Warning: Invalid block y dimension (it must be set from 1 to 1024)\n\n");			
		}
		y = string_clone(args_info->threads_orig[1]);
	}else{
		y = string_clone("1");
	}
	
	//validates grid dim z
	if (args_info->threads_given >= 3) {
		if (args_info->threads_arg[2] < 1 || args_info->threads_arg[2] > 65535) {
			printf("Warning: Invalid block z dimension (it must be set from 1 to 512)\n\n");		
		}
		z = string_clone(args_info->threads_orig[2]);
	}else{
		z = string_clone("1");
	}
	
	csvString = malloc(strlen(x) + strlen(y) + strlen(z) + 3);
	csvString[0] = 0;
	
	strcpy(csvString, x);
	if (strcmp(y, "1")) {
		sprintf(csvString,"%s,%s" ,csvString, y);
		if (strcmp(z, "1")) {
			sprintf(csvString,"%s,%s" , csvString, z);
		}
	}
	tabela_inserir(table, "$!TX!$", x);
	tabela_inserir(table, "$!TY!$", y);
	tabela_inserir(table, "$!TZ!$", z);
	tabela_inserir(table, "$!BLOCK_DIM!$", csvString);
}
