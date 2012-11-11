/**
 * @file main.c
 * @brief Ficheiro principal
 * @date 2012-10-28
 * @author 2120916@my.ipleiria.pt
 * @author cudagen@gmail.com
 * @author
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

	Coords3D grid_dim;
	Coords3D block_dim;

	int numOfBlocks = 0, numOfThreads = 0;

	char outputDir[PATH_MAX] = "";
	char filename[PATH_MAX] = "";
	char capitalFilename[PATH_MAX] = "";
	char fullPath[PATH_MAX] = "";
	char fileType[4] = "";

	char mainTemplateName[PATH_MAX] = "";
	char headerTemplateName[PATH_MAX] = "";
	char makefileTemplateName[PATH_MAX] = "";

	char fileVarMainTemplateName[PATH_MAX] = "";
	char fileVarHeaderTemplateName[PATH_MAX] = "";
	char fileVarMakefileTemplateName[PATH_MAX] = "";

	char userName[PATH_MAX] = "your name";
	char *kernelProto = NULL;
	char *fileVars = NULL;

	HASHTABLE_T *systemVarsTable;
	HASHTABLE_T *fileVarsTable;
	LISTA_GENERICA_T *varsIgnoreList;

	char *template;
	unsigned int i = 0;
	char *currentDate = NULL;

	// parse input parameters
	if (cmdline_parser(argc, argv, &args_info) != 0)
		exit(1);

	memset(&grid_dim, 0, sizeof(Coords3D));
	strcpy(grid_dim.csvString, "1"); 
	memset(&block_dim, 0, sizeof(Coords3D));
	strcpy(block_dim.csvString, "1"); 

	currentDate = getDateTime();

	// --about
	if (args_info.about_given) {
		return 0;
	}
	// --proto
	if (args_info.proto_given) {
		kernelProto = malloc(strlen(args_info.proto_arg) + 1);
		strcpy(kernelProto, args_info.proto_arg);
		parseGivenName(kernelProto);
	} else {
		kernelProto = malloc(2);
		strcpy(kernelProto, "");
	}

	// --blocs
	if (args_info.blocks_given) {
		numOfBlocks = fill_grid_dim(&grid_dim, &args_info);
	}
	// --threads
	if (args_info.threads_given) {
		numOfThreads = fill_block_dim(&block_dim, &args_info);
	}
	// --dir
	// get filename from path (the name of the last directory)
	getFilenameFromPath(args_info.dir_arg, filename);

	// the filename in capital letters
	for (i = 0; i < strlen(filename); i++) {
		capitalFilename[i] = toupper(filename[i]);
	}
	capitalFilename[i] = 0;

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
	//creates an hashtable with system generated template variables
	systemVarsTable = tabela_criar(11, (LIBERTAR_FUNC) free);
	fill_system_vars_hashtable(systemVarsTable, currentDate, &grid_dim,
				   &block_dim, filename, capitalFilename,
				   kernelProto, userName);

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
	fill_file_vars_hashtable(fileVarsTable, fileVars);
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
	fill_file_vars_hashtable(fileVarsTable, fileVars);
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
	fill_file_vars_hashtable(fileVarsTable, fileVars);
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

	free(kernelProto);
	free(currentDate);
	cmdline_parser_free(&args_info);
	tabela_destruir(&systemVarsTable);
	lista_destruir(&varsIgnoreList);
	return 0;
}

int fill_grid_dim(Coords3D * grid_dim, struct gengetopt_args_info *args_info)
{
	//validates grid dim x
	if (args_info->blocks_given >= 1) {
		if (args_info->blocks_arg[0] >= 1
		    && args_info->blocks_arg[0] <= 65535) {
			grid_dim->x = args_info->blocks_arg[0];
			strcpy(grid_dim->sx, args_info->blocks_orig[0]);
		} else {
			printf
			    ("Warning: Invalid grid x dimension (it must be set from 1 to 65535)\n\n");
		}
	}
	//validates grid dim y
	if (args_info->blocks_given >= 2) {

		if (args_info->blocks_arg[1] >= 1
		    && args_info->blocks_arg[1] <= 65535) {
			grid_dim->y = args_info->blocks_arg[1];
			strcpy(grid_dim->sy, args_info->blocks_orig[1]);
		} else {
			printf
			    ("Warning: Invalid grid y dimension (it must be set from 1 to 65535)\n\n");
		}
	}
	//validates grid dim z
	if (args_info->blocks_given >= 3) {
		if (args_info->blocks_arg[2] >= 1
		    && args_info->blocks_arg[2] <= 65535) {
			grid_dim->z = args_info->blocks_arg[2];
			strcpy(grid_dim->sz, args_info->blocks_orig[2]);
		} else {

			printf
			    ("Warning: Invalid grid z dimension (it must be set from 1 to 65535)\n\n");
		}
	}
	
	if (strcmp(grid_dim->sx, "")) {
		sprintf(grid_dim->csvString,"%s" , grid_dim->sx);
		if (strcmp(grid_dim->sy, "")) {
			sprintf(grid_dim->csvString,"%s,%s" , grid_dim->csvString, grid_dim->sy);
			if (strcmp(grid_dim->sz, "")) {
				sprintf(grid_dim->csvString,"%s,%s" , grid_dim->csvString, grid_dim->sz);
			}
		}
	}else{
		strcpy(grid_dim->csvString, "1");
	}

	return grid_dim->x * grid_dim->y * grid_dim->z;
}

int fill_block_dim(Coords3D * block_dim, struct gengetopt_args_info *args_info)
{

	//validates blocks dim x
	if (args_info->threads_given >= 1) {
		if (args_info->threads_arg[0] >= 1
		    && args_info->threads_arg[0] <= 65535) {
			block_dim->x = args_info->threads_arg[0];
			strcpy(block_dim->sx, args_info->threads_orig[0]);
		} else {
			printf
			    ("Warning: Invalid block x dimension (it must be set from 1 to 1024)\n\n");
		}
	}
	//validates blocks dim y
	if (args_info->threads_given >= 2) {
		if (args_info->threads_arg[1] >= 1
		    && args_info->threads_arg[1] <= 65535) {
			block_dim->y = args_info->threads_arg[1];
			strcpy(block_dim->sy, args_info->threads_orig[1]);
		} else {
			printf
			    ("Warning: Invalid block y dimension (it must be set from 1 to 1024)\n\n");
		}

	}
	//validates blocks dim z
	if (args_info->threads_given >= 3) {
		if (args_info->threads_arg[2] >= 1
		    && args_info->threads_arg[2] <= 65535) {
			block_dim->z = args_info->threads_arg[2];
			strcpy(block_dim->sz, args_info->threads_orig[2]);
		} else {
			printf
			    ("Warning: Invalid block z dimension (it must be set from 1 to 512)\n\n");
		}

	}

	if (strcmp(block_dim->sx, "")) {
		sprintf(block_dim->csvString,"%s" , block_dim->sx);
		if (strcmp(block_dim->sy,"")) {
			sprintf(block_dim->csvString,"%s,%s" , block_dim->csvString, block_dim->sy);
			if (strcmp(block_dim->sz, "")) {
				sprintf(block_dim->csvString,"%s,%s" , block_dim->csvString, block_dim->sz);
			}
		}
	}else{
		strcpy(block_dim->csvString, "1");
	}

	return block_dim->x * block_dim->y * block_dim->z;
}
