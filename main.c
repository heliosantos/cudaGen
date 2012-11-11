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
	HASHTABLE_T *ignoreVarsTable;
	
	char *template;
	unsigned int i = 0;
	char *currentDate = getDateTime();

	// parse input parameters
	if (cmdline_parser(argc, argv, &args_info) != 0)
		exit(1);

	// --about
	if (args_info.about_given) {
		return 0;
	}
	
	// --proto
	if (args_info.proto_given) {
		kernelProto = malloc(strlen(args_info.proto_arg) + 1);
		strcpy(kernelProto, args_info.proto_arg);
		parseGivenName(kernelProto);
	}else{
		kernelProto = malloc(2);
		strcpy(kernelProto,"");	
	}

	// --blocs
	if(args_info.blocks_given){
		numOfBlocks = fill_grid_dim(&grid_dim, &args_info);
	}
	
	// --threads
	if(args_info.threads_given){
		numOfThreads = fill_block_dim(&block_dim, &args_info);
	}
	
	// --dir
	// get filename from path (the name of the last directory)
	getFilenameFromPath(args_info.dir_arg, filename);
	
	// the filename in capital letters
	for(i = 0; i < strlen(filename); i++){
		capitalFilename[i] = toupper(filename[i]);
	}
	capitalFilename[i] = 0;	
	
	// removes the / character
	if(args_info.dir_arg[strlen(args_info.dir_arg)-1]=='/'){
		args_info.dir_arg[strlen(args_info.dir_arg) - 1] = 0;
	}
		
	//creates the output directory
	if (!createDirectory(args_info.dir_arg)) {
		
		if(args_info.Force_given){
			// removes the existing directoy	
			remove_directory(args_info.dir_arg);
			sprintf(outputDir, "%s", args_info.dir_arg);
		}else{
			// adds a date to the directory name
			sprintf(outputDir, "%s%s", args_info.dir_arg, currentDate);
		}		
		
		createDirectory(outputDir);
	} else {
		sprintf(outputDir, "%s", args_info.dir_arg);
	}
		
	sprintf(outputDir,"%s/", outputDir);
		
		
	//creates an hashtable with system generated template variables
	systemVarsTable = tabela_criar(11, NULL);
	fill_system_vars_hashtable(systemVarsTable, currentDate, &grid_dim, &block_dim, filename, capitalFilename, kernelProto, userName);	
	
			
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
	}else if(args_info.regular_code_given){
		strcpy(mainTemplateName, C_MAIN_TEMPLATE);
	
		strcpy(headerTemplateName, C_HEADER_TEMPLATE);
		
		strcpy(makefileTemplateName, C_MAKEFILE_TEMPLATE);
		
		strcpy(fileVarMainTemplateName, C_MAIN_TEMPLATE_VARS);
		
		strcpy(fileVarHeaderTemplateName, C_HEADER_TEMPLATE_VARS);
		
		strcpy(fileVarMakefileTemplateName, C_MAKEFILE_TEMPLATE_VARS);
		
		strcat(fileType, ".c");
	
	// cuda default
	}else{
		strcpy(mainTemplateName, CU_MAIN_TEMPLATE);
	
		strcpy(headerTemplateName, CU_HEADER_TEMPLATE);
		
		strcpy(makefileTemplateName, CU_MAKEFILE_TEMPLATE);
		
		strcpy(fileVarMainTemplateName, CU_MAIN_TEMPLATE_VARS);
		
		strcpy(fileVarHeaderTemplateName, CU_HEADER_TEMPLATE_VARS);
		
		strcpy(fileVarMakefileTemplateName, CU_MAKEFILE_TEMPLATE_VARS);
		
		strcat(fileType, ".cu");
	}
	
	
	/* Create Main file */
	
	// get the template
	template = fileToString(mainTemplateName);
	// get the file vars for template
	fileVars = fileToString(fileVarMainTemplateName);		
	// creates an hastable containing the file vars 
	fileVarsTable = tabela_criar(10, (LIBERTAR_FUNC)freeMultiLineString);
	fill_file_vars_hashtable(fileVarsTable, fileVars);
	free(fileVars);		
	// update the template with vars from file
	template = replace_string_with_template_multiline_variables(template, fileVarsTable);
	tabela_destruir(&fileVarsTable);
	// update the Main template with system variablese
	template = replace_string_with_template_variables(template, systemVarsTable);	
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
	fileVarsTable = tabela_criar(10, (LIBERTAR_FUNC)freeMultiLineString);
	fill_file_vars_hashtable(fileVarsTable, fileVars);
	free(fileVars);		
	// update the template with vars from file
	template = replace_string_with_template_multiline_variables(template, fileVarsTable);
	tabela_destruir(&fileVarsTable);
	// update the template with system variablese
	template = replace_string_with_template_variables(template, systemVarsTable);	
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
	fileVarsTable = tabela_criar(10, (LIBERTAR_FUNC)freeMultiLineString);
	fill_file_vars_hashtable(fileVarsTable, fileVars);
	free(fileVars);		
	// update the template with vars from file
	template = replace_string_with_template_multiline_variables(template, fileVarsTable);
	tabela_destruir(&fileVarsTable);
	// update the template with system variablese
	template = replace_string_with_template_variables(template, systemVarsTable);	
	//writes to destination file
	snprintf(fullPath, PATH_MAX, "%s%s", outputDir, MAKEFILE_NAME);
	stringToFile(fullPath, template);
	free(template);
		
		
		
	free(kernelProto);	
	free(currentDate);
	cmdline_parser_free(&args_info);	
	tabela_destruir(&systemVarsTable);
	
	return 0;
}

/**
*
* Stores a file in a string
*
*http://stackoverflow.com/questions/1285097/how-to-copy-text-file-to-string-in-c
*
*/
char *fileToString(char *fileName)
{
	long f_size;
	char *code;
	size_t code_s;
	FILE *fp = NULL;
	
	if((fp = fopen(fileName, "r")) == NULL){
		ERROR(3,"Can't open file to read");
	}
	
	fseek(fp, 0, SEEK_END);
	f_size = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	code_s = sizeof(char) * f_size + 1;
	code = malloc(code_s);
	code[code_s - 1] = 0;
	fread(code, 1, f_size, fp);
	fclose(fp);
	return code;
}


/**
*
* Replaces all occurences of string A by string B on string C
*
*http://stackoverflow.com/questions/3659694/how-to-replace-substring-in-c
*
*/
char *str_replace(const char *s, const char *old, const char *new)
{
	size_t slen = strlen(s) + 1;
	char *cout = malloc(slen), *p = cout;
	if (!p)
		return 0;
	while (*s)
		if (!strncmp(s, old, strlen(old))) {
			p = (char *)(p - cout);
			cout = realloc(cout, slen += strlen(new) - strlen(old));
			p = cout + (long)p;
			p += strlen(strcpy(p, new));
			s += strlen(old);
		} else
			*p++ = *s++;
	*p = 0;
	return cout;
}

int fill_grid_dim(Coords3D * grid_dim, struct gengetopt_args_info *args_info)
{
	//validates grid dim x

	int isValidFlag = 0;
	if (args_info->blocks_given >= 1) {
		isValidFlag = args_info->blocks_arg[0] >= 1
		    && args_info->blocks_arg[0] <= 65535;
	}
	if (isValidFlag) {
		grid_dim->x = args_info->blocks_arg[0];
		strcpy(grid_dim->sx, args_info->blocks_orig[0]);
	} else {
		printf
		    ("Warning: Invalid grid x dimension (it must be larger than 0 and lower than 65536)\n\n");
	}

	//validates grid dim y
	isValidFlag = 0;
	if (args_info->blocks_given >= 2) {
		isValidFlag = args_info->blocks_arg[1] >= 1
		    && args_info->blocks_arg[1] <= 65535;

	}
	if (isValidFlag) {
		grid_dim->y = args_info->blocks_arg[1];
		strcpy(grid_dim->sy, args_info->blocks_orig[1]);
	} else {
		printf
		    ("Warning: Invalid grid y dimension (it must be larger than 0 and lower than 65536)\n\n");
	}

	//validates grid dim z
	isValidFlag = 0;
	if (args_info->blocks_given >= 3) {
		isValidFlag = args_info->blocks_arg[2] >= 1
		    && args_info->blocks_arg[2] <= 65535;

	}
	if (isValidFlag) {
		grid_dim->z = args_info->blocks_arg[2];
		strcpy(grid_dim->sz, args_info->blocks_orig[2]);
	} else {
		printf
		    ("Warning: Invalid grid z dimension (it must be larger than 0 and lower than 65536)\n\n");
	}
	return grid_dim->x * grid_dim->y * grid_dim->z;
}

int fill_block_dim(Coords3D * block_dim, struct gengetopt_args_info *args_info)
{

	//validates blocks dim x
	int isValidFlag = 0;
	if (args_info->threads_given >= 1) {
		isValidFlag = args_info->threads_arg[0] >= 1
		    && args_info->threads_arg[0] <= 65535;
	}

	if (isValidFlag) {
		block_dim->x = args_info->threads_arg[0];
		strcpy(block_dim->sx, args_info->threads_orig[0]);
	} else {
		printf
		    ("Warning: Invalid block x dimension (it must be larger than 0 and equal or lower than 1024)\n\n");
	}

	//validates blocks dim y
	isValidFlag = 0;
	if (args_info->threads_given >= 2) {
		isValidFlag = args_info->threads_arg[1] >= 1
		    && args_info->threads_arg[1] <= 65535;

	}
	if (isValidFlag) {
		block_dim->y = args_info->threads_arg[1];
		strcpy(block_dim->sy, args_info->threads_orig[1]);
	} else {
		printf
		    ("Warning: Invalid block y dimension (it must be larger than 0 and equal or lower than 1024)\n\n");
	}

	//validates blocks dim z
	isValidFlag = 0;
	if (args_info->threads_given >= 3) {
		isValidFlag = args_info->threads_arg[2] >= 1
		    && args_info->threads_arg[2] <= 65535;

	}
	if (isValidFlag) {
		block_dim->z = args_info->threads_arg[2];
		strcpy(block_dim->sz, args_info->threads_orig[2]);
	} else {
		printf
		    ("Warning: Invalid block z dimension (it must be larger than 0 and equal or lower than 512)\n\n");
	}
	return block_dim->x * block_dim->y * block_dim->z;
}

/**
*
* Stores a string in a file
*
*/
void stringToFile(char *filename, char *string){
	FILE *fptr = NULL;
	if ((fptr = fopen(filename, "w")) == NULL){
		ERROR(3, "Can't open file to write");
	}
	fprintf(fptr, "%s", string);
	fclose(fptr);
}

