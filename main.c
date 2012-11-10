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
#include <limits.h>
#include <ctype.h>

#include "main.h"
#include "dirutils.h"
#include "cmdline.h"
#include "3rdParty/debug.h"

#define MAIN_FILE "main.cu"
#define TEMPLATE1 "templates/CudaTemplate.cu"
#define TEMPLATE2 "templates/CudaTemplate_P.cu"

#define C_HEADER_TEMPLATE "templates/CHeaderTemplate.h"
#define CU_HEADER_TEMPLATE "templates/CuHeaderTemplate.h"

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
	char templateName[PATH_MAX] = "";
	char headerTemplateName[PATH_MAX] = "";
	char fileType[4] = "";
	
	HASHTABLE_T *templateHashtable;
	HASHTABLE_T *headerHashtable;
	
	char *template;
	char *headerTemplate;
	
	unsigned int i = 0;
	

	int hasOpt = FALSE;
	int cTemplate = FALSE;
	int forceByDefault = FALSE;


	
	char *currentDate = getDateTime();

	// parse input parameters
	if (cmdline_parser(argc, argv, &args_info) != 0)
		exit(1);

	if (args_info.about_given) {
		return 0;
	}
	//TODO
	if (args_info.Force_given) {
		forceByDefault = TRUE;
	}

	//TODO
	if (args_info.regular_code_given) {
		cTemplate = TRUE;
		hasOpt = TRUE;
	}

	//fills the grid dimension
	if(args_info.blocks_given){
		numOfBlocks = fill_grid_dim(&grid_dim, &args_info);
	}
	//fills the blocks dimension
	if(args_info.threads_given){
		numOfThreads = fill_grid_dim(&block_dim, &args_info);
	}
	
	//get filename from path
	getFilenameFromPath(args_info.dir_arg, filename);
	
	
	for(i = 0; i < strlen(filename); i++){
		capitalFilename[i] = toupper(filename[i]);
	}
	capitalFilename[i] = 0;
	
	
	
	// and removes the / character
	if(args_info.dir_arg[strlen(args_info.dir_arg)-1]=='/')
		args_info.dir_arg[strlen(args_info.dir_arg) - 1] = 0;
		printf("%s\n", args_info.dir_arg);
		
	//creates the output directory
	if (!createDirectory(args_info.dir_arg)) {

		sprintf(outputDir, "%s%s", args_info.dir_arg, currentDate);
		createDirectory(outputDir);
	} else {
		sprintf(outputDir, "%s", args_info.dir_arg);
	}
		
	sprintf(outputDir,"%s/", outputDir);
		
	//creates hashtable where the key is a template tag to be replace by the key's value
	templateHashtable = tabela_criar(10, NULL);
	headerHashtable = tabela_criar(10, NULL);



	if (args_info.proto_given) {
		strcpy(templateName, TEMPLATE2);	
		parseGivenName(args_info.proto_arg);
		fill_prototype_template_hashtable(templateHashtable, args_info.proto_arg, filename, currentDate);
				
		strcpy(headerTemplateName, CU_HEADER_TEMPLATE);
		fill_header_template_hashtable(headerHashtable, filename, capitalFilename, currentDate);
		
		strcat(fileType, ".cu");
	}else{
		strcpy(templateName, TEMPLATE1);
		strcpy(headerTemplateName, CU_HEADER_TEMPLATE);
		fill_default_template_hashtable(templateHashtable, &grid_dim, &block_dim);
		strcat(fileType, ".cu");
		
		strcpy(headerTemplateName, CU_HEADER_TEMPLATE);
		fill_header_template_hashtable(headerHashtable, filename, capitalFilename, currentDate);
		
		strcat(fileType, ".cu");
	}
	
	
	//reads the template from file
	printf("%s\n", templateName);
	template = fileToString(templateName);
	template = replace_string_with_template_variables(template, templateHashtable);
	
	
	snprintf(fullPath, PATH_MAX, "%s%s%s", outputDir, filename, fileType);
	stringToFile(fullPath, template);
	
	
	//reads the template from file
	printf("%s\n", headerTemplateName);
	headerTemplate = fileToString(headerTemplateName);
	headerTemplate = replace_string_with_template_variables(headerTemplate, headerHashtable);
	
	snprintf(fullPath, PATH_MAX, "%s%s%s", outputDir, filename, ".h");
	stringToFile(fullPath, headerTemplate);
	



	
	free(currentDate);
	free(template);
	free(headerTemplate);
	//free gengetopt
	cmdline_parser_free(&args_info);
	tabela_destruir(&templateHashtable);
	tabela_destruir(&headerHashtable);
	return 0;
}

//http://stackoverflow.com/questions/1285097/how-to-copy-text-file-to-string-in-c
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

//http://stackoverflow.com/questions/3659694/how-to-replace-substring-in-c
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

void stringToFile(char *filename, char *string){
	FILE *fptr = NULL;
	if ((fptr = fopen(filename, "w")) == NULL){
		ERROR(3, "Can't open file to write");
	}
	fprintf(fptr, "%s", string);
	fclose(fptr);
}

