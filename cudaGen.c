/**
* @file cudaGen.c
* @brief cudaGen project main file
* @date 07-11-2012
* @author 2120024@my.ipleiria.pt
* @author 2120912@my.ipleiria.pt
* @author 2120916@my.ipleiria.pt
*/
#include <stdio.h>

#include "cudaGenOpt.h"

#include "cudaGen.h"
#include "chambel.h"
#include "debug.h"

int main(int argc, char *argv[])
{
	/* Variable declaration */
	struct gengetopt_args_info args_info;

	int hasOpt = FALSE;
	int cudaTemplate = FALSE;
	int cTemplate = FALSE;
	int cudaTemplateOnlyKernelDefinition = FALSE;

	int forceByDefault = FALSE;

	char *dirname;
	char *path = NULL;
	char *kernelProto;

	/* Disable warnings */
	(void)argc;
	(void)argv;

	/* Processa os parametros da linha de comando */
	if (cmdline_parser(argc, argv, &args_info) != 0) {
		return 0;
	}

	if (args_info.about_given) {

		return 0;
	} else {

		if (args_info.Force_given) {
			forceByDefault = TRUE;
		}

		if (args_info.regular_code_given) {
			cTemplate = TRUE;
			hasOpt = TRUE;
		}

		if (args_info.proto_given) {
			cudaTemplate = TRUE;
			kernelProto = parseGivenName(args_info.proto_arg);
			cudaTemplateOnlyKernelDefinition = TRUE;
			hasOpt = TRUE;
		}

		if (args_info.kernel_given) {
			cudaTemplate = TRUE;
			hasOpt = TRUE;
		}

		if (args_info.blocks_given) {
			cudaTemplate = TRUE;
			hasOpt = TRUE;
		}

		if (args_info.threads_given) {
			cudaTemplate = TRUE;
			hasOpt = TRUE;
		}

		if (!hasOpt) {
			ERROR(1,
			      "Are you sleeping?Hello!!!Where is the option dude?\n");
		}

		if (cudaTemplate && cTemplate) {
			ERROR(2,
			      "Only one option required: regular C template or another CUDA template\n");
		}

		dirname = parseGivenName(args_info.directory_arg);

		createDirectoryAndHeaderFile(forceByDefault, dirname, &path);

		if (cTemplate || cudaTemplateOnlyKernelDefinition) {
			generateStaticTemplate(dirname, path, cudaTemplate,
					       kernelProto);

			printf("OBJETIVO CONCLUIDO =)\n");
			return 0;
		} else {
			/* PRONTO SE NÃO ENTRAR NO IF ATRAS É PORQUE É UMA DAS VOSSAS OPÇÕES!
			 * SÓ FALTA FAZER O MAKEFILE DO TEMPLATE (NÃO COLOQUEI PORQUE O HÉLIO FALOU DAQUILO DOS TEMPLATES E PRONTO)
			 */
		}
	}
	return 0;
}
