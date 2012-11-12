/*
  File autogenerated by gengetopt version 2.22.5
  generated with the following command:
  gengetopt 

  The developers of gengetopt consider the fixed text that goes in all
  gengetopt output files to be in the public domain:
  we make no copyright claims on it.
*/

/* If we use autoconf.  */
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef FIX_UNUSED
#define FIX_UNUSED(X) (void) (X) /* avoid warnings for unused params */
#endif

#include <getopt.h>

#include "cmdline.h"

const char *gengetopt_args_info_purpose = "cudaGen gengetopt options file";

const char *gengetopt_args_info_usage = "Usage: cudaGen [OPTIONS]...";

const char *gengetopt_args_info_description = "";

const char *gengetopt_args_info_help[] = {
  "  -h, --help             Print help and exit",
  "  -V, --version          Print version and exit",
  "\nRequired group options:",
  "\n Group: what I want todo\n  script option",
  "  -d, --dir[=STRING]     Output directory. If it already exists, a new one will \n                           be created adding YYYY.MM.DD---HHhMM.SSs  \n                           (default=`OutDir')",
  "      --about            the about info (authors, date, etc.)",
  "\noptional options:",
  "  -r, --regular-code     Generate simple C template without CUDA  (default=off)",
  "  -F, --Force            force destination directory  (default=off)",
  "  -m, --measure          kernel measure GPU execution time  (default=off)",
  "  -s, --student=STRING   List of variables that should not be included in the \n                           template",
  "\ngeometry options:",
  "\n Mode: modeGeometry\n  Geometry mode options",
  "  -b, --blocks=INT       Kernel geometry in blocks",
  "  -t, --threads=INT      threads geometry",
  "\nkernel mode options:",
  "\n Mode: modeKernel\n  Kernel mode options",
  "  -k, --kernel[=STRING]  Kernel name  (default=`kernel')",
  "  -p, --proto=STRING     kernel proto",
    0
};

typedef enum {ARG_NO
  , ARG_FLAG
  , ARG_STRING
  , ARG_INT
} cmdline_parser_arg_type;

static
void clear_given (struct gengetopt_args_info *args_info);
static
void clear_args (struct gengetopt_args_info *args_info);

static int
cmdline_parser_internal (int argc, char **argv, struct gengetopt_args_info *args_info,
                        struct cmdline_parser_params *params, const char *additional_error);

static int
cmdline_parser_required2 (struct gengetopt_args_info *args_info, const char *prog_name, const char *additional_error);

static char *
gengetopt_strdup (const char *s);

static
void clear_given (struct gengetopt_args_info *args_info)
{
  args_info->help_given = 0 ;
  args_info->version_given = 0 ;
  args_info->dir_given = 0 ;
  args_info->about_given = 0 ;
  args_info->regular_code_given = 0 ;
  args_info->Force_given = 0 ;
  args_info->measure_given = 0 ;
  args_info->student_given = 0 ;
  args_info->blocks_given = 0 ;
  args_info->threads_given = 0 ;
  args_info->kernel_given = 0 ;
  args_info->proto_given = 0 ;
  args_info->what_I_want_todo_group_counter = 0 ;
  args_info->modeGeometry_mode_counter = 0 ;
  args_info->modeKernel_mode_counter = 0 ;
}

static
void clear_args (struct gengetopt_args_info *args_info)
{
  FIX_UNUSED (args_info);
  args_info->dir_arg = gengetopt_strdup ("OutDir");
  args_info->dir_orig = NULL;
  args_info->regular_code_flag = 0;
  args_info->Force_flag = 0;
  args_info->measure_flag = 0;
  args_info->student_arg = NULL;
  args_info->student_orig = NULL;
  args_info->blocks_arg = NULL;
  args_info->blocks_orig = NULL;
  args_info->threads_arg = NULL;
  args_info->threads_orig = NULL;
  args_info->kernel_arg = gengetopt_strdup ("kernel");
  args_info->kernel_orig = NULL;
  args_info->proto_arg = NULL;
  args_info->proto_orig = NULL;
  
}

static
void init_args_info(struct gengetopt_args_info *args_info)
{


  args_info->help_help = gengetopt_args_info_help[0] ;
  args_info->version_help = gengetopt_args_info_help[1] ;
  args_info->dir_help = gengetopt_args_info_help[4] ;
  args_info->about_help = gengetopt_args_info_help[5] ;
  args_info->regular_code_help = gengetopt_args_info_help[7] ;
  args_info->Force_help = gengetopt_args_info_help[8] ;
  args_info->measure_help = gengetopt_args_info_help[9] ;
  args_info->student_help = gengetopt_args_info_help[10] ;
  args_info->student_min = 0;
  args_info->student_max = 0;
  args_info->blocks_help = gengetopt_args_info_help[13] ;
  args_info->blocks_min = 1;
  args_info->blocks_max = 3;
  args_info->threads_help = gengetopt_args_info_help[14] ;
  args_info->threads_min = 1;
  args_info->threads_max = 3;
  args_info->kernel_help = gengetopt_args_info_help[17] ;
  args_info->proto_help = gengetopt_args_info_help[18] ;
  
}

void
cmdline_parser_print_version (void)
{
  printf ("%s %s\n",
     (strlen(CMDLINE_PARSER_PACKAGE_NAME) ? CMDLINE_PARSER_PACKAGE_NAME : CMDLINE_PARSER_PACKAGE),
     CMDLINE_PARSER_VERSION);
}

static void print_help_common(void) {
  cmdline_parser_print_version ();

  if (strlen(gengetopt_args_info_purpose) > 0)
    printf("\n%s\n", gengetopt_args_info_purpose);

  if (strlen(gengetopt_args_info_usage) > 0)
    printf("\n%s\n", gengetopt_args_info_usage);

  printf("\n");

  if (strlen(gengetopt_args_info_description) > 0)
    printf("%s\n\n", gengetopt_args_info_description);
}

void
cmdline_parser_print_help (void)
{
  int i = 0;
  print_help_common();
  while (gengetopt_args_info_help[i])
    printf("%s\n", gengetopt_args_info_help[i++]);
}

void
cmdline_parser_init (struct gengetopt_args_info *args_info)
{
  clear_given (args_info);
  clear_args (args_info);
  init_args_info (args_info);
}

void
cmdline_parser_params_init(struct cmdline_parser_params *params)
{
  if (params)
    { 
      params->override = 0;
      params->initialize = 1;
      params->check_required = 1;
      params->check_ambiguity = 0;
      params->print_errors = 1;
    }
}

struct cmdline_parser_params *
cmdline_parser_params_create(void)
{
  struct cmdline_parser_params *params = 
    (struct cmdline_parser_params *)malloc(sizeof(struct cmdline_parser_params));
  cmdline_parser_params_init(params);  
  return params;
}

static void
free_string_field (char **s)
{
  if (*s)
    {
      free (*s);
      *s = 0;
    }
}

/** @brief generic value variable */
union generic_value {
    int int_arg;
    char *string_arg;
    const char *default_string_arg;
};

/** @brief holds temporary values for multiple options */
struct generic_list
{
  union generic_value arg;
  char *orig;
  struct generic_list *next;
};

/**
 * @brief add a node at the head of the list 
 */
static void add_node(struct generic_list **list) {
  struct generic_list *new_node = (struct generic_list *) malloc (sizeof (struct generic_list));
  new_node->next = *list;
  *list = new_node;
  new_node->arg.string_arg = 0;
  new_node->orig = 0;
}

/**
 * The passed arg parameter is NOT set to 0 from this function
 */
static void
free_multiple_field(unsigned int len, void *arg, char ***orig)
{
  unsigned int i;
  if (arg) {
    for (i = 0; i < len; ++i)
      {
        free_string_field(&((*orig)[i]));
      }

    free (arg);
    free (*orig);
    *orig = 0;
  }
}

static void
free_multiple_string_field(unsigned int len, char ***arg, char ***orig)
{
  unsigned int i;
  if (*arg) {
    for (i = 0; i < len; ++i)
      {
        free_string_field(&((*arg)[i]));
        free_string_field(&((*orig)[i]));
      }
    free_string_field(&((*arg)[0])); /* free default string */

    free (*arg);
    *arg = 0;
    free (*orig);
    *orig = 0;
  }
}

static void
cmdline_parser_release (struct gengetopt_args_info *args_info)
{

  free_string_field (&(args_info->dir_arg));
  free_string_field (&(args_info->dir_orig));
  free_multiple_string_field (args_info->student_given, &(args_info->student_arg), &(args_info->student_orig));
  free_multiple_field (args_info->blocks_given, (void *)(args_info->blocks_arg), &(args_info->blocks_orig));
  args_info->blocks_arg = 0;
  free_multiple_field (args_info->threads_given, (void *)(args_info->threads_arg), &(args_info->threads_orig));
  args_info->threads_arg = 0;
  free_string_field (&(args_info->kernel_arg));
  free_string_field (&(args_info->kernel_orig));
  free_string_field (&(args_info->proto_arg));
  free_string_field (&(args_info->proto_orig));
  
  

  clear_given (args_info);
}


static void
write_into_file(FILE *outfile, const char *opt, const char *arg, const char *values[])
{
  FIX_UNUSED (values);
  if (arg) {
    fprintf(outfile, "%s=\"%s\"\n", opt, arg);
  } else {
    fprintf(outfile, "%s\n", opt);
  }
}

static void
write_multiple_into_file(FILE *outfile, int len, const char *opt, char **arg, const char *values[])
{
  int i;
  
  for (i = 0; i < len; ++i)
    write_into_file(outfile, opt, (arg ? arg[i] : 0), values);
}

int
cmdline_parser_dump(FILE *outfile, struct gengetopt_args_info *args_info)
{
  int i = 0;

  if (!outfile)
    {
      fprintf (stderr, "%s: cannot dump options to stream\n", CMDLINE_PARSER_PACKAGE);
      return EXIT_FAILURE;
    }

  if (args_info->help_given)
    write_into_file(outfile, "help", 0, 0 );
  if (args_info->version_given)
    write_into_file(outfile, "version", 0, 0 );
  if (args_info->dir_given)
    write_into_file(outfile, "dir", args_info->dir_orig, 0);
  if (args_info->about_given)
    write_into_file(outfile, "about", 0, 0 );
  if (args_info->regular_code_given)
    write_into_file(outfile, "regular-code", 0, 0 );
  if (args_info->Force_given)
    write_into_file(outfile, "Force", 0, 0 );
  if (args_info->measure_given)
    write_into_file(outfile, "measure", 0, 0 );
  write_multiple_into_file(outfile, args_info->student_given, "student", args_info->student_orig, 0);
  write_multiple_into_file(outfile, args_info->blocks_given, "blocks", args_info->blocks_orig, 0);
  write_multiple_into_file(outfile, args_info->threads_given, "threads", args_info->threads_orig, 0);
  if (args_info->kernel_given)
    write_into_file(outfile, "kernel", args_info->kernel_orig, 0);
  if (args_info->proto_given)
    write_into_file(outfile, "proto", args_info->proto_orig, 0);
  

  i = EXIT_SUCCESS;
  return i;
}

int
cmdline_parser_file_save(const char *filename, struct gengetopt_args_info *args_info)
{
  FILE *outfile;
  int i = 0;

  outfile = fopen(filename, "w");

  if (!outfile)
    {
      fprintf (stderr, "%s: cannot open file for writing: %s\n", CMDLINE_PARSER_PACKAGE, filename);
      return EXIT_FAILURE;
    }

  i = cmdline_parser_dump(outfile, args_info);
  fclose (outfile);

  return i;
}

void
cmdline_parser_free (struct gengetopt_args_info *args_info)
{
  cmdline_parser_release (args_info);
}

/** @brief replacement of strdup, which is not standard */
char *
gengetopt_strdup (const char *s)
{
  char *result = 0;
  if (!s)
    return result;

  result = (char*)malloc(strlen(s) + 1);
  if (result == (char*)0)
    return (char*)0;
  strcpy(result, s);
  return result;
}

static char *
get_multiple_arg_token(const char *arg)
{
  const char *tok;
  char *ret;
  size_t len, num_of_escape, i, j;

  if (!arg)
    return 0;

  tok = strchr (arg, ',');
  num_of_escape = 0;

  /* make sure it is not escaped */
  while (tok)
    {
      if (*(tok-1) == '\\')
        {
          /* find the next one */
          tok = strchr (tok+1, ',');
          ++num_of_escape;
        }
      else
        break;
    }

  if (tok)
    len = (size_t)(tok - arg + 1);
  else
    len = strlen (arg) + 1;

  len -= num_of_escape;

  ret = (char *) malloc (len);

  i = 0;
  j = 0;
  while (arg[i] && (j < len-1))
    {
      if (arg[i] == '\\' && 
	  arg[ i + 1 ] && 
	  arg[ i + 1 ] == ',')
        ++i;

      ret[j++] = arg[i++];
    }

  ret[len-1] = '\0';

  return ret;
}

static const char *
get_multiple_arg_token_next(const char *arg)
{
  const char *tok;

  if (!arg)
    return 0;

  tok = strchr (arg, ',');

  /* make sure it is not escaped */
  while (tok)
    {
      if (*(tok-1) == '\\')
        {
          /* find the next one */
          tok = strchr (tok+1, ',');
        }
      else
        break;
    }

  if (! tok || strlen(tok) == 1)
    return 0;

  return tok+1;
}

static int
check_multiple_option_occurrences(const char *prog_name, unsigned int option_given, unsigned int min, unsigned int max, const char *option_desc);

int
check_multiple_option_occurrences(const char *prog_name, unsigned int option_given, unsigned int min, unsigned int max, const char *option_desc)
{
  int error = 0;

  if (option_given && (min > 0 || max > 0))
    {
      if (min > 0 && max > 0)
        {
          if (min == max)
            {
              /* specific occurrences */
              if (option_given != (unsigned int) min)
                {
                  fprintf (stderr, "%s: %s option occurrences must be %d\n",
                    prog_name, option_desc, min);
                  error = 1;
                }
            }
          else if (option_given < (unsigned int) min
                || option_given > (unsigned int) max)
            {
              /* range occurrences */
              fprintf (stderr, "%s: %s option occurrences must be between %d and %d\n",
                prog_name, option_desc, min, max);
              error = 1;
            }
        }
      else if (min > 0)
        {
          /* at least check */
          if (option_given < min)
            {
              fprintf (stderr, "%s: %s option occurrences must be at least %d\n",
                prog_name, option_desc, min);
              error = 1;
            }
        }
      else if (max > 0)
        {
          /* at most check */
          if (option_given > max)
            {
              fprintf (stderr, "%s: %s option occurrences must be at most %d\n",
                prog_name, option_desc, max);
              error = 1;
            }
        }
    }
    
  return error;
}
static void
reset_group_what_I_want_todo(struct gengetopt_args_info *args_info)
{
  if (! args_info->what_I_want_todo_group_counter)
    return;
  
  args_info->dir_given = 0 ;
  free_string_field (&(args_info->dir_arg));
  free_string_field (&(args_info->dir_orig));
  args_info->about_given = 0 ;

  args_info->what_I_want_todo_group_counter = 0;
}

int
cmdline_parser (int argc, char **argv, struct gengetopt_args_info *args_info)
{
  return cmdline_parser2 (argc, argv, args_info, 0, 1, 1);
}

int
cmdline_parser_ext (int argc, char **argv, struct gengetopt_args_info *args_info,
                   struct cmdline_parser_params *params)
{
  int result;
  result = cmdline_parser_internal (argc, argv, args_info, params, 0);

  if (result == EXIT_FAILURE)
    {
      cmdline_parser_free (args_info);
      exit (EXIT_FAILURE);
    }
  
  return result;
}

int
cmdline_parser2 (int argc, char **argv, struct gengetopt_args_info *args_info, int override, int initialize, int check_required)
{
  int result;
  struct cmdline_parser_params params;
  
  params.override = override;
  params.initialize = initialize;
  params.check_required = check_required;
  params.check_ambiguity = 0;
  params.print_errors = 1;

  result = cmdline_parser_internal (argc, argv, args_info, &params, 0);

  if (result == EXIT_FAILURE)
    {
      cmdline_parser_free (args_info);
      exit (EXIT_FAILURE);
    }
  
  return result;
}

int
cmdline_parser_required (struct gengetopt_args_info *args_info, const char *prog_name)
{
  int result = EXIT_SUCCESS;

  if (cmdline_parser_required2(args_info, prog_name, 0) > 0)
    result = EXIT_FAILURE;

  if (result == EXIT_FAILURE)
    {
      cmdline_parser_free (args_info);
      exit (EXIT_FAILURE);
    }
  
  return result;
}

int
cmdline_parser_required2 (struct gengetopt_args_info *args_info, const char *prog_name, const char *additional_error)
{
  int error = 0;
  FIX_UNUSED (additional_error);

  /* checks for required options */
  if (check_multiple_option_occurrences(prog_name, args_info->student_given, args_info->student_min, args_info->student_max, "'--student' ('-s')"))
     error = 1;
  
  if (args_info->modeGeometry_mode_counter && ! args_info->blocks_given)
    {
      fprintf (stderr, "%s: '--blocks' ('-b') option required%s\n", prog_name, (additional_error ? additional_error : ""));
      error = 1;
    }
  
  if (args_info->modeGeometry_mode_counter && check_multiple_option_occurrences(prog_name, args_info->blocks_given, args_info->blocks_min, args_info->blocks_max, "'--blocks' ('-b')"))
     error = 1;
  
  if (args_info->modeGeometry_mode_counter && check_multiple_option_occurrences(prog_name, args_info->threads_given, args_info->threads_min, args_info->threads_max, "'--threads' ('-t')"))
     error = 1;
  
  if (args_info->what_I_want_todo_group_counter == 0)
    {
      fprintf (stderr, "%s: %d options of group what I want todo were given. One is required%s.\n", prog_name, args_info->what_I_want_todo_group_counter, (additional_error ? additional_error : ""));
      error = 1;
    }
  

  /* checks for dependences among options */
  if (args_info->regular_code_given && ! args_info->dir_given)
    {
      fprintf (stderr, "%s: '--regular-code' ('-r') option depends on option 'dir'%s\n", prog_name, (additional_error ? additional_error : ""));
      error = 1;
    }
  if (args_info->Force_given && ! args_info->dir_given)
    {
      fprintf (stderr, "%s: '--Force' ('-F') option depends on option 'dir'%s\n", prog_name, (additional_error ? additional_error : ""));
      error = 1;
    }
  if (args_info->measure_given && ! args_info->dir_given)
    {
      fprintf (stderr, "%s: '--measure' ('-m') option depends on option 'dir'%s\n", prog_name, (additional_error ? additional_error : ""));
      error = 1;
    }
  if (args_info->student_given && ! args_info->dir_given)
    {
      fprintf (stderr, "%s: '--student' ('-s') option depends on option 'dir'%s\n", prog_name, (additional_error ? additional_error : ""));
      error = 1;
    }
  if (args_info->blocks_given && ! args_info->dir_given)
    {
      fprintf (stderr, "%s: '--blocks' ('-b') option depends on option 'dir'%s\n", prog_name, (additional_error ? additional_error : ""));
      error = 1;
    }
  if (args_info->threads_given && ! args_info->dir_given)
    {
      fprintf (stderr, "%s: '--threads' ('-t') option depends on option 'dir'%s\n", prog_name, (additional_error ? additional_error : ""));
      error = 1;
    }
  if (args_info->kernel_given && ! args_info->dir_given)
    {
      fprintf (stderr, "%s: '--kernel' ('-k') option depends on option 'dir'%s\n", prog_name, (additional_error ? additional_error : ""));
      error = 1;
    }
  if (args_info->proto_given && ! args_info->dir_given)
    {
      fprintf (stderr, "%s: '--proto' ('-p') option depends on option 'dir'%s\n", prog_name, (additional_error ? additional_error : ""));
      error = 1;
    }

  return error;
}


static char *package_name = 0;

/**
 * @brief updates an option
 * @param field the generic pointer to the field to update
 * @param orig_field the pointer to the orig field
 * @param field_given the pointer to the number of occurrence of this option
 * @param prev_given the pointer to the number of occurrence already seen
 * @param value the argument for this option (if null no arg was specified)
 * @param possible_values the possible values for this option (if specified)
 * @param default_value the default value (in case the option only accepts fixed values)
 * @param arg_type the type of this option
 * @param check_ambiguity @see cmdline_parser_params.check_ambiguity
 * @param override @see cmdline_parser_params.override
 * @param no_free whether to free a possible previous value
 * @param multiple_option whether this is a multiple option
 * @param long_opt the corresponding long option
 * @param short_opt the corresponding short option (or '-' if none)
 * @param additional_error possible further error specification
 */
static
int update_arg(void *field, char **orig_field,
               unsigned int *field_given, unsigned int *prev_given, 
               char *value, const char *possible_values[],
               const char *default_value,
               cmdline_parser_arg_type arg_type,
               int check_ambiguity, int override,
               int no_free, int multiple_option,
               const char *long_opt, char short_opt,
               const char *additional_error)
{
  char *stop_char = 0;
  const char *val = value;
  int found;
  char **string_field;
  FIX_UNUSED (field);

  stop_char = 0;
  found = 0;

  if (!multiple_option && prev_given && (*prev_given || (check_ambiguity && *field_given)))
    {
      if (short_opt != '-')
        fprintf (stderr, "%s: `--%s' (`-%c') option given more than once%s\n", 
               package_name, long_opt, short_opt,
               (additional_error ? additional_error : ""));
      else
        fprintf (stderr, "%s: `--%s' option given more than once%s\n", 
               package_name, long_opt,
               (additional_error ? additional_error : ""));
      return 1; /* failure */
    }

  FIX_UNUSED (default_value);
    
  if (field_given && *field_given && ! override)
    return 0;
  if (prev_given)
    (*prev_given)++;
  if (field_given)
    (*field_given)++;
  if (possible_values)
    val = possible_values[found];

  switch(arg_type) {
  case ARG_FLAG:
    *((int *)field) = !*((int *)field);
    break;
  case ARG_INT:
    if (val) *((int *)field) = strtol (val, &stop_char, 0);
    break;
  case ARG_STRING:
    if (val) {
      string_field = (char **)field;
      if (!no_free && *string_field)
        free (*string_field); /* free previous string */
      *string_field = gengetopt_strdup (val);
    }
    break;
  default:
    break;
  };

  /* check numeric conversion */
  switch(arg_type) {
  case ARG_INT:
    if (val && !(stop_char && *stop_char == '\0')) {
      fprintf(stderr, "%s: invalid numeric value: %s\n", package_name, val);
      return 1; /* failure */
    }
    break;
  default:
    ;
  };

  /* store the original value */
  switch(arg_type) {
  case ARG_NO:
  case ARG_FLAG:
    break;
  default:
    if (value && orig_field) {
      if (no_free) {
        *orig_field = value;
      } else {
        if (*orig_field)
          free (*orig_field); /* free previous string */
        *orig_field = gengetopt_strdup (value);
      }
    }
  };

  return 0; /* OK */
}

/**
 * @brief store information about a multiple option in a temporary list
 * @param list where to (temporarily) store multiple options
 */
static
int update_multiple_arg_temp(struct generic_list **list,
               unsigned int *prev_given, const char *val,
               const char *possible_values[], const char *default_value,
               cmdline_parser_arg_type arg_type,
               const char *long_opt, char short_opt,
               const char *additional_error)
{
  /* store single arguments */
  char *multi_token;
  const char *multi_next;

  if (arg_type == ARG_NO) {
    (*prev_given)++;
    return 0; /* OK */
  }

  multi_token = get_multiple_arg_token(val);
  multi_next = get_multiple_arg_token_next (val);

  while (1)
    {
      add_node (list);
      if (update_arg((void *)&((*list)->arg), &((*list)->orig), 0,
          prev_given, multi_token, possible_values, default_value, 
          arg_type, 0, 1, 1, 1, long_opt, short_opt, additional_error)) {
        if (multi_token) free(multi_token);
        return 1; /* failure */
      }

      if (multi_next)
        {
          multi_token = get_multiple_arg_token(multi_next);
          multi_next = get_multiple_arg_token_next (multi_next);
        }
      else
        break;
    }

  return 0; /* OK */
}

/**
 * @brief free the passed list (including possible string argument)
 */
static
void free_list(struct generic_list *list, short string_arg)
{
  if (list) {
    struct generic_list *tmp;
    while (list)
      {
        tmp = list;
        if (string_arg && list->arg.string_arg)
          free (list->arg.string_arg);
        if (list->orig)
          free (list->orig);
        list = list->next;
        free (tmp);
      }
  }
}

/**
 * @brief updates a multiple option starting from the passed list
 */
static
void update_multiple_arg(void *field, char ***orig_field,
               unsigned int field_given, unsigned int prev_given, union generic_value *default_value,
               cmdline_parser_arg_type arg_type,
               struct generic_list *list)
{
  int i;
  struct generic_list *tmp;

  if (prev_given && list) {
    *orig_field = (char **) realloc (*orig_field, (field_given + prev_given) * sizeof (char *));

    switch(arg_type) {
    case ARG_INT:
      *((int **)field) = (int *)realloc (*((int **)field), (field_given + prev_given) * sizeof (int)); break;
    case ARG_STRING:
      *((char ***)field) = (char **)realloc (*((char ***)field), (field_given + prev_given) * sizeof (char *)); break;
    default:
      break;
    };
    
    for (i = (prev_given - 1); i >= 0; --i)
      {
        tmp = list;
        
        switch(arg_type) {
        case ARG_INT:
          (*((int **)field))[i + field_given] = tmp->arg.int_arg; break;
        case ARG_STRING:
          (*((char ***)field))[i + field_given] = tmp->arg.string_arg; break;
        default:
          break;
        }        
        (*orig_field) [i + field_given] = list->orig;
        list = list->next;
        free (tmp);
      }
  } else { /* set the default value */
    if (default_value && ! field_given) {
      switch(arg_type) {
      case ARG_INT:
        if (! *((int **)field)) {
          *((int **)field) = (int *)malloc (sizeof (int));
          (*((int **)field))[0] = default_value->int_arg; 
        }
        break;
      case ARG_STRING:
        if (! *((char ***)field)) {
          *((char ***)field) = (char **)malloc (sizeof (char *));
          (*((char ***)field))[0] = gengetopt_strdup(default_value->string_arg);
        }
        break;
      default: break;
      }
      if (!(*orig_field)) {
        *orig_field = (char **) malloc (sizeof (char *));
        (*orig_field)[0] = 0;
      }
    }
  }
}

static int check_modes(
  int given1[], const char *options1[],
                       int given2[], const char *options2[])
{
  int i = 0, j = 0, errors = 0;
  
  while (given1[i] >= 0) {
    if (given1[i]) {
      while (given2[j] >= 0) {
        if (given2[j]) {
          ++errors;
          fprintf(stderr, "%s: option %s conflicts with option %s\n",
                  package_name, options1[i], options2[j]);
        }
        ++j;
      }
    }
    ++i;
  }
  
  return errors;
}

int
cmdline_parser_internal (
  int argc, char **argv, struct gengetopt_args_info *args_info,
                        struct cmdline_parser_params *params, const char *additional_error)
{
  int c;	/* Character of the parsed option.  */

  struct generic_list * student_list = NULL;
  struct generic_list * blocks_list = NULL;
  struct generic_list * threads_list = NULL;
  int error = 0;
  struct gengetopt_args_info local_args_info;
  
  int override;
  int initialize;
  int check_required;
  int check_ambiguity;
  
  package_name = argv[0];
  
  override = params->override;
  initialize = params->initialize;
  check_required = params->check_required;
  check_ambiguity = params->check_ambiguity;

  if (initialize)
    cmdline_parser_init (args_info);

  cmdline_parser_init (&local_args_info);

  optarg = 0;
  optind = 0;
  opterr = params->print_errors;
  optopt = '?';

  while (1)
    {
      int option_index = 0;

      static struct option long_options[] = {
        { "help",	0, NULL, 'h' },
        { "version",	0, NULL, 'V' },
        { "dir",	2, NULL, 'd' },
        { "about",	0, NULL, 0 },
        { "regular-code",	0, NULL, 'r' },
        { "Force",	0, NULL, 'F' },
        { "measure",	0, NULL, 'm' },
        { "student",	1, NULL, 's' },
        { "blocks",	1, NULL, 'b' },
        { "threads",	1, NULL, 't' },
        { "kernel",	2, NULL, 'k' },
        { "proto",	1, NULL, 'p' },
        { 0,  0, 0, 0 }
      };

      c = getopt_long (argc, argv, "hVd::rFms:b:t:k::p:", long_options, &option_index);

      if (c == -1) break;	/* Exit from `while (1)' loop.  */

      switch (c)
        {
        case 'h':	/* Print help and exit.  */
          cmdline_parser_print_help ();
          cmdline_parser_free (&local_args_info);
          exit (EXIT_SUCCESS);

        case 'V':	/* Print version and exit.  */
          cmdline_parser_print_version ();
          cmdline_parser_free (&local_args_info);
          exit (EXIT_SUCCESS);

        case 'd':	/* Output directory. If it already exists, a new one will be created adding YYYY.MM.DD---HHhMM.SSs.  */
        
          if (args_info->what_I_want_todo_group_counter && override)
            reset_group_what_I_want_todo (args_info);
          args_info->what_I_want_todo_group_counter += 1;
        
          if (update_arg( (void *)&(args_info->dir_arg), 
               &(args_info->dir_orig), &(args_info->dir_given),
              &(local_args_info.dir_given), optarg, 0, "OutDir", ARG_STRING,
              check_ambiguity, override, 0, 0,
              "dir", 'd',
              additional_error))
            goto failure;
        
          break;
        case 'r':	/* Generate simple C template without CUDA.  */
        
        
          if (update_arg((void *)&(args_info->regular_code_flag), 0, &(args_info->regular_code_given),
              &(local_args_info.regular_code_given), optarg, 0, 0, ARG_FLAG,
              check_ambiguity, override, 1, 0, "regular-code", 'r',
              additional_error))
            goto failure;
        
          break;
        case 'F':	/* force destination directory.  */
        
        
          if (update_arg((void *)&(args_info->Force_flag), 0, &(args_info->Force_given),
              &(local_args_info.Force_given), optarg, 0, 0, ARG_FLAG,
              check_ambiguity, override, 1, 0, "Force", 'F',
              additional_error))
            goto failure;
        
          break;
        case 'm':	/* kernel measure GPU execution time.  */
        
        
          if (update_arg((void *)&(args_info->measure_flag), 0, &(args_info->measure_given),
              &(local_args_info.measure_given), optarg, 0, 0, ARG_FLAG,
              check_ambiguity, override, 1, 0, "measure", 'm',
              additional_error))
            goto failure;
        
          break;
        case 's':	/* List of variables that should not be included in the template.  */
        
          if (update_multiple_arg_temp(&student_list, 
              &(local_args_info.student_given), optarg, 0, 0, ARG_STRING,
              "student", 's',
              additional_error))
            goto failure;
        
          break;
        case 'b':	/* Kernel geometry in blocks.  */
          args_info->modeGeometry_mode_counter += 1;
        
          if (update_multiple_arg_temp(&blocks_list, 
              &(local_args_info.blocks_given), optarg, 0, 0, ARG_INT,
              "blocks", 'b',
              additional_error))
            goto failure;
        
          break;
        case 't':	/* threads geometry.  */
          args_info->modeGeometry_mode_counter += 1;
        
          if (update_multiple_arg_temp(&threads_list, 
              &(local_args_info.threads_given), optarg, 0, 0, ARG_INT,
              "threads", 't',
              additional_error))
            goto failure;
        
          break;
        case 'k':	/* Kernel name.  */
          args_info->modeKernel_mode_counter += 1;
        
        
          if (update_arg( (void *)&(args_info->kernel_arg), 
               &(args_info->kernel_orig), &(args_info->kernel_given),
              &(local_args_info.kernel_given), optarg, 0, "kernel", ARG_STRING,
              check_ambiguity, override, 0, 0,
              "kernel", 'k',
              additional_error))
            goto failure;
        
          break;
        case 'p':	/* kernel proto.  */
          args_info->modeKernel_mode_counter += 1;
        
        
          if (update_arg( (void *)&(args_info->proto_arg), 
               &(args_info->proto_orig), &(args_info->proto_given),
              &(local_args_info.proto_given), optarg, 0, 0, ARG_STRING,
              check_ambiguity, override, 0, 0,
              "proto", 'p',
              additional_error))
            goto failure;
        
          break;

        case 0:	/* Long option with no short option */
          /* the about info (authors, date, etc.).  */
          if (strcmp (long_options[option_index].name, "about") == 0)
          {
          
            if (args_info->what_I_want_todo_group_counter && override)
              reset_group_what_I_want_todo (args_info);
            args_info->what_I_want_todo_group_counter += 1;
          
            if (update_arg( 0 , 
                 0 , &(args_info->about_given),
                &(local_args_info.about_given), optarg, 0, 0, ARG_NO,
                check_ambiguity, override, 0, 0,
                "about", '-',
                additional_error))
              goto failure;
          
          }
          
          break;
        case '?':	/* Invalid option.  */
          /* `getopt_long' already printed an error message.  */
          goto failure;

        default:	/* bug: option not considered.  */
          fprintf (stderr, "%s: option unknown: %c%s\n", CMDLINE_PARSER_PACKAGE, c, (additional_error ? additional_error : ""));
          abort ();
        } /* switch */
    } /* while */

  if (args_info->what_I_want_todo_group_counter > 1)
    {
      fprintf (stderr, "%s: %d options of group what I want todo were given. One is required%s.\n", argv[0], args_info->what_I_want_todo_group_counter, (additional_error ? additional_error : ""));
      error = 1;
    }
  

  update_multiple_arg((void *)&(args_info->student_arg),
    &(args_info->student_orig), args_info->student_given,
    local_args_info.student_given, 0,
    ARG_STRING, student_list);
  update_multiple_arg((void *)&(args_info->blocks_arg),
    &(args_info->blocks_orig), args_info->blocks_given,
    local_args_info.blocks_given, 0,
    ARG_INT, blocks_list);
  update_multiple_arg((void *)&(args_info->threads_arg),
    &(args_info->threads_orig), args_info->threads_given,
    local_args_info.threads_given, 0,
    ARG_INT, threads_list);

  args_info->student_given += local_args_info.student_given;
  local_args_info.student_given = 0;
  args_info->blocks_given += local_args_info.blocks_given;
  local_args_info.blocks_given = 0;
  args_info->threads_given += local_args_info.threads_given;
  local_args_info.threads_given = 0;
  
  if (args_info->modeGeometry_mode_counter && args_info->modeKernel_mode_counter) {
    int modeGeometry_given[] = {args_info->blocks_given, args_info->threads_given,  -1};
    const char *modeGeometry_desc[] = {"--blocks", "--threads",  0};
    int modeKernel_given[] = {args_info->kernel_given, args_info->proto_given,  -1};
    const char *modeKernel_desc[] = {"--kernel", "--proto",  0};
    error += check_modes(modeGeometry_given, modeGeometry_desc, modeKernel_given, modeKernel_desc);
  }
  
  if (check_required)
    {
      error += cmdline_parser_required2 (args_info, argv[0], additional_error);
    }

  cmdline_parser_release (&local_args_info);

  if ( error )
    return (EXIT_FAILURE);

  return 0;

failure:
  free_list (student_list, 1 );
  free_list (blocks_list, 0 );
  free_list (threads_list, 0 );
  
  cmdline_parser_release (&local_args_info);
  return (EXIT_FAILURE);
}
