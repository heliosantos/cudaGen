/**
 * @file dirutils.c
 * @brief A set of functions that help using the directories
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
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <string.h>
#include <dirent.h>
#include <errno.h>
#include <limits.h>

#include "dirutils.h"
#include "3rdParty/debug.h"

/**
 * Checks if a directory exists
 *
 * @param directory directory name to be tested
 * @return (TRUE or FALSE)
 */
bool directoryExists(char *directory)
{
	struct stat st;

	//memset(&st, 0, sizeof(st));

	if (stat(directory, &st) == -1) {
	return FALSE} else {
		return TRUE;
	}
}

/**
 * Creates a directory tree (recursive mkdir)
 *
 * @param directory directory name to be created
 * @return nothing
 * @see http://nion.modprobe.de/blog/archives/357-Recursive-directory-creation.html
 */
void rmkdir(char *directory)
{
    char tmp[PATH_MAX];
    char *p = NULL;
    size_t len;
    
    snprintf(tmp, sizeof(tmp), "%s", directory);
    len = strlen(tmp);
    if (tmp[len - 1] == '/')
        tmp[len - 1] = 0;
    for (p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            // S_IRWXU = 00700 -> mask for file owner permissions (from man 2 stat)
            mkdir(tmp, S_IRWXU);
            *p = '/';
        }
    }
    mkdir(tmp, S_IRWXU);
}


/**
 * Function to check if directory exists
 *
 * @param dirname directory name to be created
 * @return result (TRUE or FALSE)
 * @see 
 */
bool createDirectory(char *directory)
{
	if (!directoryExists(directory)) {
		rmkdir(directory);

		return TRUE;
	}

	return FALSE;
}



/**
 * Function to prevent white spaces in directory name string
 *
 * @param givenName directory name to be parsed
 * @return void
 * @see 
 */
void parseGivenName(char *givenName)
{
	unsigned int i = 0;
	for (; i < strlen(givenName); i++) {
		if (givenName[i] == ' ') {
			givenName[i] = '_';
		}
	}
}

/**
 * Function to remove directory if exists
 * @author http://stackoverflow.com/questions/2256945/removing-a-non-empty-directory-programmatically-in-c-or-c
 * @date 05/11/2012
 * @param path path of directory
 * @return result (TRUE or FALSE)
 * @see 
 */
int remove_directory(const char *path)
{
	DIR *d;

	if ((d = opendir(path)) == NULL)
		ERROR(4, "Can't open dir to write");

	size_t path_len = strlen(path);
	int r = -1;

	if (d) {
		struct dirent *p;

		r = 0;

		while (!r && (p = readdir(d))) {
			int r2 = -1;
			char *buf;
			size_t len;

			/* Skip the names "." and ".." as we don't want to recurse on them. */
			if (!strcmp(p->d_name, ".") || !strcmp(p->d_name, "..")) {
				continue;
			}

			len = path_len + strlen(p->d_name) + 2;
			buf = malloc(len);

			if (buf) {
				struct stat statbuf;

				snprintf(buf, len, "%s/%s", path, p->d_name);

				if (!stat(buf, &statbuf)) {
					if (S_ISDIR(statbuf.st_mode)) {
						r2 = remove_directory(buf);
					} else {
						r2 = unlink(buf);
					}
				}

				free(buf);
			}

			r = r2;
		}

		closedir(d);
	}

	if (!r) {
		r = rmdir(path);
	}

	return r;
}

/**
 * Gets the file (or last directory) from a path
 *
 * @param path the path
 * @param filename a string to store the filename
 *
 * @return void
 */
void getFilenameFromPath(char *path, char *filename){
	
	char *result = NULL;
	char *temp = NULL;
	char pathCopy[PATH_MAX] = "";	
	
	strcpy(pathCopy, path);
	
	result = strtok(pathCopy, "/");
	while(result != NULL){
		temp = result;
		result = strtok(NULL, "/");
	}
	strcpy(filename, temp);
}

