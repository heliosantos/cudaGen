/** 
 *  @file main.h
 *  @brief O ficheiro header para a unidade principal do Cudagen
 *  @author 2120916@my.ipleiria.pt
 *  @author 
 *  @author 
 */

#ifndef MAIN_H
#define MAIN_H

#include "cmdline.h"
#include "3rdParty/hashtables.h"
#include "templateGen.h"

void store_grid_geometry(HASHTABLE_T * table, struct gengetopt_args_info *args_info);

void store_blocks_geometry(HASHTABLE_T * table, struct gengetopt_args_info *args_info);

#endif
