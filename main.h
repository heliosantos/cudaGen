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

int fill_grid_dim(Coords3D *grid_dim, struct gengetopt_args_info *args_info);

int fill_block_dim(Coords3D *block_dim, struct gengetopt_args_info *args_info);

#endif
