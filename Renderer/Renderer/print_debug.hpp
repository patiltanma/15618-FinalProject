#ifndef PRINT_TERRAIN_DEBUG
#include "erosion_kernel.h"
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>

void debug_print_terrain(cellData* map, cellData* new_map, int map_dim);
void debug_print_water(struct cellData* map, struct cellData* new_map, int map_dim);
void debug_print_sediment(struct cellData* map, struct cellData* new_map, int map_dim);

__device__ void debug_print_cell(int x, int y, struct cellData *cell);
__device__ void debug_printdump();
__device__ void debug_compare_maps(struct cellData* total_map, struct cellData* new_total_map);


#endif