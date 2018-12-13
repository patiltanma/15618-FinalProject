#include "print_debug.hpp"
#include <stdio.h>

#ifdef DEBUG
#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s at %s:%d\n",
			cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
#else
#define cudaCheckError(ans) ans
#endif



__device__ void print_cell(int x, int y, cellData* cell) {
#ifdef DEBUG
	printf("Cell %i,%i: h:%f wvol:%f wheght: %f sed:%f",
		x, y, cell->height, cell->water_vol, cell->water_height, cell->sediment);
	/*printf("sediment cap: %f, new erosion: %f new deposition: %f\n",sediment_capacity,new_erosion,new_deposition);*/
	/*printf("water height_dir: %f %f %f %f\n",*/
	/*water_height_dir.x, water_height_dir.y, water_height_dir.w, water_height_dir.z);*/
	/*printf("water flux_dir: %f %f %f %f\nsediment_mov_dir %f %f %f %f\n", water_flux.x, water_flux.y, water_flux.w, water_flux.z,*/
	/*sediment_movement_dir.x, sediment_movement_dir.y, sediment_movement_dir.w, sediment_movement_dir.z);*/
	/*printf("height %f -> %f\n", total_map[index].height, new_total_map[index].height);*/
#endif
}

__device__ void debug_erode_dump(int print_ind, int mapDim, int globalMapDim, cellData* map, cellData* cell, cellData* newcell,
	float sediment_capacity, float cell_erosion, float new_deposition,
	float4 water_height_dir, float4 delta_h_dir, float4 v_dir,
	float4 water_flux, float4 water_flux_norm, float4 total_water_flux, float4 sediment_flux)
{
#ifdef DEBUG
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int index = x + globalMapDim * y;

	int left = max(0, x - 1);
	int right = min(x + 1, mapDim - 1);
	int up = max(0, y - 1);
	int down = min(y + 1, mapDim - 1);

	if (index == print_ind) {
		printf("\n");
		print_cell(x, y, cell);
		print_cell(x, y, newcell);

		printf("\ncell water_height: %f\n", cell->water_height);
		printf("water_height_dir: %f %f %f %f\n", water_height_dir.x, water_height_dir.y, water_height_dir.w, water_height_dir.z);
		printf("slope dir: %f %f %f %f\n", delta_h_dir.x, delta_h_dir.y, delta_h_dir.w, delta_h_dir.z);
		printf("v_dir: %f %f %f %f\n", v_dir.x, v_dir.y, v_dir.w, v_dir.z);

		printf("sediment cap: %f, new erosion: %f new deposition: %f\n", sediment_capacity, cell_erosion, new_deposition);
		/*printf("height change: %f -> %f : %f\n", oldheight, newheight,  newheight-oldheight);*/

		printf("slope dir: %f %f %f %f\n", delta_h_dir.x, delta_h_dir.y, delta_h_dir.w, delta_h_dir.z);
		printf("water height_dir: %f %f %f %f\n",
			water_height_dir.x, water_height_dir.y, water_height_dir.w, water_height_dir.z);

		printf("total water flux %f\n", total_water_flux);
		printf("water flux_dir: %f %f %f %f\nsediment_flux_dir %f %f %f %f\n", water_flux.x, water_flux.y, water_flux.w, water_flux.z,
			sediment_flux.x, sediment_flux.y, sediment_flux.w, sediment_flux.z);
		printf("norm water flux: %f %f %f %f\n",
			water_flux_norm.x, water_flux_norm.y, water_flux_norm.w, water_flux_norm.z);

		printf("\nold water vol: %f -> new water vol %f\n", cell->water_vol*WATER_LOSS, sum(water_flux) + newcell->water_vol);

		print_cell(x, y, cell);
		print_cell(x - 1, y, &map[left + y * mapDim]);
		print_cell(x + 1, y, &map[right + y * mapDim]);
		print_cell(x, y - 1, &map[x + up + mapDim]);
		print_cell(x, y + 1, &map[x + down * mapDim]);
	}

#endif

}

void debug_print_terrain(cellData* map, cellData* old_map, int map_dim) {
#ifdef DEBUG
	for (int x = 0; x < map_dim; x++) {
		for (int y = 0; y < map_dim; y++) {
			if (map[x + map_dim * y].height > old_map[x + y * map_dim].height) {
				printf("\033[37;1m%5.2f \033[0m", map[x + map_dim * y].height);
			}
			else if (map[x + map_dim * y].height < old_map[x + y * map_dim].height) {
				printf("\033[30;1m%5.2f \033[0m", map[x + map_dim * y].height);
			}
			else
				printf("%5.2f ", map[x + map_dim * y].height);
		}
		printf("\n");
	}
	printf("\033[0m\n");
#endif
}

void debug_print_water(cellData* map, cellData* old_map, int map_dim) {
#ifdef DEBUG
	for (int x = 0; x < map_dim; x++) {
		for (int y = 0; y < map_dim; y++) {
			int index = x + map_dim * y;
			if (map[index].water_vol > 0.005)
				printf("\033[36m%5.2f \033[0m", map[index].water_height);
			else
				printf("%5.2f ", map[index].water_height);
		}
		printf("\n");
	}
	printf("\033[0m\n");
#endif
}

void debug_print_sediment(cellData* map, cellData* old_map, int map_dim) {
#ifdef DEBUG
	for (int x = 0; x < map_dim; x++) {
		for (int y = 0; y < map_dim; y++) {
			int index = x + map_dim * y;
			if (map[index].sediment >= .005) {
				printf("\033[33m%5.2f \033[0m", map[index].sediment);
			}
			else {
				printf("%5.2f ", map[index].sediment);
			}
		}
		printf("\n");
	}
	printf("\033[0m\n");
#endif
}



__device__ void debug_compare_maps(cellData* total_map, cellData* new_total_map, int globalMapDim) {
#ifdef DEBUG

	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int index = x + globalMapDim * y;

	if (index == 0) {
		/*printf("\n\ncell %i %i\n", threadIdx.x, threadIdx.y);*/
		float old_volume = 0;
		float old_heightsum = 0;
		float old_sedsum = 0;
		float old_water_vol = 0;
		float new_water_volsum = 0;
		float new_volume = 0;
		float new_heightsum = 0;
		float new_sedsum = 0;

		for (int j = 0; j<blockDim.y; j++) {
			for (int i = 0; i<blockDim.x; i++) {

				old_heightsum += total_map[i + j * blockDim.x].height;
				old_sedsum += total_map[i + j * blockDim.x].sediment;
				old_water_vol += total_map[i + j * blockDim.x].water_vol;

				new_heightsum += new_total_map[i + j * blockDim.x].height;
				new_sedsum += new_total_map[i + j * blockDim.x].sediment;
				new_water_volsum += new_total_map[i + j * blockDim.x].water_vol;
			}
		}

		old_volume = old_heightsum + old_sedsum;
		new_volume = new_heightsum + new_sedsum;

		printf("old height: %f new height: %f\n", old_heightsum, new_heightsum);
		printf("old sed: %f new sed: %f\n", old_sedsum, new_sedsum);
		printf("old volume: %f new volume: %f diff:%f\n", old_volume, new_volume, new_volume - old_volume);
		printf("old water volume: %f new water volume: %f diff:%f\n", old_water_vol, new_water_volsum, new_water_volsum - old_water_vol);
	}
#endif
}


void printMap(float *map, int dim) {
#ifdef DEBUG

	for (int x = 0; x < dim; x++) {
		for (int y = 0; y < dim; y++) {
			printf("%4i ", int(map[x + dim * y]));
		}
		printf("\n");
	}
	printf("\n");
#endif
}