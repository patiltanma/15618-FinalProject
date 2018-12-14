/*
This example demonstrates how to use the Cuda OpenGL bindings to
dynamically modify a vertex buffer using a Cuda kernel.

The steps are:
1. Create an empty vertex buffer object (VBO)
2. Register the VBO with Cuda
3. Map the VBO for writing from Cuda
4. Run Cuda kernel to modify the vertex positions
5. Unmap the VBO
6. Render the results using OpenGL

Host code
*/

#include "global.h"
#include "open_gl_utils.h"
#include "cuda_utils.h"
#include "FastNoise.hpp"
#include "cycleTimer.h"

//#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>


FastNoise myNoise; // Create a FastNoise object
const char *sSDKsample = "simpleGL (VBO)";
int *pArgc = NULL;
char **pArgv = NULL;


float rainCenterCircle(int x, int y);
float rainBar(int x, int y);
float conicHeight(int x, int y);
float mountainHeight(int x, int y);
float slantHeight(int x, int y);
float randomHeight(int x, int y);


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	printf("%s starting...\n", sSDKsample);
	printf("\n");

	pArgc = &argc;
	pArgv = argv;

	float(*heightFunc[4])(int x, int y);
	heightFunc[0] = randomHeight;
	heightFunc[1] = mountainHeight;
	heightFunc[2] = conicHeight;
	heightFunc[3] = slantHeight;

	float(*rainFunc[2])(int x, int y);
	rainFunc[0] = rainBar;
	rainFunc[1] = rainCenterCircle;

	myNoise.SetNoiseType(FastNoise::SimplexFractal); // Set the desired noise type

	int h_ind = 2;
	int r_ind = 0;

	int mapSize = MESH_DIM * MESH_DIM;
	
	float *height = (float*) calloc(mapSize, sizeof(float)); // 2D heightmap to create terrain
	float *water = (float*)calloc(mapSize, sizeof(float)); // 2D map of precipitation values
	float *sediment = (float*)calloc(mapSize, sizeof(float));

	float *new_height = (float*)calloc(mapSize, sizeof(float));    // 2D heightmap to create terrain
	float *new_water = (float*)calloc(mapSize, sizeof(float)); // 2D map of precipitation values
	float *new_sediment = (float*)calloc(mapSize, sizeof(float)); // 2D map of precipitation values

	float *rain = (float*)calloc(mapSize, sizeof(float));

	float heightsum = 0.0f;
	float new_heightsum = 0.0f;

	for (int x = 0; x < MESH_DIM; x++) {
		for (int y = 0; y < MESH_DIM; y++) {
			int index = x + MESH_DIM * y;

			height[index] = heightFunc[h_ind](x, y);
			rain[index] = rainFunc[r_ind](x, y);
			heightsum += heightFunc[h_ind](x, y);
		}
	}

	cellData *map = (cellData*) malloc(sizeof(cellData));
	map->height = height;
	map->water = water;
	map->sediment = sediment;

	cellData *new_map = (cellData*)malloc(sizeof(cellData));
	new_map->height = new_height;
	new_map->water = new_water;
	new_map->sediment = new_sediment;



	initAndRunGL(argc, argv, map, new_map, rain);
	//erodeCuda(cellMap, rainMap, dim, 1);

	// sanity checks
	for (int x = 0; x < MESH_DIM; x++) {
		for (int y = 0; y < MESH_DIM; y++) {
			int index = x + MESH_DIM * y;
			new_heightsum += map->height[index];
		}
	}

	printf("old height: %f new height: %f\n", heightsum, new_heightsum);
	printf("%s completed!\n", sSDKsample);

	exit(0);
}

// return GB/s
float toBW(int bytes, float sec) {
	return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}

float rainCenterCircle(int x, int y) {
	float dist = sqrt(pow(x - MESH_DIM / 2, 2) + pow(y - MESH_DIM / 2, 2));
	if (dist < MESH_DIM) {
		return 1.0;
	}
	else {
		return 0.0;
	}
}

float rainBar(int x, int y) {
	if (y == 4) {
		return 1.0;
	}
	else
		return 0.0;
}

float conicHeight(int x, int y) {
	float dist = sqrt(pow(x - MESH_DIM / 2, 2) + pow(y - MESH_DIM / 2, 2));
	return 100 * (dist / (MESH_DIM));
}

float mountainHeight(int x, int y) {
	float dist = sqrt(pow(x - MESH_DIM / 2, 2) + pow(y - MESH_DIM / 2, 2));
	return 100 * (1 - dist / (MESH_DIM));
}

float slantHeight(int x, int y) { return 100 * (MESH_DIM - y) / (MESH_DIM); }

float randomHeight(int x, int y) { return 100 * (1 + myNoise.GetNoise(x, y)) / 2.0f; }




