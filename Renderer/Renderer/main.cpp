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

	/*
	mapCell *heightMap = (mapCell *)malloc(sizeof(mapCell)*mesh_width*mesh_height);
	for (int xx = 0; xx < mesh_width; xx++) {
		for (int yy = 0; yy < mesh_height; yy++) {
			heightMap[yy*mesh_width + xx].x = xx;
			heightMap[yy*mesh_width + xx].y = yy;
			heightMap[yy*mesh_width + xx].z = myNoise.GetNoise(xx, yy);
			printf("%f\n", heightMap[yy*mesh_width + xx].z);
		}
	}
	*/

	int h_ind = 2;
	int r_ind = 0;

	auto *cellMap = new cellData[mesh_width * mesh_height]; // 2D heightmap to create terrain
	auto *oldheightMap = new float[mesh_width * mesh_height];    // 2D heightmap to create terrain
	float *rainMap = new float[mesh_width * mesh_height]; // 2D map of precipitation values

	float heightsum = 0.0f;
	float new_heightsum = 0.0f;

	for (int x = 0; x < mesh_width; x++) {
		for (int y = 0; y < mesh_height; y++) {
			int index = x + mesh_width * y;

			cellMap[index].height = heightFunc[h_ind](x, y);
			cellMap[index].water_height = heightFunc[h_ind](x, y);
			rainMap[index] = rainFunc[r_ind](x, y);

			oldheightMap[index] = heightFunc[h_ind](x, y);
			heightsum += heightFunc[h_ind](x, y);
		}
	}

	cellData* newMap = new cellData[mesh_width * mesh_height];
	memcpy(newMap, cellMap, mesh_width * mesh_height * sizeof(cellData));

	initAndRunGL(argc, argv, cellMap, newMap, rainMap);
	//erodeCuda(cellMap, rainMap, dim, 1);

	// sanity checks
	for (int x = 0; x < mesh_width; x++) {
		for (int y = 0; y < mesh_height; y++) {
			int index = x + mesh_width * y;
			new_heightsum += cellMap[index].height;
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
	float dist = sqrt(pow(x - mesh_width / 2, 2) + pow(y - mesh_height / 2, 2));
	if (dist < mesh_width) {
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
	float dist = sqrt(pow(x - mesh_width / 2, 2) + pow(y - mesh_height / 2, 2));
	return 100 * (dist / (mesh_width));
}

float mountainHeight(int x, int y) {
	float dist = sqrt(pow(x - mesh_width / 2, 2) + pow(y - mesh_height / 2, 2));
	return 100 * (1 - dist / (mesh_width));
}

float slantHeight(int x, int y) { return 100 * (mesh_height - y) / (mesh_width); }

float randomHeight(int x, int y) { return 100 * (1 + myNoise.GetNoise(x, y)) / 2.0f; }




