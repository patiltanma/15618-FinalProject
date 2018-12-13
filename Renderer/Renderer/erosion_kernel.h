#pragma once



//simulation constants
#define G         1.0
#define WATER_LOSS .98
#define DEEP_WATER 10.0
#define VEL_LOSS   0.9
#define SOLUBILITY 1.0
#define ABRAISION  .01


// block size
#define BLOCKDIM 32

//#define MAX(a,b) ((a > b) ? a : b)

struct cellData {
	float height;
	float water_height;
	float water_vol;
	float sediment;
};

void runCuda(struct cudaGraphicsResource **vbo_resource, unsigned int, unsigned int, float);
void erodeCuda(struct cudaGraphicsResource **vbo_resource_map, struct cudaGraphicsResource **vbo_resource_new_map,
	struct cudaGraphicsResource **vbo_resource_rain_map, unsigned int, unsigned int);

