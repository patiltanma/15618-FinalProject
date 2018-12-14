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
	float* height;
	float* water;
	float* sediment;
};

struct vrc_cellData {
	struct cudaGraphicsResource **height;
	struct cudaGraphicsResource **water;
	struct cudaGraphicsResource **sediment;
};

void runCuda(struct cudaGraphicsResource **vbo_resource, unsigned int, unsigned int, float);
void erodeCuda(struct cudaGraphicsResource **cvr_height, struct cudaGraphicsResource **cvr_water, struct cudaGraphicsResource **cvr_sediment,
	struct cudaGraphicsResource **cvr_new_height, struct cudaGraphicsResource **cvr_new_water, struct cudaGraphicsResource **cvr_new_sediment, 
	struct cudaGraphicsResource **cvr_rain, int mesh_dim);

