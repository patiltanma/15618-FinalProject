

//#include <cuda.h>
//#include <cuda_runtime.h>
//#include <driver_functions.h>

#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "helper_math.h"
#include "cycleTimer.h"
#include "erosion_kernel.h"
#include "print_debug.hpp"


#ifdef DEBUG
#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "CUDA Error: %s at %s:%d\n",
			cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
#else
#define cudaCheckError(ans) ans
#endif


extern float toBW(int bytes, float sec);
void swap_maps(float* map, float* new_map);

__device__ void
clear_dest_map(cellData* new_map, int numCells) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < numCells) {
		memset(&new_map[index], 0, sizeof(cellData));
	}
}


// construct a shared memory copy of the global map, appropriately 
// padding the edges of the simulation with same-height zero vel squares
__device__ void
get_map(cellData* map, cellData* total_map, int mapDim, int globalMapDim) {
	mapDim++;
	int start_x = blockIdx.x*blockDim.x - 1;
	int start_y = blockIdx.y*blockDim.y - 1;

	//TODO: literally any edge checking. assumption is that invalid cells
	//      get filled with their valid neightbors
	for (int local_x = threadIdx.x; local_x < mapDim; local_x += blockDim.x) {
		for (int local_y = threadIdx.y; local_y < mapDim; local_y += blockDim.y) {
			int local_ind = local_x + local_y * mapDim;
			int global_ind = start_x + local_x + (start_y + local_y)*globalMapDim;

			map[local_ind] = total_map[global_ind];
		}
	}
}


__global__ void add_rain(float* water, float* rain, int mesh_dim) {

	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int index = x + mesh_dim * y;

	water[index] += rain[index];
}


__global__ void
erode(float* height, float* water_vol, float* sediment, float* new_height, float* new_water_vol, float* new_sediment, int numCells, int globalMapDim) {

	//TODO: lock down block sizes and dimensioning scheme

	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int index = x + globalMapDim * y;

	int left = max(0, x - 1) + y * globalMapDim;
	int right = min(x + 1, globalMapDim - 1) + y * globalMapDim;
	int up = x + max(0, y - 1)*globalMapDim;
	int down = x + min(y + 1, globalMapDim - 1)*globalMapDim;

	// positive values indicate outward flow
	float4 water_vol_dir = make_float4(water_vol[left], water_vol[right], water_vol[up], water_vol[down]);
	float4 height_dir = make_float4(height[left], height[right], height[up], height[down]);
	float4 water_height_dir = height_dir + water_vol_dir;

	float cell_height = height[index];
	float cell_water_vol = water_vol[index];
	float cell_sediment = sediment[index];

	float cell_water_height = height[index] + cell_water_vol;
	float4 delta_h_dir = -(water_height_dir)+cell_water_height;

	//assume all water than can flow will AT CONSTANT SPEED
	float4 water_flow = clamp(delta_h_dir, 0.0, 1000) / 4;
	float total_water_flow = sum(water_flow);

	float total_water_flux = fminf(cell_water_vol, total_water_flow);

	float4 water_flux_norm = make_float4(0.0);
	float flux_fraction = 0.0;

	if (total_water_flux>0.0) {
		water_flux_norm = water_flow / total_water_flow;
		flux_fraction = total_water_flux / cell_water_vol;
	}

	/*if (total_water_flux > cell->water_vol){*/
	/*water_flux_norm = make_float4(1.0);*/
	/*flux_fraction = 1.0;*/
	/*}*/

	float4 water_flux = total_water_flux * water_flux_norm;


	//dump sediment
	float falloff = (DEEP_WATER - min(DEEP_WATER, cell_water_vol)) / DEEP_WATER; //maintain thin water assumption
	float sediment_capacity = falloff * cell_water_vol*SOLUBILITY; //might want to make this a max of flow or something

																   //velocity*steepness = material eroded cos(atan(2/abs(dH.x+dH.y)))
	float cell_erosion = total_water_flux * ABRAISION; //could add additional effects of sediment on erosion here

	float new_cell_deposition = fmaxf(0, cell_sediment + cell_erosion - sediment_capacity);
	float new_cell_sediment = fminf(cell_sediment + cell_erosion, sediment_capacity);

	float4 sediment_flux = flux_fraction * water_flux_norm*new_cell_sediment;

	//account for frictional and water volume losses
	float new_cell_water_vol = (cell_water_vol - total_water_flux) * WATER_LOSS;

	atomicExch(&new_height[index], cell_height + new_cell_deposition - cell_erosion);
	atomicExch(&new_sediment[index], new_cell_sediment - sum(sediment_flux));
	atomicExch(&new_water_vol[index], new_cell_water_vol);

	if (x>0) { // left neighbor
		atomicAdd(&new_water_vol[x - 1 + globalMapDim * y], water_flux.x);
		/*atomicAdd(&new_water_height[x-1+globalMapDim*y], water_flux.x);*/
		atomicAdd(&new_sediment[x - 1 + globalMapDim * y], sediment_flux.x);
	}
	/*__syncthreads();*/
	if (x<globalMapDim - 1) { //right neighbor
		atomicAdd(&new_water_vol[x + 1 + globalMapDim * y], water_flux.y);
		/*atomicAdd(&new_water_height[x+1+globalMapDim*y], water_flux.y);*/
		atomicAdd(&new_sediment[x + 1 + globalMapDim * y], sediment_flux.y);
	}
	/*__syncthreads();*/
	if (y<globalMapDim - 1) { //top neighbor
		atomicAdd(&new_water_vol[x + globalMapDim * (y + 1)], water_flux.w);
		/*atomicAdd(&new_ma->water_height[x+globalMapDim*(y+1)], water_flux.w);*/
		atomicAdd(&new_sediment[x + globalMapDim * (y + 1)], sediment_flux.w);
	}
	/*__syncthreads();*/
	if (y>0) { // bottom neighbor
		atomicAdd(&new_water_vol[x + globalMapDim * (y - 1)], water_flux.z);
		/*atomicAdd(&new_ma->water_height[x+globalMapDim*(y-1)], water_flux.z);*/
		atomicAdd(&new_sediment[x + globalMapDim * (y - 1)], sediment_flux.z);
	}

	/*__syncthreads();*/

	/*debug_printdump();*/
	/*debug_compare_maps(total_map, new_total_map);*/
}


void swap_maps(float* map, float* new_map) {

	float* temp = new_map;
	new_map = map;
	map = temp;
}

static bool state = true;

void 
erodeCuda(struct cudaGraphicsResource **cvr_height, struct cudaGraphicsResource **cvr_water, struct cudaGraphicsResource **cvr_sediment,
	struct cudaGraphicsResource **cvr_new_height, struct cudaGraphicsResource **cvr_new_water, struct cudaGraphicsResource **cvr_new_sediment,
	struct cudaGraphicsResource **cvr_rain, int mesh_dim) {

	size_t num_bytes;

	/*
	// map OpenGL buffer object for writing from CUDA
	float *d_height;
	checkCudaErrors(cudaGraphicsMapResources(1, cvr_height, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_height, &num_bytes, *cvr_height));

	float *d_water;
	checkCudaErrors(cudaGraphicsMapResources(1, cvr_water, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_water, &num_bytes, *cvr_water));

	float *d_sediment;
	checkCudaErrors(cudaGraphicsMapResources(1, cvr_sediment, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_sediment, &num_bytes, *cvr_sediment));

	float *d_new_height;
	checkCudaErrors(cudaGraphicsMapResources(1, cvr_new_height, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_new_height, &num_bytes, *cvr_new_height));

	float *d_new_water;
	checkCudaErrors(cudaGraphicsMapResources(1, cvr_new_water, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_new_water, &num_bytes, *cvr_new_water));

	float *d_new_sediment;
	checkCudaErrors(cudaGraphicsMapResources(1, cvr_new_sediment, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_new_sediment, &num_bytes, *cvr_new_sediment));

	float *d_rain;
	checkCudaErrors(cudaGraphicsMapResources(1, cvr_rain, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_rain, &num_bytes, *cvr_rain));
	*/
	
	// map OpenGL buffer object for writing from CUDA
	float *d_height, *d_water, *d_sediment, *d_new_height, *d_new_water, *d_new_sediment;
	if (state)
	{
		checkCudaErrors(cudaGraphicsMapResources(1, cvr_height, 0));
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_height, &num_bytes, *cvr_height));

		checkCudaErrors(cudaGraphicsMapResources(1, cvr_water, 0));
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_water, &num_bytes, *cvr_water));

		checkCudaErrors(cudaGraphicsMapResources(1, cvr_sediment, 0));
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_sediment, &num_bytes, *cvr_sediment));

		checkCudaErrors(cudaGraphicsMapResources(1, cvr_new_height, 0));
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_new_height, &num_bytes, *cvr_new_height));

		checkCudaErrors(cudaGraphicsMapResources(1, cvr_new_water, 0));
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_new_water, &num_bytes, *cvr_new_water));

		checkCudaErrors(cudaGraphicsMapResources(1, cvr_new_sediment, 0));
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_new_sediment, &num_bytes, *cvr_new_sediment));

		state = false;
	}
	else
	{
		checkCudaErrors(cudaGraphicsMapResources(1, cvr_new_height, 0));
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_height, &num_bytes, *cvr_new_height));

		checkCudaErrors(cudaGraphicsMapResources(1, cvr_new_water, 0));
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_water, &num_bytes, *cvr_new_water));

		checkCudaErrors(cudaGraphicsMapResources(1, cvr_new_sediment, 0));
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_sediment, &num_bytes, *cvr_new_sediment));

		checkCudaErrors(cudaGraphicsMapResources(1, cvr_height, 0));
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_new_height, &num_bytes, *cvr_height));

		checkCudaErrors(cudaGraphicsMapResources(1, cvr_water, 0));
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_new_water, &num_bytes, *cvr_water));

		checkCudaErrors(cudaGraphicsMapResources(1, cvr_sediment, 0));
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_new_sediment, &num_bytes, *cvr_sediment));

		state = true;
	}

	float *d_rain;
	checkCudaErrors(cudaGraphicsMapResources(1, cvr_rain, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_rain, &num_bytes, *cvr_rain));
	
	/*
	swap_maps(d_height, d_new_height);
	swap_maps(d_water, d_new_water);
	swap_maps(d_sediment, d_new_sediment);
	*/

	int numCells = mesh_dim * mesh_dim;

	// execute the kernel
	int gridSize = (mesh_dim + BLOCKDIM - 1) / BLOCKDIM;
	dim3 grid(gridSize, gridSize);
	dim3 block(BLOCKDIM, BLOCKDIM);
	
	add_rain <<<grid, block>>>(d_water, d_rain, mesh_dim);
	cudaThreadSynchronize();
	erode <<<grid, block>>>(d_height, d_water, d_sediment, d_new_height, d_new_water, d_new_sediment, numCells, mesh_dim);
	cudaThreadSynchronize();

	

	// unmap buffer object
	checkCudaErrors(cudaGraphicsUnmapResources(1, cvr_height, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, cvr_water, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, cvr_sediment, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, cvr_new_height, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, cvr_new_water, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, cvr_new_sediment, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, cvr_rain, 0));
}

__global__ void
erode2(float3 *pos, unsigned int width, unsigned int height)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	// calculate uv coordinates
	float u = x / (float)width;
	float v = y / (float)height;
	u = u * 2.0f - 1.0f;
	v = v * 2.0f - 1.0f;

	float w = pos[y*width + x].z;

	// write output vertex
	pos[y*width + x] = make_float3(u, v, w);
}



void
erodeCuda2(struct cudaGraphicsResource **vbo_resource, unsigned int mesh_width,
	unsigned int mesh_height) {

	// map OpenGL buffer object for writing from CUDA
	float3 *dptr;
	checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, *vbo_resource));

	// execute the kernel
	//launch_kernel(dptr, mesh_width, mesh_height, g_fAnim);
	dim3 block(8, 8, 1);
	dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
	erode2 <<<grid, block >>>(dptr, mesh_width, mesh_height);

	// unmap buffer object
	checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

void
printCudaInfo() {

	// for fun, just print out some stats on the machine

	int deviceCount = 0;
	cudaError_t err = cudaGetDeviceCount(&deviceCount);

	printf("---------------------------------------------------------\n");
	printf("Found %d CUDA devices\n", deviceCount);

	for (int i = 0; i<deviceCount; i++) {
		cudaDeviceProp deviceProps;
		cudaGetDeviceProperties(&deviceProps, i);
		printf("Device %d: %s\n", i, deviceProps.name);
		printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
		printf("   Global mem: %.0f MB\n",
			static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
		printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
	}
	printf("---------------------------------------------------------\n");
}


///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
__global__ void simple_vbo_kernel(float4 *pos, unsigned int width, unsigned int height, float time)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	// calculate uv coordinates
	float u = x / (float)width;
	float v = y / (float)height;
	u = u * 2.0f - 1.0f;
	v = v * 2.0f - 1.0f;

	// calculate simple sine wave pattern
	float freq = 4.0f;
	float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;

	// write output vertex
	pos[y*width + x] = make_float4(u, w, v, 1.0f);
}


////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(struct cudaGraphicsResource **vbo_resource, unsigned int mesh_width,
	unsigned int mesh_height, float g_fAnim)
{
	// map OpenGL buffer object for writing from CUDA
	float4 *dptr;
	checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, *vbo_resource));

	// execute the kernel
	//launch_kernel(dptr, mesh_width, mesh_height, g_fAnim);
	dim3 block(8, 8, 1);
	dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
	simple_vbo_kernel << <grid, block >> >(dptr, mesh_width, mesh_height, 0);

	// unmap buffer object
	checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}



