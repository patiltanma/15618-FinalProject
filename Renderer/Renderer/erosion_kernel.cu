

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

#define L_BLOCKDIM	34						//(BLOCKDIM+2)				// 34
#define L_BLOCKSIZE	1156					//L_BLOCKDIM*L_BLOCKDIM	// 34*34 = 1156

__global__ void
erode_2(float* a_height, float* a_water_vol, float* a_sediment, 
	float* a_new_height, float* a_new_water_vol, float* a_new_sediment, 
	float4* water_block_side_buffer, float4* sediment_block_side_buffer, int mesh_dim) {

	//TODO: lock down block sizes and dimensioning scheme
	int x = threadIdx.x + 1;
	int y = threadIdx.y + 1;
	int l_index = x + L_BLOCKDIM * y;				// local index

	int xx = blockDim.x*blockIdx.x + threadIdx.x;
	int yy = blockDim.y*blockIdx.y + threadIdx.y;
	int g_index = xx + mesh_dim * yy;			// global index

	// copy data to shared memory
	__shared__ float height[L_BLOCKSIZE];				// 1156 * 4 bytes
	__shared__ float water_vol[L_BLOCKSIZE];			// 1156 * 4 bytes
	__shared__ float sediment[L_BLOCKSIZE];				// 1156 * 4 bytes

														// create new shared memory to write back
	__shared__ float new_height[L_BLOCKSIZE];				// 1156 * 4 bytes
	__shared__ float new_water_vol[L_BLOCKSIZE];			// 1156 * 4 bytes
	__shared__ float new_sediment[L_BLOCKSIZE];				// 1156 * 4 bytes

	height[l_index] = a_height[g_index];
	water_vol[l_index] = a_water_vol[g_index];
	sediment[l_index] = a_sediment[g_index];

	// filling up and down cells for the block
	// possibly done in a single warp
	if (threadIdx.y == 0)
	{
		// upper cell is the limit of the mesh
		if (yy == 0)
		{	//basically clamping it to the value of edge
			height[x] = a_height[g_index];
			water_vol[x] = a_water_vol[g_index];
			sediment[x] = a_sediment[g_index];
		}
		else
		{	// get the respective value from the global memory
			height[x] = a_height[xx + mesh_dim * (yy - 1)];
			water_vol[x] = a_water_vol[xx + mesh_dim * (yy - 1)];
			sediment[x] = a_sediment[xx + mesh_dim * (yy - 1)];
		}
	}
	else if (threadIdx.y == BLOCKDIM - 1)
	{
		// lower cells add this to the block
		if (yy == mesh_dim - 1)
		{	//basically clamping it to the value of edge
			height[x + L_BLOCKDIM * (y + 1)] = a_height[g_index];
			water_vol[x + L_BLOCKDIM * (y + 1)] = a_water_vol[g_index];
			sediment[x + L_BLOCKDIM * (y + 1)] = a_sediment[g_index];
		}
		else
		{	// get the respective value from the global memory
			height[x + L_BLOCKDIM * (y + 1)] = a_height[xx + mesh_dim * (yy + 1)];
			water_vol[x + L_BLOCKDIM * (y + 1)] = a_water_vol[xx + mesh_dim * (yy + 1)];
			sediment[x + L_BLOCKDIM * (y + 1)] = a_sediment[xx + mesh_dim * (yy + 1)];
		}
	}

	// whole of the warp will be blocked! (cant think of optimizing this)
	if (threadIdx.x == 0)
	{
		// left cell is the limit of the mesh
		if (xx == 0)
		{	//basically clamping it to the value of edge
			height[L_BLOCKDIM * y] = a_height[g_index];
			water_vol[L_BLOCKDIM * y] = a_water_vol[g_index];
			sediment[L_BLOCKDIM * y] = a_sediment[g_index];
		}
		else
		{	// get the respective value from the global memory
			height[L_BLOCKDIM * y] = a_height[(xx - 1) + mesh_dim * yy];
			water_vol[L_BLOCKDIM * y] = a_water_vol[(xx - 1) + mesh_dim * yy];
			sediment[L_BLOCKDIM * y] = a_sediment[(xx - 1) + mesh_dim * yy];
		}
	}
	else if (threadIdx.x == BLOCKDIM - 1)
	{
		// left cell is the limit of the mesh
		if (xx == mesh_dim - 1)
		{	//basically clamping it to the value of edge
			height[L_BLOCKDIM - 1 + L_BLOCKDIM * y] = a_height[g_index];
			water_vol[L_BLOCKDIM - 1 + L_BLOCKDIM * y] = a_water_vol[g_index];
			sediment[L_BLOCKDIM - 1 + L_BLOCKDIM * y] = a_sediment[g_index];
		}
		else
		{	// get the respective value from the global memory
			height[L_BLOCKDIM - 1 + L_BLOCKDIM * y] = a_height[(xx + 1) + mesh_dim * yy];
			water_vol[L_BLOCKDIM - 1 + L_BLOCKDIM * y] = a_water_vol[(xx + 1) + mesh_dim * yy];
			sediment[L_BLOCKDIM - 1 + L_BLOCKDIM * y] = a_sediment[(xx + 1) + mesh_dim * yy];
		}
	}
	__syncthreads();

	int left = x - 1 + y * L_BLOCKDIM;
	int right = x + 1 + y * L_BLOCKDIM;
	int up = x + (y - 1) * L_BLOCKDIM;
	int down = x + (y + 1) * L_BLOCKDIM;

	// positive values indicate outward flow
	float4 water_vol_dir = make_float4(water_vol[left], water_vol[right], water_vol[up], water_vol[down]);
	float4 height_dir = make_float4(height[left], height[right], height[up], height[down]);
	float4 water_height_dir = height_dir + water_vol_dir;

	float cell_height = height[l_index];
	float cell_sediment = sediment[l_index];
	float cell_water_vol = water_vol[l_index];

	float cell_water_height = cell_height + cell_water_vol;
	float4 delta_h_dir = -(water_height_dir)+cell_water_height;

	//assume all water than can flow will AT CONSTANT SPEED
	float4 water_flow = clamp(delta_h_dir, 0.0, 1000.0) / 4;
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
	float sediment_capacity = falloff * cell_water_vol * SOLUBILITY; //might want to make this a max of flow or something

																	 //velocity*steepness = material eroded cos(atan(2/abs(dH.x+dH.y)))
	float cell_erosion = total_water_flux * ABRAISION; //could add additional effects of sediment on erosion here

	float new_cell_deposition = fmaxf(0, cell_sediment + cell_erosion - sediment_capacity);
	float new_cell_sediment = fminf(cell_sediment + cell_erosion, sediment_capacity);

	float4 sediment_flux = flux_fraction * water_flux_norm * new_cell_sediment;

	//account for frictional and water volume losses
	float new_cell_water_vol = (cell_water_vol - total_water_flux) * WATER_LOSS;

	new_height[l_index] = cell_height + new_cell_deposition - cell_erosion;
	new_water_vol[l_index] = new_cell_water_vol;
	new_sediment[l_index] = new_cell_sediment - sum(sediment_flux);
	__syncthreads();
	
	
	//the whole wrap will execute this is warp step, so i am guessing there should not be a problem.
	// and any way the bank conflicts will serialise it.
	
	/*
	new_water_vol[x - 1 + L_BLOCKDIM * y] += water_flux.x;
	new_water_vol[x + 1 + L_BLOCKDIM * y] += water_flux.y;
	new_water_vol[x + L_BLOCKDIM * (y + 1)] += water_flux.w;					////////////////////////////////take a look at this later
	new_water_vol[x + L_BLOCKDIM * (y - 1)] += water_flux.z;
	*/
	
	
	__shared__ float4 side_values[L_BLOCKSIZE];
	
	side_values[left].x = water_flux.x;
	side_values[right].y = water_flux.y;
	side_values[up].z = water_flux.z;
	side_values[down].w = water_flux.w;

	__syncthreads();
	new_water_vol[l_index] += sum(side_values[l_index]);
	__syncthreads();
	
	side_values[left].x = sediment_flux.x;
	side_values[right].y = sediment_flux.y;
	side_values[up].z = sediment_flux.z;
	side_values[down].w = sediment_flux.w;

	__syncthreads();
	new_water_vol[l_index] += sum(side_values[l_index]);
	__syncthreads();
	/*
	atomicAdd(&new_water_vol[left], water_flux.x);
	atomicAdd(&new_water_vol[right], water_flux.y);
	atomicAdd(&new_water_vol[up], water_flux.w);
	atomicAdd(&new_water_vol[down], water_flux.z);
	
	
	atomicAdd(&new_sediment[left], sediment_flux.x);
	atomicAdd(&new_sediment[right], sediment_flux.y);
	atomicAdd(&new_sediment[up], sediment_flux.w);
	atomicAdd(&new_sediment[down], sediment_flux.z);
	
	__syncthreads();
	*/
	
	a_new_height[g_index] = new_height[l_index];
	a_new_water_vol[g_index] = new_water_vol[l_index];
	a_new_sediment[g_index] = new_sediment[l_index];

	// thing to synchronize all the threads running this kernel!
	__syncthreads();


	int gridSize = (mesh_dim + BLOCKDIM - 1) / BLOCKDIM;
	int blocknum = blockIdx.x + gridSize * blockIdx.y;

	// whole of the warp will be blocked! (cant think of optimizing this)
	if (threadIdx.x == 0)
	{
		if (xx == 0)
		{
			atomicAdd(&a_new_water_vol[g_index], new_water_vol[left]);
			atomicAdd(&a_new_sediment[g_index], new_sediment[left]);
		}
		else
		{
			atomicAdd(&a_new_water_vol[xx - 1 + mesh_dim * yy], new_water_vol[left]);
			atomicAdd(&a_new_sediment[xx - 1 + mesh_dim * yy], new_sediment[left]);
		}
	}
	else if (threadIdx.x == BLOCKDIM - 1)
	{
		if (xx == mesh_dim - 1)
		{
			atomicAdd(&a_new_water_vol[g_index], new_water_vol[right]);
			atomicAdd(&a_new_sediment[g_index], new_sediment[right]);
		}
		else
		{
			atomicAdd(&a_new_water_vol[xx + 1 + mesh_dim * yy], new_water_vol[right]);
			atomicAdd(&a_new_sediment[xx + 1 + mesh_dim * yy], new_sediment[right]);
		}
	}

	if (threadIdx.y == 0)
	{
		if (yy == 0)
		{
			atomicAdd(&a_new_water_vol[g_index], new_water_vol[up]);
			atomicAdd(&a_new_sediment[g_index], new_sediment[up]);
		}
		else
		{
			atomicAdd(&a_new_water_vol[xx + mesh_dim * (yy - 1)], new_water_vol[up]);
			atomicAdd(&a_new_sediment[xx + mesh_dim * (yy - 1)], new_sediment[up]);
		}
		//water_block_side_buffer[blocknum].z = new_water_vol[up];
		//sediment_block_side_buffer[blocknum].z = new_sediment[up];
	}
	else if (threadIdx.y == BLOCKDIM - 1)
	{
		if (yy == mesh_dim - 1)
		{
			atomicAdd(&a_new_water_vol[g_index], new_water_vol[down]);
			atomicAdd(&a_new_sediment[g_index], new_sediment[down]);
		}
		else
		{
			atomicAdd(&a_new_water_vol[xx + mesh_dim * (yy + 1)], new_water_vol[down]);
			atomicAdd(&a_new_sediment[xx + mesh_dim * (yy + 1)], new_sediment[down]);
		}

		//water_block_side_buffer[blocknum].w = new_water_vol[down];
		//sediment_block_side_buffer[blocknum].w = new_sediment[down];
	}

	/*__syncthreads();*/

	/*debug_printdump();*/
	/*debug_compare_maps(total_map, new_total_map);*/
}


__global__ void
correct_overlap (float* d_new_height, float* d_new_water, float* d_new_sediment,
	float4* waterBlockSideBuffer, float4* sedimentBlockSideBuffer, int mesh_dim)
{
	int x, y;
	int gridSize = (mesh_dim + BLOCKDIM - 1) / BLOCKDIM;
	int blocknum = blockIdx.x + gridSize * blockIdx.y;

	if (threadIdx.y == 0) // thread calculating left
	{
		x = blockIdx.x + gridSize;
		y = threadIdx.y;
		int index = x + mesh_dim * y;
		if (x == 0)
		{
			//atomicAdd(&d_new_water[], waterBlockSideBuffer[blocknum].x);
		}
		else
		{
			//atomicAdd(&d_new_water[], waterBlockSideBuffer[blocknum].x);
		}
		
	}

}

__global__ void
erode(float* height, float* water_vol, float* sediment, float* new_height, float* new_water_vol, float* new_sediment, int numCells, int mesh_dim) {

	//TODO: lock down block sizes and dimensioning scheme

	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int index = x + mesh_dim * y;

	int left = max(0, x - 1) + y * mesh_dim;
	int right = min(x + 1, mesh_dim - 1) + y * mesh_dim;
	int up = x + max(0, y - 1) * mesh_dim;
	int down = x + min(y + 1, mesh_dim - 1) * mesh_dim;

	// positive values indicate outward flow
	float4 water_vol_dir = make_float4(water_vol[left], water_vol[right], water_vol[up], water_vol[down]);
	float4 height_dir = make_float4(height[left], height[right], height[up], height[down]);
	float4 water_height_dir = height_dir + water_vol_dir;

	float cell_height = height[index];
	float cell_water_vol = water_vol[index];
	float cell_sediment = sediment[index];

	float cell_water_height = cell_height + cell_water_vol;
	float4 delta_h_dir = -(water_height_dir) + cell_water_height;

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
	float sediment_capacity = falloff * cell_water_vol * SOLUBILITY; //might want to make this a max of flow or something

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
	__syncthreads();

	if (x>0) { // left neighbor
		atomicAdd(&new_water_vol[x - 1 + mesh_dim * y], water_flux.x);
		/*atomicAdd(&new_water_height[x-1+globalMapDim*y], water_flux.x);*/
		atomicAdd(&new_sediment[x - 1 + mesh_dim * y], sediment_flux.x);
	}
	__syncthreads();
	if (x<mesh_dim - 1) { //right neighbor
		atomicAdd(&new_water_vol[x + 1 + mesh_dim * y], water_flux.y);
		/*atomicAdd(&new_water_height[x+1+globalMapDim*y], water_flux.y);*/
		atomicAdd(&new_sediment[x + 1 + mesh_dim * y], sediment_flux.y);
	}
	__syncthreads();
	if (y<mesh_dim - 1) { //bottom neighbor
		atomicAdd(&new_water_vol[x + mesh_dim * (y + 1)], water_flux.w);
		/*atomicAdd(&new_ma->water_height[x+globalMapDim*(y+1)], water_flux.w);*/
		atomicAdd(&new_sediment[x + mesh_dim * (y + 1)], sediment_flux.w);
	}
	__syncthreads();
	if (y>0) { // top neighbor
		atomicAdd(&new_water_vol[x + mesh_dim * (y - 1)], water_flux.z);
		/*atomicAdd(&new_ma->water_height[x+globalMapDim*(y-1)], water_flux.z);*/
		atomicAdd(&new_sediment[x + mesh_dim * (y - 1)], sediment_flux.z);
	}


}


static bool state = true;

void 
erodeCuda_Approach1(struct cudaGraphicsResource **cvr_height, struct cudaGraphicsResource **cvr_water, struct cudaGraphicsResource **cvr_sediment,
	struct cudaGraphicsResource **cvr_new_height, struct cudaGraphicsResource **cvr_new_water, struct cudaGraphicsResource **cvr_new_sediment,
	struct cudaGraphicsResource **cvr_rain, int mesh_dim) {

	size_t num_bytes;
	
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

	int numCells = mesh_dim * mesh_dim;
	// execute the kernel
	int gridSize = (mesh_dim + BLOCKDIM - 1) / BLOCKDIM;
	dim3 grid(gridSize, gridSize);
	dim3 block(BLOCKDIM, BLOCKDIM);
	
	add_rain <<<grid, block>>>(d_water, d_rain, mesh_dim);
	cudaThreadSynchronize();

	erode <<<grid, block >>> (d_height, d_water, d_sediment, d_new_height, d_new_water, d_new_sediment, numCells, mesh_dim);
	cudaThreadSynchronize();

	/*
	cudaMemcpy((void *)d_height, (void *)d_new_height, numCells*sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy((void *)d_water, (void *)d_new_water, numCells * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy((void *)d_sediment, (void *)d_new_sediment, numCells * sizeof(float), cudaMemcpyDeviceToDevice);
	*/

	// unmap buffer object
	checkCudaErrors(cudaGraphicsUnmapResources(1, cvr_height, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, cvr_water, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, cvr_sediment, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, cvr_new_height, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, cvr_new_water, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, cvr_new_sediment, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, cvr_rain, 0));
}


void
erodeCuda_Approach2(struct cudaGraphicsResource **cvr_height, struct cudaGraphicsResource **cvr_water, struct cudaGraphicsResource **cvr_sediment,
	struct cudaGraphicsResource **cvr_new_height, struct cudaGraphicsResource **cvr_new_water, struct cudaGraphicsResource **cvr_new_sediment,
	struct cudaGraphicsResource **cvr_rain, int mesh_dim) {

	size_t num_bytes;

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

	int numCells = mesh_dim * mesh_dim;
	// execute the kernel
	int gridSize = (mesh_dim + BLOCKDIM - 1) / BLOCKDIM;
	dim3 grid(gridSize, gridSize);
	dim3 block(BLOCKDIM, BLOCKDIM);

	/*
	float4* waterBlockSideBuffer;
	cudaMalloc((void**)&waterBlockSideBuffer, gridSize * gridSize * sizeof(float4));

	float4* sedimentBlockSideBuffer;
	cudaMalloc((void**)&sedimentBlockSideBuffer, gridSize * gridSize * sizeof(float4));
	*/

	add_rain << <grid, block >> >(d_water, d_rain, mesh_dim);
	cudaThreadSynchronize();

	erode_2 <<<grid, block>>>(d_height, d_water, d_sediment,
	d_new_height, d_new_water, d_new_sediment,
	waterBlockSideBuffer, sedimentBlockSideBuffer, mesh_dim);
	cudaThreadSynchronize();
	

	/*
	dim3 grid2(gridSize, gridSize);
	dim3 block2(BLOCKDIM, 4);

	correct_overlap <<<grid2, block2>>>(d_new_height, d_new_water, d_new_sediment,
	waterBlockSideBuffer, sedimentBlockSideBuffer, mesh_dim);
	cudaThreadSynchronize();
	*/

	//cudaFree(waterBlockSideBuffer);
	//cudaFree(sedimentBlockSideBuffer);

	// unmap buffer object
	checkCudaErrors(cudaGraphicsUnmapResources(1, cvr_height, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, cvr_water, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, cvr_sediment, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, cvr_new_height, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, cvr_new_water, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, cvr_new_sediment, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, cvr_rain, 0));
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

