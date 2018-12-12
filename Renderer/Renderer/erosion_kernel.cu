

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


__device__ void
get_neighbors(cellData* map, cellData* neighbors, int mapDim) {

	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;

	//currently other parts of the program stop updates outside of the 
	//absolute edges of the map
	int left = max(0, x - 1);
	int right = min(x + 1, mapDim - 1);
	int up = max(0, y - 1);
	int down = min(y + 1, mapDim - 1);

	neighbors[0] = map[left + y * mapDim];
	neighbors[1] = map[right + y * mapDim];
	neighbors[2] = map[x + up * mapDim];
	neighbors[3] = map[x + down * mapDim];
}


__global__ void add_rain(cellData* total_map, float* rainMap, int globalMapDim) {

	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int index = x + globalMapDim * y;

	total_map[index].water_vol += rainMap[index];
	total_map[index].water_height += rainMap[index];
	/*cellData* total_map = d_total_map;*/

}


__global__ void
erode(cellData* total_map, cellData* new_total_map, int numCells, int globalMapDim) {

	//TODO: lock down block sizes and dimensioning scheme

	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int index = x + globalMapDim * y;

	/*cellData* total_map = d_total_map;*/
	/*cellData* new_total_map = d_new_total_map;*/
	clear_dest_map(new_total_map, numCells);


	cellData* cell = &total_map[index];
	cellData* newcell = &new_total_map[index];

	memcpy(newcell, cell, sizeof(cellData));

	__syncthreads();

	cellData  neighbors[4];
	get_neighbors(total_map, neighbors, globalMapDim);

	cellData left = neighbors[0];
	cellData right = neighbors[1];
	cellData up = neighbors[2];
	cellData down = neighbors[3];

	/* do actual erosion updates for given timestep.
	this happens first because it simplifies propogating
	sediment to other cells and it's happening in a cycle
	so order doesn't matter much */

	//account for frictional and water volume losses
	/*cell->water_vol= fmax(0, cell->water_vol- WATER_LOSS);*/

	/*if (cell->water_vol> 0)*/
	/*debug_print_cell(x,y,cell);*/

	// positive values indicate outward flow
	float4 height_dir = make_float4(left.height, right.height, up.height, down.height);
	float4 water_height_dir = make_float4(left.water_height, right.water_height, up.water_height, down.water_height);

	float4 delta_h_dir = -water_height_dir + cell->water_height;

	//this is velocity because assuming unit distance between cells
	float4 cell_v = VEL_LOSS * make_float4(cell->vel.x, -cell->vel.x, cell->vel.y, -cell->vel.y);
	/*float4 v_dir = cell_v+G/4;*/
	float4 v_dir = make_float4(G / 4.0);
	float vel_total = sum(v_dir);

	float4 water_flux = clamp(delta_h_dir*v_dir, 0.0, 1000);
	float flux_total = sum(water_flux);

	//dump sediment
	float falloff = (DEEP_WATER - min(DEEP_WATER, cell->water_vol)) / DEEP_WATER; //maintain thin water assumption
	float sediment_capacity = falloff * cell->water_vol*flux_total*SOLUBILITY; //might want to make this a max of flow or something

																			   //velocity*steepness = material eroded cos(atan(2/abs(dH.x+dH.y)))
	float cell_erosion = cell->water_vol*flux_total*ABRAISION; //could add additional effects of sediment on erosion here

	float new_deposition = fmaxf(0, cell->sediment + cell_erosion - sediment_capacity);
	float new_sediment = fminf(cell->sediment + cell_erosion, sediment_capacity);

	/*cell->height += new_deposition;*/
	/*cell->sediment -= new_deposition;*/

	/*propogate stuff to von neumann neighbors. current algorithm assumes that water
	doesn't pile up, but will transfer velocity to adjacent cells regardless */

	float total_water_flux = sum(water_flux);

	float4 water_flux_norm = make_float4(0.0);
	if (total_water_flux>0)
		water_flux_norm = water_flux / total_water_flux;

	if (total_water_flux > cell->water_vol) {
		water_flux = water_flux_norm * cell->water_vol;
		total_water_flux = cell->water_vol;
	}

	float4 sediment_flux = water_flux_norm * new_sediment;

	float new_water_vol = (newcell->water_vol - total_water_flux)*WATER_LOSS;

	//in theory water moving into a new cell only changes the direction of flow proportional to its mass or something
	float4 vel_update = water_flux_norm * v_dir;

	/*cell->water_vol -= total_water_flux;*/
	/*cell->sediment -= sum(sediment_movement_dir);*/
	/*cell->vel = make_float4(0.0);*/
	/*float old_water_value = newcell.water_vol; */
	/*newcell->water_height = cell->height + cell->water_vol;*/

	/*float oldheight = newcell->height;*/


	atomicAdd(&newcell->height, new_deposition - cell_erosion);
	atomicExch(&newcell->sediment, new_sediment - sum(sediment_flux));
	atomicExch(&newcell->water_vol, new_water_vol);
	/*newcell->water_vol = max(newcell->water_vol*WATER_LOSS-total_water_flux, 0.0);*/
	newcell->vel = make_float2(0.0); //read current velocity instead of trying to account for stuff or something
	atomicExch(&newcell->water_height, newcell->height + new_water_vol);

	/*float newheight = newcell->height;*/

	__syncthreads();


	if (x>0) { // left neighbor
		atomicAdd(&new_total_map[x - 1 + globalMapDim * y].water_vol, water_flux.x);
		atomicAdd(&new_total_map[x - 1 + globalMapDim * y].water_height, water_flux.x);
		atomicAdd(&new_total_map[x - 1 + globalMapDim * y].vel.x, vel_update.x);
		atomicAdd(&new_total_map[x - 1 + globalMapDim * y].sediment, sediment_flux.x);
	}
	if (x<globalMapDim - 1) { //right neighbor
		atomicAdd(&new_total_map[x + 1 + globalMapDim * y].water_vol, water_flux.y);
		atomicAdd(&new_total_map[x + 1 + globalMapDim * y].water_height, water_flux.y);
		atomicAdd(&new_total_map[x + 1 + globalMapDim * y].vel.x, -vel_update.y);
		atomicAdd(&new_total_map[x + 1 + globalMapDim * y].sediment, sediment_flux.y);
	}
	if (y<globalMapDim - 1) { //top neighbor
		atomicAdd(&new_total_map[x + globalMapDim * (y + 1)].water_vol, water_flux.w);
		atomicAdd(&new_total_map[x + globalMapDim * (y + 1)].water_height, water_flux.w);
		atomicAdd(&new_total_map[x + globalMapDim * (y + 1)].vel.y, vel_update.w);
		atomicAdd(&new_total_map[x + globalMapDim * (y + 1)].sediment, sediment_flux.w);
	}
	if (y>0) { // bottom neighbor
		atomicAdd(&new_total_map[x + globalMapDim * (y - 1)].water_vol, water_flux.z);
		atomicAdd(&new_total_map[x + globalMapDim * (y - 1)].water_height, water_flux.z);
		atomicAdd(&new_total_map[x + globalMapDim * (y - 1)].vel.y, -vel_update.z);
		atomicAdd(&new_total_map[x + globalMapDim * (y - 1)].sediment, sediment_flux.z);
	}

	__syncthreads();

	/*debug_printdump();*/
	/*debug_compare_maps(total_map, new_total_map);*/

	//swap pointers at end of iteration
	/*cellData* temp = new_total_map;*/
	/**d_new_total_map = total_map;*/
	/**d_total_map = temp;*/
}


void
erodeCuda(struct cudaGraphicsResource **vbo_resource_map,
	struct cudaGraphicsResource **vbo_resource_new_map,
	struct cudaGraphicsResource **vbo_resource_rain_map,
	unsigned int mesh_width, unsigned int mesh_height) {

	size_t num_bytes;

	// map OpenGL buffer object for writing from CUDA
	cellData *dptr_map;
	checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource_map, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr_map, &num_bytes, *vbo_resource_map));

	cellData *dptr_new_map;
	checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource_new_map, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr_new_map, &num_bytes, *vbo_resource_new_map));

	float *dptr_rain_map;
	checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource_rain_map, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr_rain_map, &num_bytes, *vbo_resource_rain_map));

	// execute the kernel
	//launch_kernel(dptr, mesh_width, mesh_height, g_fAnim);
	dim3 block(BLOCKDIM, BLOCKDIM, 1);
	dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);

	int numCells = mesh_width * mesh_height;
	int map_dim = mesh_width;

	add_rain <<<grid, block>>>(dptr_map, dptr_rain_map, map_dim);
	cudaThreadSynchronize();
	erode <<<grid, block>>>(dptr_map, dptr_new_map, numCells, map_dim);
	cudaThreadSynchronize();

	//erode <<< grid, block >>>(dptr_map, dptr_new_map, dptr_rain_map, mesh_width, mesh_height);

	// unmap buffer object
	checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource_map, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource_new_map, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource_rain_map, 0));
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
	erode2 << <grid, block >> >(dptr, mesh_width, mesh_height);

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



