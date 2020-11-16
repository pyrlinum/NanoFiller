#pragma once
#include <cuda_runtime.h>
#include <stdio.h>

// Initialize arrays:
template<typename T>
__global__ void set_arr(T val,unsigned int size, T *d_arr){
	unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;
	if (tid<size) d_arr[tid] = val;
}
template<typename T>
T* cuda_init(T val, unsigned int size) {
	T* data_ptr;
	cudaError_t error = cudaMalloc( &data_ptr, size*sizeof(T));
	if (error != cudaSuccess) {
		printf("ERROR allocating memory on device: %s\n",cudaGetErrorString(error));
	} else {
		int		currentDev;		 cudaGetDevice(&currentDev);
		cudaDeviceProp	devProp; cudaGetDeviceProperties(&devProp,currentDev);
		int block_size = devProp.maxThreadsPerBlock;

		dim3 grid((int) ceil(size*1./block_size));
		dim3 block(block_size);
		set_arr<T><<<grid,block>>>(val,size, data_ptr);
		error = cudaGetLastError();
		if ( error != cudaSuccess ) { printf("ERROR initialising array on device to %d: %s\n",val,cudaGetErrorString(error)); }
	}
	return data_ptr;
}
