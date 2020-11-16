#pragma once
#include <cuda_runtime.h>
#include "definitions.h"

inline __device__ void comparator(int A,int B,char dir,float *shrArr);
inline __device__ void swap(int A,int B,float *shrArr);
// bitonic sort:
inline __device__ void bitonicSort(float *shrArr,unsigned int dir) {
		for(unsigned int size=2;size<blockDim.x;size*=2) {
			unsigned int ddd = dir ^ ((threadIdx.x&(size/2))!=0);
			for(unsigned int stride = size/2;stride>0;stride/=2) {
				__syncthreads();
				unsigned int pos = 2*threadIdx.x-(threadIdx.x&(stride-1));
				if (threadIdx.x<(blockDim.x/2))
					comparator(pos,pos+stride,ddd,shrArr); 
			}
		}
		for(unsigned int stride = blockDim.x/2;stride>0;stride/=2) {
				__syncthreads();
				unsigned int pos = 2*threadIdx.x-(threadIdx.x&(stride-1));
				if (threadIdx.x<(blockDim.x/2))
					comparator(pos,pos+stride,dir,shrArr);
		}
		__syncthreads();
}

// Auxilary:
inline __device__ void comparator(int A,int B,char dir,float *shrArr) {
	if ((shrArr[A + 8*blockDim.x] > shrArr[B + 8*blockDim.x])==dir)
		swap(A,B,shrArr);
}

inline __device__ void swap(int A,int B,float *shrArr) {
	float reg;
	#pragma unroll
	for (int i=0;i<9;i++) {
		reg = shrArr[A + i*blockDim.x];
		shrArr[A + i*blockDim.x] = shrArr[B + i*blockDim.x];
		shrArr[B + i*blockDim.x] = reg;
	}
}


// prefix scan:
#define lnb	5		// log(Number of Banks) = 5 for Fermi
inline __device__ void pscan(int num, int *sum, float *keyArr, int *posArr) {
// num must be a power of 2
	__shared__ int temp[BLOCK+(BLOCK>>lnb)];
	//__shared__ int temp[BLOCK+BLOCK>>lnb];

	int tid = threadIdx.x;
	int offset = 1;

	__syncthreads();

	if (tid < num/2) {
		int ai = tid;
		int bi = (tid + num/2);

		temp[ai + (ai>>lnb)] = (keyArr[ai]>=0);
		temp[bi + (bi>>lnb)] = (keyArr[bi]>=0);
	}

	// build sum tree:
	for (int d=num/2;d>0;d>>=1) {
		__syncthreads();
		if (tid < d) {
			int ai = offset*(2*tid+1) - 1;
			int bi = offset*(2*tid+2) - 1;
			ai += ai>>lnb;
			bi += bi>>lnb;
			temp[bi] +=temp[ai]; 
		}
		offset <<= 1;
	}
	// get result from sum tree:
	if (tid == 0) {
		int last = num-1 + ((num-1)>>lnb);
		*sum = temp[last];
		temp[last] = 0;
	}

	for (int d=1;d<num;d<<=1) {
		offset >>= 1;
		__syncthreads();
		if (tid < d) {
			int ai = offset*(2*tid+1) - 1;
			int bi = offset*(2*tid+2) - 1;
			ai += ai>>lnb;
			bi += bi>>lnb;
			int buff  = temp[ai];
			temp[ai]  = temp[bi];
			temp[bi] += buff;
		}
	}
	__syncthreads();

	if (tid < num)
		posArr[tid] = temp[tid+(tid>>lnb)];
	__syncthreads();

}
inline __device__ void pscanNoB(int num, int *sum, float *keyArr, int *posArr) {
// num must be a power of 2
	__shared__ int temp[BLOCK];

	int tid = threadIdx.x;
	int offset = 1;

	__syncthreads();

	if (tid < num/2) {
		int ai = tid;
		int bi = (tid + num/2);

		temp[ai] = (keyArr[ai]>=0);
		temp[bi] = (keyArr[bi]>=0);
	}

	// build sum tree:
	for (int d=num/2;d>0;d>>=1) {
		__syncthreads();
		if (tid < d) {
			int ai = offset*(2*tid+1) - 1;
			int bi = offset*(2*tid+2) - 1;
			temp[bi] +=temp[ai]; 
		}
		offset <<= 1;
	}
	// get result from sum tree:
	if (tid == 0) {
		int last = num-1;
		*sum = temp[last];
		temp[last] = 0;
	}

	for (int d=1;d<num;d<<=1) {
		offset >>= 1;
		__syncthreads();
		if (tid < d) {
			int ai = offset*(2*tid+1) - 1;
			int bi = offset*(2*tid+2) - 1;
			int buff  = temp[ai];
			temp[ai]  = temp[bi];
			temp[bi] += buff;
		}
	}
	__syncthreads();

	if (tid < num)
		posArr[tid] = temp[tid];
	__syncthreads();

}
