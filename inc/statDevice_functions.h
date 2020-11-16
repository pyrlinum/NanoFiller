#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include "simPrms.h"

// compute domain grid patch basepoint:
inline __device__ int set_grdPatchStart(int3 loc,int basePnt[3],float maxLength, float statGridScl[3], float dmnGridScl[3]) {
		basePnt[0] =(int) floor((loc.x*statGridScl[0]-maxLength)/dmnGridScl[0]);	// xlo
		basePnt[1] =(int) floor((loc.y*statGridScl[1]-maxLength)/dmnGridScl[1]);	// ylo
		basePnt[2] =(int) floor((loc.z*statGridScl[2]-maxLength)/dmnGridScl[2]);	// zlo

	return 0;
}
// compute current grid cell key:
inline __device__ int set_keyInPeriodic(int basePnt[3], int3 displ, int dmnGrdExt[3]) {

	int x[3];
	x[0] = basePnt[0] + displ.x;
	x[1] = basePnt[1] + displ.y;
	x[2] = basePnt[2] + displ.z;
#pragma unroll
	for(int i=0;i<3;i++){
		x[i] = (x[i]<0?x[i]+dmnGrdExt[i]:x[i]);
		x[i] = (x[i]>=dmnGrdExt[i]?x[i]%dmnGrdExt[i]:x[i]);
	}
	return x[0] + x[1]*dmnGrdExt[0] + x[2]*dmnGrdExt[0]*dmnGrdExt[1];
}
// load CNTs:
inline __device__ int loadCNT2shr(char numCoords, char *idxCoords, float *d_result, float *shr_arr, int st_addr, int load, int glb_patch) {
	if (threadIdx.x<load) {
		#pragma unrol
		for(int i=0;i<numCoords;i++)
			shr_arr[threadIdx.x+idxCoords[i]*blockDim.x] = d_result[st_addr+threadIdx.x+i*glb_patch];
	}
	__threadfence_block();
	__syncthreads();

	return 0;
}
// get (x,y,z) by blockNumber:
inline __device__ int3 locate3D(int blockID, int gridExt[3]){
	int3 loc;
	loc.x = blockID%gridExt[0];
	loc.y = (blockID/gridExt[0])%gridExt[1];
	loc.z = (blockID/gridExt[0])/gridExt[1];
	return loc;
}

inline __device__ int initRecord2shr(int vol,unsigned int *shr_arr){
	if(threadIdx.x < vol) {
		for(int i=0;i<warpSize;i++)	shr_arr[threadIdx.x*warpSize+i] = 0;
	}
	__threadfence_block();
	__syncthreads();
	return 0;
}

inline __device__ int initRecord2shr(int vol,unsigned int *cnts,float *avgs,float *disp,unsigned int *shr_arr){
	if(threadIdx.x < warpSize) {
		cnts[threadIdx.x]	= 0;
		avgs[threadIdx.x]	= 0.0f;
		disp[threadIdx.x]	= 0.0f;
	}
	if(threadIdx.x < vol) {
		for(int i=0;i<warpSize;i++)	shr_arr[threadIdx.x*warpSize+i] = 0;
	}
	__threadfence_block();
	__syncthreads();
	return 0;
}


inline __device__ int readRecord2shr(int blockID,int vol,unsigned int *shr_arr,int *d_arr){
	if(threadIdx.x < vol) {
		shr_arr[threadIdx.x] = d_arr[vol*blockID+threadIdx.x];
	}
	__threadfence_block();
	__syncthreads();
	return 0;
}

inline __device__ int writeRecord2dev(int blockID,int vol,unsigned int *shr_arr,unsigned int *d_arr,int stride){
	if(threadIdx.x < vol) {
		atomicAdd(&d_arr[vol*blockID+threadIdx.x], shr_arr[threadIdx.x*stride]);
	}
	__threadfence();
	__syncthreads();
	return 0;
}

inline __device__ int reductRecords(int vol,unsigned int *cnts,float *avgs,float *disp,unsigned int *shr_arr){

	for(unsigned int s=warpSize/2;s>0;s>>=1) {
		if (threadIdx.x<s) {
			cnts[threadIdx.x] += cnts[threadIdx.x+s];
			avgs[threadIdx.x] += avgs[threadIdx.x+s];
			disp[threadIdx.x] += disp[threadIdx.x+s];
		}
		__threadfence_block();
		__syncthreads();
	}

	if(threadIdx.x < vol) {
		for(unsigned int i=1;i<warpSize;i++)
		shr_arr[threadIdx.x*warpSize+0] += shr_arr[threadIdx.x*warpSize+i];
	}
	__threadfence();
	__syncthreads();
	return 0;
}

inline __device__ int reductOverWarp(int vol,unsigned int *shr_arr){

	if(threadIdx.x < vol) {
		for(unsigned int i=1;i<warpSize;i++)
			shr_arr[threadIdx.x*warpSize+0] += shr_arr[threadIdx.x*warpSize+i];
	}
	__threadfence();
	__syncthreads();
	return 0;
}
inline __device__ int reductOverWarp(int vol,int *shr_arr){

	if(threadIdx.x < vol) {
		for(unsigned int i=1;i<warpSize;i++)
			shr_arr[threadIdx.x*warpSize+0] += shr_arr[threadIdx.x*warpSize+i];
	}
	__threadfence();
	__syncthreads();
	return 0;
}
inline __device__ int reductOverWarp(int vol, volatile float *shr_arr){

	if(threadIdx.x < vol) {
		for(unsigned int i=1;i<warpSize;i++)
			shr_arr[threadIdx.x*warpSize+0] += shr_arr[threadIdx.x*warpSize+i];
	}
	__threadfence();
	__syncthreads();
	return 0;
}
inline __device__ int reductOverWarp(int vol, float *shr_arr){

	if(threadIdx.x < vol) {
		for(unsigned int i=1;i<warpSize;i++)
			shr_arr[threadIdx.x*warpSize+0] += shr_arr[threadIdx.x*warpSize+i];
	}
	__threadfence();
	__syncthreads();
	return 0;
}
inline __device__ int reductOverWarp(int vol,double *shr_arr){

	if(threadIdx.x < vol) {
		for(unsigned int i=1;i<warpSize;i++)
			shr_arr[threadIdx.x*warpSize+0] += shr_arr[threadIdx.x*warpSize+i];
	}
	__threadfence();
	__syncthreads();
	return 0;
}
//---------------------------------------------------------------------------------
inline __device__ int reduce(unsigned int size, volatile float *shr_arr){
	
	int tid = threadIdx.x;

	for(int s = size/2; s>32; s>>=1 ) {
		if(tid < s) shr_arr[tid]+=shr_arr[tid+s];
		__syncthreads();
	}
	// last iteration - warp-synchronous reduction:
	if (tid < 32)
		for(int s = 32; s>0; s>>=1 )
			shr_arr[tid]+=shr_arr[tid+s];
	__syncthreads();

	return 0;
}
