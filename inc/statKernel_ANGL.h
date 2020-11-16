#pragma once
#include "statDevice_functions.h"

#define PREC 0.001f

__constant__ int	numBins2D[2];			// number of bins to bild histogrtams
__constant__ float	dPhi;					// step for phi angle
__constant__ float	dTheta;					// step for theta angle

#define numCrds	3
// Kernel to compute phi-theta distribution function:
// Should be run once - uses only single shared histogram and atomicInc
__global__ void statKernel_ANGL(	char	row,
									int		*d_dmnAddr,
									short	*d_dmnOcc,
									float	*d_CNTres,
									unsigned int *d_recrods)
{	
	__shared__ int		startAddr_self;			//	starting adress to read
	__shared__ int		CNTload_self;			//	number of CNTs in current cell to check
	//results:
	extern __shared__ unsigned int shrRecord[];	// accumulates histogram data	

	int	blockID = row*gridDim.x+blockIdx.x;		// for grids too big for singe run

	if (blockID < grdExt_stat[0]*grdExt_stat[1]*grdExt_stat[2]) {
		
		char idxCrds[numCrds] = { 3, 4, 5 };	// CNT data to be loaded
		
		// load probes:
		if (threadIdx.x==0)	{
			startAddr_self = (blockID!=0?d_dmnAddr[blockID-1]:0);
			__threadfence_block();
			CNTload_self = d_dmnOcc[blockID];
		}
		__threadfence_block();
		__syncthreads();
		
		float3	probe;
		if ( threadIdx.x < CNTload_self) {
				probe.x = d_CNTres[startAddr_self+threadIdx.x+idxCrds[0]*glbPtchSz_stat];
				probe.y = d_CNTres[startAddr_self+threadIdx.x+idxCrds[1]*glbPtchSz_stat];
				probe.z = d_CNTres[startAddr_self+threadIdx.x+idxCrds[2]*glbPtchSz_stat];
		}
		__threadfence_block();
		__syncthreads();
	
		//initialize counters:
		
		int vol = numBins2D[0]*numBins2D[1];
		for(char idx = 0; idx*blockDim.x<vol; idx++)
			if (threadIdx.x+idx*blockDim.x<vol)
				shrRecord[threadIdx.x+idx*blockDim.x] = 0;
		__threadfence_block();
		__syncthreads();

		//collect statistical data:
		if (threadIdx.x<CNTload_self) {

			float theta = acosf(probe.z);
			float phi = (( (theta > PREC)&&(theta < acosf(-1)-PREC) ) ? acosf(probe.x*rsqrtf(1-probe.z*probe.z)) : 0.0f);

			int binPhi		= (int) floor(phi/dPhi);
			int binTheta	= (int) floor(theta/dTheta);
			atomicInc(&shrRecord[binPhi+binTheta*numBins2D[0]],glbPtchSz_stat);
		}
		__syncthreads(); 
		

		for(char idx = 0; idx*blockDim.x<vol; idx++)
			if(threadIdx.x+idx*blockDim.x < vol) 
				d_recrods[vol*blockID+threadIdx.x+idx*blockDim.x] = shrRecord[threadIdx.x+idx*blockDim.x];

	}

}