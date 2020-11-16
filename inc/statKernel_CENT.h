#pragma once
#include "statDevice_functions.h"

#define numCrds	3

// Kernel to compute center to center distance statistics
// Should be run several times to collect also statistics in between domain cells
__global__ void statKernel_CENT(	int		row,
									int3	displ,
									int		*d_dmnAddr,
									short	*d_dmnOcc,
									float	*d_CNTres,
									unsigned int *d_recrods,
									float	*d_avgVal,
									float	*d_disp,
									unsigned int *d_count)
{	
	__shared__ int		grdPatchStart[3];	// container to keep base point of patch to check for contribution
	__shared__ int		startAddr_self;		//	starting adress to read
	__shared__ int		startAddr;			//	starting adress to read
	__shared__ int		key;				//	index of block to load into shared memory
	__shared__ int		CNTload_self;		//	number of CNTs in current cell to check
	__shared__ int		CNTload;			//	number of CNTs in current cell to check
	__shared__ float	shrArr[3*BLOCK];	//	cnt-data
	//results:
	extern __shared__ unsigned int shr_arr[];
	unsigned int	*shr_count	=	(unsigned int*)	&(shr_arr[0*warpSize]);	// accumulates count of intersections in range
	float			*shr_avg	=	(float*)		&(shr_arr[1*warpSize]);	// accumulates sum of X for averages
	float			*shr_disp	=	(float*)		&(shr_arr[2*warpSize]);	// accumulates sum of X^2 for dispersions
	unsigned int	*shrRecord	=	(unsigned int*)	&(shr_arr[3*warpSize]);	// accumulates histogram data


	int	blockID = row*gridDim.x+blockIdx.x; 

	if (blockID < grdExt_stat[0]*grdExt_stat[1]*grdExt_stat[2]) {

		char idxCrds[numCrds] = { 0, 1, 2 };		// CNT data to be loaded
		
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
		// load to shared memory:
		int3 pos3D;
		float maxLen;
		if (threadIdx.x==0) {
			maxLen = binInterval*(numBins-1);
			pos3D = locate3D(blockID,grdExt_stat);
			set_grdPatchStart( pos3D, grdPatchStart, maxLen, statGridStep, grdStep);
			key = set_keyInPeriodic(grdPatchStart,displ,grdExt_stat);
			startAddr = (key!=0?d_dmnAddr[key-1]:0);
			CNTload = d_dmnOcc[key];
		}
		__threadfence_block();
		__syncthreads();
		loadCNT2shr(numCrds,idxCrds,d_CNTres,shrArr,startAddr,CNTload,glbPtchSz_stat);
	
		//initialize counters:
		//if (threadIdx.x==0) shr_count = 0; 
		initRecord2shr(numBins,shr_count,shr_avg,shr_disp,shrRecord);
		
		int colomn = threadIdx.x%warpSize;	
		float dist = 0.0f;
		int binId = 0;

		//collect statistical data:
		if (threadIdx.x<CNTload_self) {
		if (key == blockID)	{

				for(int i=threadIdx.x+1;i<CNTload;i++) {
					dist =	sqrtf(	(probe.x-shrArr[i+0*blockDim.x])*(probe.x-shrArr[i+0*blockDim.x])+
									(probe.y-shrArr[i+1*blockDim.x])*(probe.y-shrArr[i+1*blockDim.x])+
									(probe.z-shrArr[i+2*blockDim.x])*(probe.z-shrArr[i+2*blockDim.x])	);

					binId = (int) floor(dist/binInterval);
					if ((binId < numBins)&&(binId >= 0)) {
						atomicInc(&shr_count[colomn],blockDim.x*blockDim.x);
						atomicAdd(&shr_avg[colomn],dist);
						atomicAdd(&shr_disp[colomn],dist*dist);
						atomicInc(&shrRecord[colomn+binId*warpSize],glbPtchSz_stat);
					}		
				}
		} else {
				for(int i=0;i<CNTload;i++) {
					dist =	sqrtf(	(probe.x-shrArr[i+0*blockDim.x])*(probe.x-shrArr[i+0*blockDim.x])+
									(probe.y-shrArr[i+1*blockDim.x])*(probe.y-shrArr[i+1*blockDim.x])+
									(probe.z-shrArr[i+2*blockDim.x])*(probe.z-shrArr[i+2*blockDim.x])	);

					binId = (int) floor(dist/binInterval);
					if ((binId < numBins)&&(binId >= 0)) {
						atomicInc(&shr_count[colomn],blockDim.x*blockDim.x);
						atomicAdd(&shr_avg[colomn],dist);
						atomicAdd(&shr_disp[colomn],dist*dist);
						atomicInc(&shrRecord[colomn+binId*warpSize],glbPtchSz_stat);
					}		
				}
				
		}}
		__syncthreads();

		reductRecords(numBins,shr_count,shr_avg,shr_disp,shrRecord);
		writeRecord2dev(blockID,numBins,shrRecord,d_recrods,warpSize);
		if (threadIdx.x==0) {
			d_avgVal[blockID] += shr_avg[0];
			d_disp[blockID] +=  shr_disp[0];
			d_count[blockID] += shr_count[0];
		}

	}

}