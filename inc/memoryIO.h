#pragma once
#include <cstdio>
#include <cuda_runtime.h>
#include "definitions.h"
#include "sort.h"
#include "CNT.h"


//--------------------------------------------------------------------------------------------------------------------
// Address conversion:
inline __device__ int stride2glbID(char3 stride, int baseID, int *dmnExt) { 

	int x1 = (baseID%dmnExt[0])		+ stride.x;
	int y1 = (baseID/dmnExt[0])%dmnExt[1]	+ stride.y;
	int z1 = (baseID/dmnExt[0])/dmnExt[1]	+ stride.z;

	x1 = (x1>=0?x1:dmnExt[0]+x1);	x1 = (x1<dmnExt[0]?x1:x1-dmnExt[0]);
	y1 = (y1>=0?y1:dmnExt[1]+y1);	y1 = (y1<dmnExt[1]?y1:y1-dmnExt[1]);
	z1 = (z1>=0?z1:dmnExt[2]+z1);	z1 = (z1<dmnExt[2]?z1:z1-dmnExt[2]);

	return x1+y1*dmnExt[0]+z1*dmnExt[0]*dmnExt[1];
}
inline __device__ int stride2glbID_directed(char dir, char3 stride, int baseID, int *dmnExt) { 

	int x1 = (baseID%dmnExt[0])				+ stride.x;
	int y1 = (baseID/dmnExt[0])%dmnExt[1]	+ stride.y;
	int z1 = (baseID/dmnExt[0])/dmnExt[1]	+ stride.z;

	// check direction: x - dir & 1  y - dir & 2 z - dir & 4
	x1 = ( x1>=0		? x1 : (dir & 1 ? dmnExt[0]+x1 : -1));
	x1 = ( x1<dmnExt[0] ? x1 : (dir & 1 ? x1-dmnExt[0] : -1));
	y1 = ( y1>=0		? y1 : (dir & 2 ? dmnExt[1]+y1 : -1));
	y1 = ( y1<dmnExt[1]	? y1 : (dir & 2 ? y1-dmnExt[1] : -1));
	z1 = ( z1>=0		? z1 : (dir & 4 ? dmnExt[2]+z1 : -1));
	z1 = ( z1<dmnExt[2] ? z1 : (dir & 4 ? z1-dmnExt[2] : -1));

	return ((x1>=0)&&(y1>=0)&&(z1>=0)) ? x1+y1*dmnExt[0]+z1*dmnExt[0]*dmnExt[1] : -1;
}

inline __device__ int stride2glbID(char3 stride, unsigned int baseID, int *dmnExt) {

	int x1 = (baseID%dmnExt[0])		+ stride.x;
	int y1 = (baseID/dmnExt[0])%dmnExt[1]	+ stride.y;
	int z1 = (baseID/dmnExt[0])/dmnExt[1]	+ stride.z;

	x1 = (x1>=0?x1:dmnExt[0]+x1);	x1 = (x1<dmnExt[0]?x1:x1-dmnExt[0]);
	y1 = (y1>=0?y1:dmnExt[1]+y1);	y1 = (y1<dmnExt[1]?y1:y1-dmnExt[1]);
	z1 = (z1>=0?z1:dmnExt[2]+z1);	z1 = (z1<dmnExt[2]?z1:z1-dmnExt[2]);

	return x1+y1*dmnExt[0]+z1*dmnExt[0]*dmnExt[1];
}

inline __device__ int stride2glbID_directed(char dir, char3 stride, unsigned int baseID, int *dmnExt) {

	int x1 = (baseID%dmnExt[0])				+ stride.x;
	int y1 = (baseID/dmnExt[0])%dmnExt[1]	+ stride.y;
	int z1 = (baseID/dmnExt[0])/dmnExt[1]	+ stride.z;

	// check direction: x - dir & 1  y - dir & 2 z - dir & 4
	x1 = ( x1>=0		? x1 : (dir & 1 ? dmnExt[0]+x1 : -1));
	x1 = ( x1<dmnExt[0] ? x1 : (dir & 1 ? x1-dmnExt[0] : -1));
	y1 = ( y1>=0		? y1 : (dir & 2 ? dmnExt[1]+y1 : -1));
	y1 = ( y1<dmnExt[1]	? y1 : (dir & 2 ? y1-dmnExt[1] : -1));
	z1 = ( z1>=0		? z1 : (dir & 4 ? dmnExt[2]+z1 : -1));
	z1 = ( z1<dmnExt[2] ? z1 : (dir & 4 ? z1-dmnExt[2] : -1));

	return ((x1>=0)&&(y1>=0)&&(z1>=0)) ? x1+y1*dmnExt[0]+z1*dmnExt[0]*dmnExt[1] : -1;
}

inline __device__ unsigned int stride2glbID_directed_uint(char dir, char3 stride, unsigned int baseID, int *dmnExt) {

	int x1 = (baseID%dmnExt[0])				+ stride.x;
	int y1 = (baseID/dmnExt[0])%dmnExt[1]	+ stride.y;
	int z1 = (baseID/dmnExt[0])/dmnExt[1]	+ stride.z;

	// check direction: x - dir & 1  y - dir & 2 z - dir & 4
	x1 = ( x1>=0		? x1 : (dir & 1 ? dmnExt[0]+x1 : -1));
	x1 = ( x1<dmnExt[0] ? x1 : (dir & 1 ? x1-dmnExt[0] : -1));
	y1 = ( y1>=0		? y1 : (dir & 2 ? dmnExt[1]+y1 : -1));
	y1 = ( y1<dmnExt[1]	? y1 : (dir & 2 ? y1-dmnExt[1] : -1));
	z1 = ( z1>=0		? z1 : (dir & 4 ? dmnExt[2]+z1 : -1));
	z1 = ( z1<dmnExt[2] ? z1 : (dir & 4 ? z1-dmnExt[2] : -1));

	return ((x1>=0)&&(y1>=0)&&(z1>=0)) ? x1+y1*dmnExt[0]+z1*dmnExt[0]*dmnExt[1] : UINT_MAX;
}

inline __device__ int key2glbID(int key, int *mask, int *dmnExt) {

	int x1 = (key%mask[3])		+ mask[0];
	int y1 = (key/mask[3])%mask[4]	+ mask[1];
	int z1 = (key/mask[3])/mask[4]	+ mask[2];
	return x1+y1*dmnExt[0]+z1*dmnExt[0]*dmnExt[1];
}
inline __device__ int glbID2key(int glbID, int *mask, int *dmnExt) {

	int x1 = (glbID%dmnExt[0])		- mask[0];
	int y1 = (glbID/dmnExt[0])%dmnExt[1]	- mask[1];
	int z1 = (glbID/dmnExt[0])/dmnExt[1]	- mask[2];
	return x1+y1*mask[3]+z1*mask[3]*mask[4];
}
inline __device__ bool isInCell(float x, float y, float z, int key, int *mask, int *dmnExt, float *phsDim, float epsilon) {
      
	int x0 = (key%mask[3])		+ mask[0];
	int y0 = (key/mask[3])%mask[4]	+ mask[1];
	int z0 = (key/mask[3])/mask[4]	+ mask[2];
	
	bool isX = ((x0+epsilon)*phsDim[0] <= x)&&(x < (x0+1-epsilon)*phsDim[0]);
	bool isY = ((y0+epsilon)*phsDim[1] <= y)&&(y < (y0+1-epsilon)*phsDim[1]);
	bool isZ = ((z0+epsilon)*phsDim[2] <= z)&&(z < (z0+1-epsilon)*phsDim[2]);
	
	return isX && isY && isZ;
}
//--------------------------------------------------------------------------------------------------------------------
inline __device__ int getMask(int *mask, int *d_masks) {

		mask[0] = d_masks[ blockIdx.x + 0 * gridDim.x];
		mask[1] = d_masks[ blockIdx.x + 2 * gridDim.x];
		mask[2] = d_masks[ blockIdx.x + 4 * gridDim.x];

		mask[3] = d_masks[ blockIdx.x + 1 * gridDim.x]+1 - mask[0];
		mask[4] = d_masks[ blockIdx.x + 3 * gridDim.x]+1 - mask[1];
		mask[5] = d_masks[ blockIdx.x + 5 * gridDim.x]+1 - mask[2]; 

		return mask[3]*mask[4]*mask[5];
}

inline __device__ int glbID2mask(int tid, int *mask, int *dmnExt) {

		mask[0] = (tid%dmnExt[0]);
		mask[1] = (tid/dmnExt[0])%dmnExt[1];
		mask[2] = (tid/dmnExt[0])/dmnExt[1];

		mask[3] = 1;
		mask[4] = 1;
		mask[5] = 1; 

		return 1;
}
//--------------------------------------------------------------------------------------------------------------------
inline __device__ CNT_t shrID2regCNT(int shrID,float *shrArr) {

	CNT_t probe;
		
		probe.r.x = shrArr[shrID + 0*blockDim.x];
		probe.r.y = shrArr[shrID + 1*blockDim.x];
		probe.r.z = shrArr[shrID + 2*blockDim.x];

		probe.c.x = shrArr[shrID + 3*blockDim.x];
		probe.c.y = shrArr[shrID + 4*blockDim.x];
		probe.c.z = shrArr[shrID + 5*blockDim.x];

		probe.l	  = shrArr[shrID + 6*blockDim.x];
		probe.a   = shrArr[shrID + 7*blockDim.x];
		probe.k   = shrArr[shrID + 8*blockDim.x];

	return probe;
}

inline __device__ CNT_t shr2regCNT(float *shrArr) {
	return shrID2regCNT(threadIdx.x,shrArr);
}
inline __device__ void reg2shrCNT(CNT_t probe, float *shrArr, int i) {

		shrArr[i + 0*blockDim.x] = probe.r.x;
		shrArr[i + 1*blockDim.x] = probe.r.y;
		shrArr[i + 2*blockDim.x] = probe.r.z;

		shrArr[i + 3*blockDim.x] = probe.c.x;
		shrArr[i + 4*blockDim.x] = probe.c.y;
		shrArr[i + 5*blockDim.x] = probe.c.z;

		shrArr[i + 6*blockDim.x] = probe.l;
		shrArr[i + 7*blockDim.x] = probe.a;
		shrArr[i + 8*blockDim.x] = probe.k;

}
inline __device__ void reg2shrCNT(CNT_t probe, float *shrArr) {
	reg2shrCNT(probe, shrArr, threadIdx.x);
}
//--------------------------------------------------------------------------------------------------------------------
template<class Ti>
inline __device__ void shrID2regINC(Ti* probe, int shrID,float *shrArr) {

		probe->r.x = shrArr[shrID + 0*blockDim.x];
		probe->r.y = shrArr[shrID + 1*blockDim.x];
		probe->r.z = shrArr[shrID + 2*blockDim.x];

		probe->c.x = shrArr[shrID + 3*blockDim.x];
		probe->c.y = shrArr[shrID + 4*blockDim.x];
		probe->c.z = shrArr[shrID + 5*blockDim.x];

		probe->l	  = shrArr[shrID + 6*blockDim.x];
		probe->a   = shrArr[shrID + 7*blockDim.x];
		probe->k   = shrArr[shrID + 8*blockDim.x];
}
template<class Ti>
inline __device__ void shr2regINC(Ti* probe, float *shrArr) {
	return shrID2regINC(probe, threadIdx.x,shrArr);
}
template<class Ti>
inline __device__ void reg2shrINC(Ti* probe, float *shrArr, int i) {

		shrArr[i + 0*blockDim.x] = probe->r.x;
		shrArr[i + 1*blockDim.x] = probe->r.y;
		shrArr[i + 2*blockDim.x] = probe->r.z;

		shrArr[i + 3*blockDim.x] = probe->c.x;
		shrArr[i + 4*blockDim.x] = probe->c.y;
		shrArr[i + 5*blockDim.x] = probe->c.z;

		shrArr[i + 6*blockDim.x] = probe->l;
		shrArr[i + 7*blockDim.x] = probe->a;
		shrArr[i + 8*blockDim.x] = probe->k;

}
template<class Ti>
inline __device__ void reg2shrINC(Ti* probe, float *shrArr) {
	reg2shrINC(probe, shrArr, threadIdx.x);
}


inline __device__ void reg2shr(float* regArr, float *shrArr, short i, unsigned int nFields, short stride ) {

	if (i<stride) {
		for(unsigned int f=0; f<nFields; f++ )
			shrArr[i + f*stride] = regArr[f];
	} else printf("ERROR: An attempt to copy inclusion %i data past the available shared space %i\n",i,stride);

}

inline __device__ void reg2shr(float* regArr, float *shrArr, unsigned int nFields) {
	reg2shr(regArr, shrArr, threadIdx.x, nFields, blockDim.x );
}

inline __device__ void shr2reg(float* regArr, float *shrArr, short i, unsigned int nFields, short stride ) {

	if (i<stride) {
		for(unsigned int f=0; f<nFields; f++ )
			regArr[f] = shrArr[i + f*stride];
	} else printf("ERROR: An attempt to copy inclusion %i data past the available shared space %i\n",i,stride);

}

inline __device__ void shr2reg(float* regArr, float *shrArr, unsigned int nFields) {
	shr2reg(regArr, shrArr, threadIdx.x, nFields, blockDim.x );
}


inline __device__ void regClean(float* regArr, unsigned int nFields ) {
	for(unsigned int f=0; f<nFields; f++ )
		regArr[f] = DEF_EMPTY_VAL;
}

//--------------------------------------------------------------------------------------------------------------------
inline __device__ void shrClean(float *shrArr) {
#pragma unroll
		for (int i=0;i<9;i++)
			shrArr[threadIdx.x + i*blockDim.x] = -0.5f;
		__threadfence_block();
		__syncthreads();
}

inline __device__ int prepareDynamicShr(int vol, int *mask, int *devAddr, int *d_dmnAddr, short *d_dmnOcc, int *dmnExt) {
// prepare dynamic array to store cell occupancy and border addresses
	int glbPos = (threadIdx.x < vol ? key2glbID(threadIdx.x,mask,dmnExt) : -1);
	if (threadIdx.x < vol) {
		devAddr[threadIdx.x+0*vol] = (glbPos>0?d_dmnAddr[glbPos-1]:0) + d_dmnOcc[glbPos];	// Start posision (Addr + Occ)
		devAddr[threadIdx.x+1*vol] = d_dmnAddr[glbPos];										// End posision
	} 
	__syncthreads();
	__threadfence_block();

	return glbPos;
}

inline __device__ int prepareDynamicShr(int vol, int *mask, int *devAddr, int *d_dmnAddr, short *d_dmnOcc, int *shrCrtd, short *d_dmnCrtd, int *dmnExt) {
// prepare dynamic array to store cell occupancy and border addresses
	int glbPos = (threadIdx.x < vol ? key2glbID(threadIdx.x,mask,dmnExt) : -1);
	if (threadIdx.x < vol) {
		devAddr[threadIdx.x+0*vol] = (glbPos>0?d_dmnAddr[glbPos-1]:0) + d_dmnOcc[glbPos];	// Start posision (Addr + Occ)
		devAddr[threadIdx.x+1*vol] = d_dmnAddr[glbPos];										// End posision
		shrCrtd[threadIdx.x] = d_dmnCrtd[glbPos];
	} 
	__syncthreads();
	__threadfence_block();

	return glbPos;
}

//--------------------------------------------------------------------------------------------------------------------
// global memory read and write:
inline __device__ void  GlbRead(int nCNT, float *shrArr, int glbAddr, float *d_result, int numCNT) {

	if (threadIdx.x<nCNT) {
			#pragma unroll
			for (int i=0;i<9;i++)
				shrArr[threadIdx.x + i*blockDim.x] = d_result[glbAddr + threadIdx.x + i*numCNT];
			
	}
	__threadfence_block();
	__syncthreads();

}
inline __device__ void  GlbWrite(int nCNT, float *shrArr, int glbAddr, float *d_result, int numCNT) {

	if (threadIdx.x<nCNT) {
			#pragma unroll
			for (int i=0;i<9;i++)
				d_result[glbAddr+threadIdx.x+i*numCNT] = shrArr[threadIdx.x + i*blockDim.x];
	}
	__threadfence_block();
	__syncthreads();
}
inline __device__ void  GlbClean(int nCNT, float *shrArr, int glbAddr, float *d_result, int numCNT) {

	if (threadIdx.x<nCNT) {
			#pragma unroll
			for (int i=0;i<9;i++)
				d_result[glbAddr+threadIdx.x+i*numCNT] = 0;//-10000;
	}
	__threadfence();
	__syncthreads();

}
//--------------------------------------------------------------------------------------------------------------------
// safe global memory read and write:
inline __device__ void  safeGlbRead(int nCNT, int blockID, float *shrArr, int glbAddr, float *d_result,int *d_dmnLck, int numCNT) {
	if (threadIdx.x == 0 ) while(atomicCAS(&(d_dmnLck[blockID]),-1,blockIdx.x) != -1);
	__syncthreads();

	GlbRead(nCNT, shrArr, glbAddr, d_result,numCNT);

	if (threadIdx.x == 0 ) atomicExch(&d_dmnLck[blockID],-1);//d_dmnLck[blockID] = 1;
	__threadfence();
	__syncthreads();
}

inline __device__ void  safeGlbWrite(int nCNT,int blockID, float *shrArr, int glbAddr, float *d_result,int *d_dmnLck, int numCNT) {

	if (threadIdx.x == 0 ) while(atomicCAS(&d_dmnLck[blockID],-1,blockIdx.x) != -1);
	__syncthreads();

	GlbWrite(nCNT, shrArr, glbAddr, d_result,numCNT);

	if (threadIdx.x == 0 ) atomicExch(&d_dmnLck[blockID],-1);
	__threadfence();
	__syncthreads();
}
inline __device__ void  safeGlbClean(int nCNT,int blockID, float *shrArr, int glbAddr, float *d_result,int *d_dmnLck, int numCNT) {
			
	if (threadIdx.x == 0 ) while(atomicCAS(&(d_dmnLck[blockID]),-1,blockIdx.x)!=-1);
	__syncthreads();

	GlbClean(nCNT, shrArr, glbAddr, d_result,numCNT);

	if (threadIdx.x == 0 ) atomicExch(&d_dmnLck[blockID],-1);
	__threadfence();
	__syncthreads();
}
//--------------------------------------------------------------------------------------------------------------------
inline __device__ void writeGlbMem(int vol, int *mask, int *devAddr, int *shrAddr, short *d_Crtd, float *shrArr, float *d_result, int *dmnExt, int numCNT) {

	// bitonic sort ascanding:
	unsigned int dir1 = 1;
	bitonicSort(shrArr,dir1);
	__syncthreads();

	// prepare array to store beginings and ends of diffefent domain inclusions in shared memory:
	if (threadIdx.x < 2*vol)
		shrAddr[threadIdx.x] = 0;
	__threadfence_block();
	__syncthreads();

	// calculate obtained domain occupancies:
	int Ekey =(int) shrArr[threadIdx.x + 8*blockDim.x];
	int Bkey =(int) (threadIdx.x<(blockDim.x-1)?shrArr[threadIdx.x+1 + 8*blockDim.x]:-2);
	if (Ekey != Bkey) {
		int numE = (Ekey>=0?Ekey:-1); 
		int numB = (Bkey>=0?Bkey:-1);

		if (numB>=0) shrAddr[numB+0*vol] = threadIdx.x;
		if (numE>=0) shrAddr[numE+1*vol] = threadIdx.x;
	} 
	__threadfence_block();
	__syncthreads();

	// to account for the element in the begining
	if ((threadIdx.x == 0)&&(shrArr[0+8*blockDim.x]>-1)) {
		int num = shrArr[threadIdx.x + 8*blockDim.x];
		shrAddr[num] = -1; 
	}
	__syncthreads();
	
	//write results to global memory:
	int num = shrArr[threadIdx.x+8*blockDim.x];
	if (num >= 0) {
		int shrStart = shrAddr[num+0*vol];
		int countS   = threadIdx.x - shrStart;

		int glbStart = devAddr[num+0*vol];
		int glbEnd   = devAddr[num+1*vol];
		int countG   = glbEnd - glbStart;
		
		int pos = (countS<=countG?glbStart+countS-1:-1);
		if (pos>=0) {
#ifndef D_CHECK_RESULT
			for(int i=0;i<9;i++)
				d_result[pos+i*numCNT] = shrArr[threadIdx.x + i*blockDim.x];
#else
			for(int i=0;i<8;i++)
				d_result[pos+i*numCNT] = shrArr[threadIdx.x + i*blockDim.x];
				d_result[pos+8*numCNT] = key2glbID(shrArr[threadIdx.x + 8*blockDim.x],mask,dmnExt);
#endif
			__threadfence();
		}
	}
	__syncthreads();

	// write created CNTs per domain cell number:
	if (threadIdx.x < vol) {
		int countS   = shrAddr[threadIdx.x+1*vol] - shrAddr[threadIdx.x+0*vol];		// per domain End-Begin
		int countG   = devAddr[threadIdx.x+1*vol] - devAddr[threadIdx.x+0*vol];		// per domain global End-Begin
		int glbPos = key2glbID(threadIdx.x,mask,dmnExt);
		d_Crtd[glbPos] = (countS<countG?countS:countG);
	} 
	__threadfence();
	__syncthreads();

}

template<class Ti>
inline __device__ void reWriteGlbMem(Ti* probe, short selfINC, int* numInc, int devAddr, float *shrArr, float *d_result, int numINC) {
	__shared__ int posArr[BLOCK];

	// prefix scan:
	int num = (selfINC > 0 ? __powf(2,ceil(__log2f(selfINC))) : 0);

	if (threadIdx.x == 0) {
		*numInc = 0;
	}

	if (threadIdx.x<num)
		shrArr[threadIdx.x+8*blockDim.x] = probe->k;
	__syncthreads();
	if (num > 1)
		pscanNoB(num,numInc,&shrArr[8*blockDim.x],posArr);
	else if (threadIdx.x == 0) {
		*numInc = (probe->k >= 0);
		posArr[0] = 0;
	}

	// write CNTs to shared memory:
	if (threadIdx.x<num)
		shrArr[threadIdx.x+8*blockDim.x] = -1;	// clean shared mem
	__syncthreads();
	if ((threadIdx.x<num)&&(probe->k>=0))
		reg2shrINC(probe,shrArr,posArr[threadIdx.x]);
	__syncthreads();

	//write results to global memory:
	GlbWrite(*numInc, shrArr, devAddr, d_result,numINC);

}

inline __device__ void reWriteGlbMem(int selfCNT, CNT_t probe, int *numInc, int *devAddr, float *shrArr, float *d_result, int numCNT) {
	__shared__ int posArr[BLOCK];

	// prefix scan:
	int num = (selfCNT > 0 ? __powf(2,ceil(__log2f(selfCNT))) : 0);

	if (threadIdx.x == 0) {
		numInc[0] = 0;
	}

	if (threadIdx.x<num) 
		shrArr[threadIdx.x+8*blockDim.x] = probe.k;
	__syncthreads();
	if (num > 1)
		pscanNoB(num,&numInc[0],&shrArr[8*blockDim.x],posArr);
	else if (threadIdx.x == 0) {
		numInc[0] = (probe.k >= 0);
		posArr[0] = 0;
	}

	// write CNTs to shared memory:
	if (threadIdx.x<num) 
		shrArr[threadIdx.x+8*blockDim.x] = -1;	// clean shared mem
	__syncthreads();
	if ((threadIdx.x<num)&&(probe.k>=0))
		reg2shrCNT(probe,shrArr,posArr[threadIdx.x]);
	__syncthreads();
	
	//write results to global memory:
	GlbWrite(numInc[0], shrArr, devAddr[0], d_result,numCNT);

}

inline __device__ int testWriteGlbMem(int selfCNT, CNT_t probe, int *numInc, int *devAddr, float *shrArr, float *d_result, int numCNT) {
	__shared__ int posArr[BLOCK];

	// prefix scan:
	int num = (selfCNT > 0 ? __powf(2,ceil(__log2f(selfCNT))) : 0);

	if (threadIdx.x == 0) {
		*numInc = 0;
	}

	if (threadIdx.x<num)
		shrArr[threadIdx.x+8*blockDim.x] = probe.k;
	__syncthreads();
	if (num > 1)
		pscanNoB(num,numInc,&shrArr[8*blockDim.x],posArr);
	else if (threadIdx.x == 0) {
		*numInc = (probe.k >= 0);
		posArr[0] = 0;
	}

	// write CNTs to shared memory:
	if (threadIdx.x<num)
		shrArr[threadIdx.x+8*blockDim.x] = -1;	// clean shared mem
	__syncthreads();
	if ((threadIdx.x<num)&&(probe.k>=0))
		reg2shrCNT(probe,shrArr,posArr[threadIdx.x]);
	__syncthreads();

	//write results to global memory:
	//GlbWrite(*numInc, shrArr, devAddr[0], d_result,numCNT);
	__syncthreads();
	return *numInc;
}
//--------------------------------------------------------------------------------------------------------------------
// get displacements:
inline __device__ char3 stride2dspl(char3 stride, int baseID, int *dmnExt) { 

	int x1 = (baseID%dmnExt[0])				+ stride.x;
	int y1 = (baseID/dmnExt[0])%dmnExt[1]	+ stride.y;
	int z1 = (baseID/dmnExt[0])/dmnExt[1]	+ stride.z;

	char x = 0;
	if (x1<0) x = -1;
		else if (x1>=dmnExt[0]) x = 1;

	char y = 0;
	if (y1<0) y = -1;
		else if (y1>=dmnExt[1]) y = 1;

	char z = 0;
	if (z1<0) z = -1;
		else if (z1>=dmnExt[2]) z = 1;


	return make_char3(x,y,z);
}
// translate on load:
inline __device__ void tarnsCNT(int nCNT, char3 dspl, float *shrArr) {

	if (threadIdx.x < nCNT) {
		if (dspl.x != 0) shrArr[threadIdx.x + 0*blockDim.x] += dspl.x;
		if (dspl.y != 0) shrArr[threadIdx.x + 1*blockDim.x] += dspl.y;
		if (dspl.z != 0) shrArr[threadIdx.x + 2*blockDim.x] += dspl.z;
	}
	__threadfence_block();
	__syncthreads();

}
inline __device__ void tarnsCNT(int nCNT, float3 dspl, float *shrArr) {

	if (threadIdx.x < nCNT) {
		if (dspl.x != 0) shrArr[threadIdx.x + 0*blockDim.x] += dspl.x;
		if (dspl.y != 0) shrArr[threadIdx.x + 1*blockDim.x] += dspl.y;
		if (dspl.z != 0) shrArr[threadIdx.x + 2*blockDim.x] += dspl.z;
	}
	__threadfence_block();
	__syncthreads();

}
